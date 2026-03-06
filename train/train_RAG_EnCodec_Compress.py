"""
RAG-conditioned EnCodec + bGPT audio compression.

Pipeline:
  raw audio
    → EnCodec RAG retriever  (finds similar audio from corpus)
    → EnCodec tokenizer      (frozen, encodes audio → RVQ discrete codes → bytes)
    → EnCodec encoder + LoRA transformer adapter
                             (encodes retrieved audio → RAG context prefix for bGPT)
    → bGPT causal LM         (frozen hierarchical byte model, conditioned on prefix)
    → arithmetic coding      (compress using bGPT's byte probabilities)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from glob import glob

from transformers import (
    EncodecModel,
    AutoProcessor,
    GPT2Config,
    GPT2Model,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

from bgpt.utils import bGPTLMHeadModel, PatchLevelDecoder, ByteLevelDecoder
from bgpt.config import PATCH_SIZE, HIDDEN_SIZE, PATCH_NUM_LAYERS, BYTE_NUM_LAYERS, PATCH_LENGTH
from utils.encodec_rag import EnCodecRAGRetriever, EnCodecRAGConfig


# ==================== Configuration ====================
class Config:
    ENCODEC_MODEL = "facebook/encodec_16khz"
    ENCODEC_SAMPLE_RATE = 16000
    ENCODEC_DIM = 128          # EnCodec encoder output channels
    ENCODEC_BANDWIDTH = 6.0    # kbps; controls number of active codebooks

    # LoRA adapter (small GPT2 sitting on top of frozen EnCodec features)
    ADAPTER_N_EMBD = 128       # matches ENCODEC_DIM
    ADAPTER_N_HEAD = 4
    ADAPTER_N_LAYER = 2
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    # bGPT
    BGPT_CHECKPOINT_AUDIO = "./pretrained/bgpt/weights-audio.pth"
    BGPT_PATCH_SIZE = PATCH_SIZE      # 16 bytes per patch
    BGPT_HIDDEN_SIZE = HIDDEN_SIZE    # 768
    BGPT_PATCH_LENGTH = PATCH_LENGTH  # 1024
    BGPT_PATCH_NUM_LAYERS = PATCH_NUM_LAYERS
    BGPT_BYTE_NUM_LAYERS = BYTE_NUM_LAYERS

    # Compression
    PRECISION = 64
    PREFIX_LENGTH = 1

    # Training
    SAMPLE_RATE = ENCODEC_SAMPLE_RATE
    CHUNK_DURATION = 1.0      # seconds per audio chunk fed to the model
    MAX_PATCHES = 512         # max patches per audio chunk for bGPT
    TOP_K_RETRIEVAL = 1

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Audio Tokenizer ====================
def encodec_audio_to_bytes(
    encodec_model: EncodecModel,
    waveform: torch.Tensor,
    bandwidth: float = Config.ENCODEC_BANDWIDTH,
) -> List[int]:
    """
    Tokenize audio with EnCodec and serialize RVQ codes to a flat byte list.

    Each RVQ code is in [0, 1023], stored as a little-endian uint16 (2 bytes).
    Codes are written codebook-interleaved: for each time frame t, write codes
    for all codebooks in order, then advance to frame t+1.

    waveform: [1, T]  (single channel, already at ENCODEC_SAMPLE_RATE)
    returns: flat list of ints in [0, 255]
    """
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [1, 1, T]
    waveform = waveform.to(next(encodec_model.parameters()).device)

    with torch.no_grad():
        encoded = encodec_model.encode(waveform, bandwidth=bandwidth)
    codes = encoded.audio_codes  # [1, num_codebooks, T_frames]
    codes = codes.squeeze(0).cpu().numpy()  # [num_codebooks, T_frames]

    # Flatten codebook-interleaved: [T_frames, num_codebooks] → flat
    codes_interleaved = codes.T.flatten()  # [T_frames * num_codebooks]
    # Each code is uint16 (2 bytes, little-endian)
    byte_data = codes_interleaved.astype(np.uint16).tobytes()
    return list(byte_data)


def pad_bytes_to_patches(bytes_list: List[int], patch_size: int = Config.BGPT_PATCH_SIZE) -> List[int]:
    """Pad byte list so its length is a multiple of patch_size (using value 256 as pad)."""
    remainder = len(bytes_list) % patch_size
    if remainder != 0:
        bytes_list = bytes_list + [256] * (patch_size - remainder)
    return bytes_list


# ==================== ConditionedPatchLevelDecoder ====================
class ConditionedPatchLevelDecoder(nn.Module):
    """
    Wraps bGPT's PatchLevelDecoder to accept optional prefix embeddings.
    Prefix embeddings are prepended to the patch sequence before GPT2 attention.
    """

    def __init__(self, patch_level_decoder: PatchLevelDecoder):
        super().__init__()
        self.patch_embedding = patch_level_decoder.patch_embedding
        self.base = patch_level_decoder.base  # GPT2Model

    def forward(
        self,
        patches: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        patches:       [B, N_patches, PATCH_SIZE]  (int64)
        masks:         [B, N_patches]               (0/1)
        prefix_embeds: [B, N_prefix, HIDDEN_SIZE]   (float)
        returns: last_hidden_state [B, N_prefix+N_patches, HIDDEN_SIZE]
        """
        # Embed patches: one-hot → linear projection
        patch_oh = torch.nn.functional.one_hot(patches, num_classes=257).to(
            dtype=self.patch_embedding.weight.dtype,
            device=self.patch_embedding.weight.device,
        )
        patch_oh = patch_oh.reshape(patches.shape[0], -1, PATCH_SIZE * 257)
        patch_embeds = self.patch_embedding(patch_oh)  # [B, N, H]

        if prefix_embeds is not None:
            patch_embeds = torch.cat([prefix_embeds, patch_embeds], dim=1)
            if masks is not None:
                prefix_mask = torch.ones(
                    masks.shape[0], prefix_embeds.shape[1],
                    device=masks.device, dtype=masks.dtype,
                )
                masks = torch.cat([prefix_mask, masks], dim=1)

        if masks is None:
            return self.base(inputs_embeds=patch_embeds)
        else:
            return self.base(inputs_embeds=patch_embeds, attention_mask=masks)


# ==================== Main Model ====================
class RAGEnCodecBGPT(nn.Module):
    """
    RAG-conditioned audio compressor.

    Encoder path (trained):
      retrieved audio → frozen EnCodec encoder → LoRA GPT2 adapter → context bridge
                     → prefix embeddings for bGPT

    Decoder path (frozen):
      target audio bytes → bGPT patches + RAG prefix → byte-level logits → arithmetic coding
    """

    def __init__(
        self,
        encodec_model_name: str = Config.ENCODEC_MODEL,
        bgpt_checkpoint: str = Config.BGPT_CHECKPOINT_AUDIO,
        lora_r: int = Config.LORA_R,
        lora_alpha: int = Config.LORA_ALPHA,
        lora_dropout: float = Config.LORA_DROPOUT,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype

        # --- Frozen EnCodec (tokenizer + feature extractor) ---
        self.encodec = EncodecModel.from_pretrained(encodec_model_name)
        self.encodec.eval()
        for p in self.encodec.parameters():
            p.requires_grad = False

        # --- LoRA transformer adapter on top of EnCodec encoder features ---
        adapter_cfg = GPT2Config(
            n_embd=Config.ADAPTER_N_EMBD,
            n_head=Config.ADAPTER_N_HEAD,
            n_layer=Config.ADAPTER_N_LAYER,
            vocab_size=1,
            n_positions=4096,
        )
        base_adapter = GPT2Model(adapter_cfg)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"],
            bias="none",
        )
        self.lora_adapter = get_peft_model(base_adapter, lora_cfg)

        # --- Context bridge: adapter dim → bGPT hidden size ---
        self.context_bridge = nn.Linear(
            Config.ADAPTER_N_EMBD, Config.BGPT_HIDDEN_SIZE, bias=False, dtype=dtype
        )
        nn.init.normal_(self.context_bridge.weight, std=0.02)

        # --- Frozen bGPT ---
        self.bgpt = _load_bgpt(bgpt_checkpoint)
        for p in self.bgpt.parameters():
            p.requires_grad = False
        self.bgpt.eval()

        # --- Conditioned patch decoder (wraps bGPT's patch_level_decoder) ---
        self.conditioned_patch_decoder = ConditionedPatchLevelDecoder(
            self.bgpt.patch_level_decoder
        )

    def encode_context(self, retrieved_waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode retrieved audio into RAG context prefix embeddings.
        retrieved_waveform: [B, 1, T]
        returns: [B, T_enc, HIDDEN_SIZE]
        """
        retrieved_waveform = retrieved_waveform.to(next(self.encodec.parameters()).device)
        with torch.no_grad():
            enc_out = self.encodec.encoder(retrieved_waveform)  # [B, D, T_enc]
        enc_out = enc_out.transpose(1, 2)  # [B, T_enc, D]

        # LoRA adapter expects inputs_embeds, not input_ids
        ctx_feats = self.lora_adapter(inputs_embeds=enc_out.to(self.dtype)).last_hidden_state
        ctx_embeds = self.context_bridge(ctx_feats)  # [B, T_enc, HIDDEN_SIZE]
        return ctx_embeds

    def forward(
        self,
        audio_patches: torch.Tensor,
        audio_masks: torch.Tensor,
        retrieved_waveform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        audio_patches:      [B, N_patches * PATCH_SIZE]  flat byte tokens (int64, values 0-256)
        audio_masks:        [B, N_patches]                patch-level attention mask
        retrieved_waveform: [B, 1, T]                     retrieved similar audio

        returns: dict with 'loss' and 'logits'
        """
        B = audio_patches.shape[0]
        device = audio_patches.device

        # 1. Encode retrieved audio → RAG prefix [B, T_enc, H]
        ctx_embeds = self.encode_context(retrieved_waveform)

        # Pool encoder time dim to a fixed number of prefix patches
        # Use a single mean-pooled patch for simplicity; can be extended to multiple
        prefix_embeds = ctx_embeds.mean(dim=1, keepdim=True)  # [B, 1, H]
        n_prefix = prefix_embeds.shape[1]

        # 2. Reshape audio patches for patch-level processing
        audio_patches_2d = audio_patches.reshape(B, -1, PATCH_SIZE)  # [B, N, PATCH_SIZE]

        # 3. Run conditioned patch-level decoder (injects RAG prefix)
        out = self.conditioned_patch_decoder(
            patches=audio_patches_2d,
            masks=audio_masks,
            prefix_embeds=prefix_embeds.to(device),
        )
        # last_hidden_state: [B, n_prefix + N_patches, H]
        encoded_patches = out["last_hidden_state"]

        # 4. Drop prefix positions from encoded patches (causal: prefix already influenced audio)
        encoded_patches = encoded_patches[:, n_prefix:, :]  # [B, N_patches, H]

        # 5. Byte-level decoding — replicate bGPT's masking logic
        left_shift_masks = audio_masks * (audio_masks.flip(1).cumsum(1).flip(1) > 1)
        audio_masks_modified = audio_masks.clone()
        audio_masks_modified[:, 0] = 0

        encoded_patches_valid = encoded_patches[left_shift_masks == 1]
        target_patches_valid = audio_patches_2d[audio_masks_modified == 1]

        output = self.bgpt.byte_level_decoder(encoded_patches_valid, target_patches_valid)
        return {"loss": output.loss, "logits": output.logits}

    def get_logits_for_compression(
        self,
        audio_patches: torch.Tensor,
        audio_masks: torch.Tensor,
        retrieved_waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns byte-level logits for arithmetic coding.
        Shape: [N_valid_patches, PATCH_SIZE+1, 257]
        """
        with torch.inference_mode():
            out = self.forward(audio_patches, audio_masks, retrieved_waveform)
        return out["logits"]


# ==================== Helper: load bGPT ====================
def _load_bgpt(checkpoint_path: str) -> bGPTLMHeadModel:
    patch_config = GPT2Config(
        num_hidden_layers=Config.BGPT_PATCH_NUM_LAYERS,
        max_length=Config.BGPT_PATCH_LENGTH,
        max_position_embeddings=Config.BGPT_PATCH_LENGTH,
        hidden_size=Config.BGPT_HIDDEN_SIZE,
        n_head=Config.BGPT_HIDDEN_SIZE // 64,
        vocab_size=1,
    )
    byte_config = GPT2Config(
        num_hidden_layers=Config.BGPT_BYTE_NUM_LAYERS,
        max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1,
        hidden_size=Config.BGPT_HIDDEN_SIZE,
        n_head=Config.BGPT_HIDDEN_SIZE // 64,
        vocab_size=257,
    )
    model = bGPTLMHeadModel(patch_config, byte_config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded bGPT from {checkpoint_path}")
    else:
        print(f"Warning: bGPT checkpoint not found at '{checkpoint_path}', using random weights")
    model.eval()
    return model


# ==================== Dataset ====================
class AudioDataset(Dataset):
    """
    Loads WAV files, chunks them, tokenizes with EnCodec, and returns
    (byte_patches, patch_masks, retrieved_waveform) triples.
    """

    def __init__(
        self,
        file_paths: List[str],
        encodec_model: EncodecModel,
        rag_retriever: EnCodecRAGRetriever,
        chunk_duration: float = Config.CHUNK_DURATION,
        max_patches: int = Config.MAX_PATCHES,
        top_k: int = Config.TOP_K_RETRIEVAL,
    ):
        self.file_paths = file_paths
        self.encodec = encodec_model
        self.retriever = rag_retriever
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(Config.ENCODEC_SAMPLE_RATE * chunk_duration)
        self.max_patches = max_patches
        self.top_k = top_k

        # Pre-enumerate all (file, chunk_start) pairs
        self.samples: List[Tuple[str, int]] = []
        for path in file_paths:
            info = torchaudio.info(path)
            sr = info.sample_rate
            n_frames = info.num_frames
            # Number of complete chunks
            n_chunks = n_frames // int(sr * chunk_duration)
            for i in range(max(1, n_chunks)):
                self.samples.append((path, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, chunk_idx = self.samples[idx]

        signal, sr = torchaudio.load(path)
        if sr != Config.ENCODEC_SAMPLE_RATE:
            signal = torchaudio.functional.resample(signal, sr, Config.ENCODEC_SAMPLE_RATE)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        # Extract chunk
        start = chunk_idx * self.chunk_samples
        chunk = signal[:, start: start + self.chunk_samples]
        if chunk.shape[-1] < self.chunk_samples:
            # Pad short last chunk
            pad = self.chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        # Tokenize chunk to bytes
        bytes_list = encodec_audio_to_bytes(self.encodec, chunk)
        bytes_list = pad_bytes_to_patches(bytes_list)

        # Truncate to max_patches
        max_bytes = self.max_patches * PATCH_SIZE
        bytes_list = bytes_list[:max_bytes]
        n_patches = len(bytes_list) // PATCH_SIZE

        # Build patch mask
        patch_mask = [1] * n_patches
        # Pad to max_patches if needed
        if n_patches < self.max_patches:
            pad_patches = self.max_patches - n_patches
            bytes_list = bytes_list + [256] * (pad_patches * PATCH_SIZE)
            patch_mask = patch_mask + [0] * pad_patches

        # Retrieve similar audio
        results = self.retriever.retrieve(chunk, k=self.top_k)
        if results:
            retrieved_path = results[0]["path"]
            retrieved_signal, rsr = torchaudio.load(retrieved_path)
            if rsr != Config.ENCODEC_SAMPLE_RATE:
                retrieved_signal = torchaudio.functional.resample(
                    retrieved_signal, rsr, Config.ENCODEC_SAMPLE_RATE
                )
            if retrieved_signal.shape[0] > 1:
                retrieved_signal = retrieved_signal.mean(dim=0, keepdim=True)
            # Use same duration as chunk
            if retrieved_signal.shape[-1] < self.chunk_samples:
                pad = self.chunk_samples - retrieved_signal.shape[-1]
                retrieved_signal = torch.nn.functional.pad(retrieved_signal, (0, pad))
            else:
                retrieved_signal = retrieved_signal[:, : self.chunk_samples]
        else:
            # Fall back to the chunk itself if retrieval fails
            retrieved_signal = chunk

        return {
            "audio_patches": torch.tensor(bytes_list, dtype=torch.long),
            "audio_masks": torch.tensor(patch_mask, dtype=torch.long),
            "retrieved_waveform": retrieved_signal,
        }


# ==================== Compression (inference) ====================
def compress_audio_file(
    model: RAGEnCodecBGPT,
    wav_path: str,
    rag_retriever: EnCodecRAGRetriever,
    output_path: str,
    device: str = Config.DEVICE,
    chunk_duration: float = Config.CHUNK_DURATION,
    precision: int = Config.PRECISION,
    prefix_length: int = Config.PREFIX_LENGTH,
):
    """
    Compress a WAV file using RAGEnCodecBGPT + arithmetic coding.
    Saves compressed output to output_path.
    """
    from arithmetic_coder import ac_utils, arithmetic_coder
    from evaluation.LLMCompress import write_padded_bytes, Metric

    model.eval()
    model.to(device)
    chunk_samples = int(Config.ENCODEC_SAMPLE_RATE * chunk_duration)

    signal, sr = torchaudio.load(wav_path)
    if sr != Config.ENCODEC_SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, Config.ENCODEC_SAMPLE_RATE)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    total_metric = Metric()
    all_compressed = []

    n_chunks = max(1, signal.shape[-1] // chunk_samples)
    for i in range(n_chunks):
        chunk = signal[:, i * chunk_samples: (i + 1) * chunk_samples]
        if chunk.shape[-1] < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

        # Tokenize
        bytes_list = encodec_audio_to_bytes(model.encodec, chunk)
        bytes_list = pad_bytes_to_patches(bytes_list)
        n_patches = len(bytes_list) // PATCH_SIZE

        audio_patches = torch.tensor(bytes_list, dtype=torch.long).unsqueeze(0).to(device)
        audio_masks = torch.ones(1, n_patches, dtype=torch.long, device=device)

        # Retrieve
        results = rag_retriever.retrieve(chunk, k=1)
        if results:
            ret_signal, rsr = torchaudio.load(results[0]["path"])
            if rsr != Config.ENCODEC_SAMPLE_RATE:
                ret_signal = torchaudio.functional.resample(ret_signal, rsr, Config.ENCODEC_SAMPLE_RATE)
            if ret_signal.shape[0] > 1:
                ret_signal = ret_signal.mean(dim=0, keepdim=True)
            if ret_signal.shape[-1] < chunk_samples:
                ret_signal = torch.nn.functional.pad(ret_signal, (0, chunk_samples - ret_signal.shape[-1]))
            else:
                ret_signal = ret_signal[:, :chunk_samples]
        else:
            ret_signal = chunk
        retrieved_waveform = ret_signal.unsqueeze(0).to(device)

        # Get logits
        logits = model.get_logits_for_compression(audio_patches, audio_masks, retrieved_waveform)
        # logits: [N_valid, PATCH_SIZE+1, 257]
        logits = logits[:, :-1, :]  # drop last byte position in each patch
        logits = logits.reshape(1, -1, 257)  # [1, N*PATCH_SIZE, 257]

        # Prepare input_ids for arithmetic coding
        # Input ids = bytes excluding BOS patch, with leading dummy token
        input_ids = audio_patches[:, PATCH_SIZE:-PATCH_SIZE] if n_patches > 2 else audio_patches
        input_ids = torch.cat([torch.tensor([[256]], device=device), input_ids], dim=1)

        # Arithmetic encode
        output_bits = []
        encoder = arithmetic_coder.Encoder(
            base=2, precision=precision, output_fn=output_bits.append
        )
        target = input_ids[:, prefix_length:].detach().cpu().numpy().reshape(-1)
        probs_np = logits[:, prefix_length - 1:, :].softmax(dim=-1).detach().cpu().numpy().squeeze(0)

        for symbol, prob in zip(target, probs_np):
            encoder.encode(
                ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32), symbol
            )
        encoder.terminate()

        compressed_bits = "".join(map(str, output_bits))
        compressed_bytes, num_padded_bits = ac_utils.bits_to_bytes(compressed_bits)
        total_metric.accumulate(len(compressed_bytes), len(target))
        all_compressed.append((compressed_bytes, num_padded_bits, len(target)))

    compress_rate, compress_ratio = total_metric.compute_ratio()
    print(f"Compression ratio: {compress_ratio:.4f}  |  Rate: {compress_rate:.4f} bpb")

    # Save all chunks to output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    import pickle
    with open(output_path, "wb") as f:
        pickle.dump(all_compressed, f)
    print(f"Saved compressed output to {output_path}")
    return compress_ratio


# ==================== Training ====================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train RAG-EnCodec-bGPT audio compressor")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory of WAV training files")
    parser.add_argument("--retriever_path", type=str, default="retriever_cache/encodec",
                        help="Path to save/load FAISS index")
    parser.add_argument("--bgpt_checkpoint", type=str,
                        default=Config.BGPT_CHECKPOINT_AUDIO)
    parser.add_argument("--output_dir", type=str, default="./rag_encodec_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = Config.DEVICE

    # Load EnCodec (used as frozen tokenizer and feature extractor)
    encodec = EncodecModel.from_pretrained(Config.ENCODEC_MODEL)
    encodec.to(device).eval()
    for p in encodec.parameters():
        p.requires_grad = False

    # Build / load RAG index
    retriever = EnCodecRAGRetriever(persist_path=args.retriever_path)
    audio_files = glob(os.path.join(args.audio_dir, "**/*.wav"), recursive=True)
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {args.audio_dir}")

    if retriever.index.ntotal == 0:
        retriever.index_audio_files(audio_files)

    # Build model
    model = RAGEnCodecBGPT(
        encodec_model_name=Config.ENCODEC_MODEL,
        bgpt_checkpoint=args.bgpt_checkpoint,
    )
    model.to(device)
    print("Trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}: {p.numel():,}")

    # Dataset
    dataset = AudioDataset(
        file_paths=audio_files,
        encodec_model=encodec,
        rag_retriever=retriever,
    )
    print(f"Dataset: {len(dataset)} chunks from {len(audio_files)} files")

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        logging_steps=20,
        save_steps=500,
        remove_unused_columns=False,
    )

    def collate_fn(batch):
        return {
            "audio_patches": torch.stack([b["audio_patches"] for b in batch]),
            "audio_masks": torch.stack([b["audio_masks"] for b in batch]),
            "retrieved_waveform": torch.stack([b["retrieved_waveform"] for b in batch]),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
