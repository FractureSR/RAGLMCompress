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

import io
import os
import struct
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torchaudio
import librosa
import wave
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
    ENCODEC_MODEL = "facebook/encodec_24khz"
    ENCODEC_SAMPLE_RATE = 24000
    ENCODEC_DIM = 128          # EnCodec encoder output channels
    ENCODEC_BANDWIDTH = 3.0    # kbps; controls number of active codebooks

    # LoRA adapter (small GPT2 sitting on top of frozen EnCodec features)
    ADAPTER_N_EMBD = 128       # matches ENCODEC_DIM
    ADAPTER_N_HEAD = 4
    ADAPTER_N_LAYER = 2
    LORA_R = 32
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    # bGPT
    BGPT_CHECKPOINT_AUDIO = "./pretrained/bgpt/weights-audio.pth"
    BGPT_PATCH_SIZE = PATCH_SIZE      # 16 bytes per patch
    BGPT_HIDDEN_SIZE = HIDDEN_SIZE    # 768
    BGPT_PATCH_LENGTH = 512  # 1024
    BGPT_PATCH_NUM_LAYERS = PATCH_NUM_LAYERS
    BGPT_BYTE_NUM_LAYERS = BYTE_NUM_LAYERS

    # Compression
    PRECISION = 64
    PREFIX_LENGTH = 1
    NUM_PREFIX_TOKENS = 64
    PREFIX_POOL_HEADS = 4

    # Training
    SAMPLE_RATE = ENCODEC_SAMPLE_RATE
    CHUNK_DURATION = 0.1      # requested seconds per audio chunk, capped by bGPT byte budget
    MAX_PATCHES = BGPT_PATCH_LENGTH - PREFIX_LENGTH
    TOP_K_RETRIEVAL = 5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Audio Tokenizer ====================
def _wav_frames_to_mono_tensor(
    frames: bytes,
    n_channels: int,
    sampwidth: int,
    sample_rate: int,
) -> torch.Tensor:
    """Convert WAV PCM frames into a mono float tensor shaped [1, T]."""
    dtype_map = {
        1: np.uint8,
        2: np.int16,
        4: np.int32,
    }
    if sampwidth not in dtype_map:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    samples = np.frombuffer(frames, dtype=dtype_map[sampwidth])
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if sampwidth == 1:
        samples = (samples.astype(np.float32) - 128.0) / 128.0
    else:
        scale = float(2 ** (8 * sampwidth - 1))
        samples = samples.astype(np.float32) / scale

    waveform = torch.from_numpy(samples).unsqueeze(0)
    if sample_rate != Config.ENCODEC_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, Config.ENCODEC_SAMPLE_RATE
        )
    return waveform


def _prepare_pcm_audio_bytes(
    pcm_bytes: bytes,
    max_patches: int = Config.MAX_PATCHES,
) -> Tuple[List[int], List[int]]:
    """
    Format raw PCM bytes for bGPT: one format patch, PCM payload patches, then one end patch.
    This keeps the compression target lossless while avoiding WAV-container header bytes.
    """
    fmt_bytes = list(bytearray("pcm", "utf-8"))[:PATCH_SIZE]
    bos_patch = fmt_bytes + [256] * (PATCH_SIZE - len(fmt_bytes))
    payload = list(pcm_bytes)
    if len(payload) % PATCH_SIZE != 0:
        payload += [256] * (PATCH_SIZE - len(payload) % PATCH_SIZE)

    bytes_list = bos_patch + payload + [256] * PATCH_SIZE
    valid_patches = len(bytes_list) // PATCH_SIZE
    if valid_patches > max_patches:
        raise ValueError(
            f"Chunk requires {valid_patches} patches, exceeds bGPT limit of {max_patches}. "
            "Use a smaller audio chunk."
        )

    patch_mask = [1] * valid_patches + [0] * (max_patches - valid_patches)
    bytes_list += [256] * ((max_patches - valid_patches) * PATCH_SIZE)
    return bytes_list, patch_mask


def _chunk_frame_budget(
    params: wave._wave_params,
    chunk_duration: float,
    max_patches: int = Config.MAX_PATCHES,
) -> int:
    """
    Compute a safe frame budget per chunk so the raw PCM payload fits within bGPT.
    """
    requested_frames = max(1, int(params.framerate * chunk_duration))
    max_payload_bytes = (max_patches - 2) * PATCH_SIZE
    bytes_per_frame = params.nchannels * params.sampwidth
    max_frames_by_bytes = max(1, max_payload_bytes // bytes_per_frame)
    return min(requested_frames, max_frames_by_bytes)


def load_wav_chunk(
    path: str,
    chunk_idx: int,
    chunk_duration: float,
    max_patches: int = Config.MAX_PATCHES,
) -> Dict[str, Any]:
    """
    Load a single chunk as exact PCM payload bytes for bGPT and waveform for EnCodec RAG.
    """
    with wave.open(path, "rb") as wav_in:
        params = wav_in.getparams()
        chunk_frames = _chunk_frame_budget(
            params,
            chunk_duration,
            max_patches=max_patches,
        )
        wav_in.setpos(chunk_idx * chunk_frames)
        frames = wav_in.readframes(chunk_frames)

    waveform = _wav_frames_to_mono_tensor(
        frames,
        n_channels=params.nchannels,
        sampwidth=params.sampwidth,
        sample_rate=params.framerate,
    )
    bytes_list, patch_mask = _prepare_pcm_audio_bytes(
        frames,
        max_patches=max_patches,
    )
    return {
        "bytes_list": bytes_list,
        "patch_mask": patch_mask,
        "waveform": waveform,
        "params": params,
        "pcm_num_bytes": len(frames),
    }


def _load_retrieved_waveform(path: str, target_len: int) -> torch.Tensor:
    retrieved_signal, rsr = librosa.load(path, sr=None, mono=False)
    retrieved_signal = torch.from_numpy(retrieved_signal).float()
    if retrieved_signal.ndim == 1:
        retrieved_signal = retrieved_signal.unsqueeze(0)
    if rsr != Config.ENCODEC_SAMPLE_RATE:
        retrieved_signal = torchaudio.functional.resample(
            retrieved_signal, rsr, Config.ENCODEC_SAMPLE_RATE
        )
    if retrieved_signal.shape[0] > 1:
        retrieved_signal = retrieved_signal.mean(dim=0, keepdim=True)
    if retrieved_signal.shape[-1] < target_len:
        pad = target_len - retrieved_signal.shape[-1]
        retrieved_signal = torch.nn.functional.pad(retrieved_signal, (0, pad))
    else:
        retrieved_signal = retrieved_signal[:, :target_len]
    return retrieved_signal


def _load_audio_file_waveform(path: str) -> torch.Tensor:
    signal, sr = librosa.load(path, sr=None, mono=False)
    signal = torch.from_numpy(signal).float()
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    if sr != Config.ENCODEC_SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, Config.ENCODEC_SAMPLE_RATE)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal


def _aggregate_retrieved_waveforms(
    results: List[Dict[str, Any]],
    fallback_waveform: torch.Tensor,
) -> Tuple[torch.Tensor, List[str]]:
    if not results:
        return fallback_waveform, []

    target_len = fallback_waveform.shape[-1]
    retrieved_signals = []
    retrieved_paths: List[str] = []
    for result in results:
        try:
            retrieved_signals.append(_load_retrieved_waveform(result["path"], target_len))
            retrieved_paths.append(result["path"])
        except Exception:
            continue

    if not retrieved_signals:
        return fallback_waveform, []

    stacked = torch.stack(retrieved_signals, dim=0)
    return stacked.mean(dim=0), retrieved_paths


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
                    masks.shape[0],
                    prefix_embeds.shape[1],
                    device=masks.device,
                    dtype=masks.dtype,
                )
                masks = torch.cat([prefix_mask, masks], dim=1)

        max_len = self.base.config.max_position_embeddings
        patch_embeds = patch_embeds[:, :max_len, :]
        if masks is not None:
            masks = masks[:, :max_len]

        if masks is None:
            return self.base(inputs_embeds=patch_embeds)
        else:
            return self.base(inputs_embeds=patch_embeds, attention_mask=masks)

# ==================== Main Model ====================
class RAGEnCodecBGPT(nn.Module):
    """
    RAG-conditioned lossless audio compressor over PCM byte patches.

    Retrieved audio is encoded with EnCodec into trainable prefix embeddings,
    while the compression target remains the exact PCM byte stream.
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

        self.encodec = EncodecModel.from_pretrained(encodec_model_name)
        self.encodec.eval()
        for p in self.encodec.parameters():
            p.requires_grad = False

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

        self.context_bridge = nn.Linear(
            Config.ADAPTER_N_EMBD, Config.BGPT_HIDDEN_SIZE, bias=False, dtype=dtype
        )
        nn.init.normal_(self.context_bridge.weight, std=0.02)

        self.prefix_norm = nn.LayerNorm(Config.BGPT_HIDDEN_SIZE, dtype=dtype)
        self.sep_embed = nn.Parameter(
            torch.zeros(1, 1, Config.BGPT_HIDDEN_SIZE, dtype=dtype)
        )
        nn.init.normal_(self.sep_embed, std=0.02)

        self.bgpt = _load_bgpt(bgpt_checkpoint)
        for p in self.bgpt.parameters():
            p.requires_grad = False

        # Add LoRA to both bGPT decoders so they adapt to audio bytes
        bgpt_lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"],
            bias="none",
        )
        self.bgpt.patch_level_decoder.base = get_peft_model(
            self.bgpt.patch_level_decoder.base, bgpt_lora_cfg
        )
        self.bgpt.byte_level_decoder.base = get_peft_model(
            self.bgpt.byte_level_decoder.base, bgpt_lora_cfg
        )

        self.conditioned_patch_decoder = ConditionedPatchLevelDecoder(
            self.bgpt.patch_level_decoder
        )

    def encode_context(self, retrieved_waveform: torch.Tensor) -> torch.Tensor:
        device = retrieved_waveform.device
        retrieved_waveform = retrieved_waveform.to(device)

        with torch.no_grad():
            enc_out = self.encodec.encoder(retrieved_waveform)

        enc_out = enc_out.transpose(1, 2)
        ctx_feats = self.lora_adapter(inputs_embeds=enc_out.to(self.dtype)).last_hidden_state
        ctx_embeds = self.context_bridge(ctx_feats)
        return ctx_embeds

    def pool_context(self, ctx_embeds: torch.Tensor) -> torch.Tensor:
        seq_len = ctx_embeds.shape[1]
        if seq_len == 0:
            raise ValueError("Context embeddings must contain at least one timestep.")

        if seq_len == 1:
            sampled = ctx_embeds.expand(-1, Config.NUM_PREFIX_TOKENS, -1)
        else:
            indices = torch.linspace(
                0,
                seq_len - 1,
                steps=Config.NUM_PREFIX_TOKENS,
                device=ctx_embeds.device,
            ).round().long()
            sampled = ctx_embeds.index_select(1, indices)
        return self.prefix_norm(sampled)

    def forward(
        self,
        audio_patches: torch.Tensor,
        audio_masks: torch.Tensor,
        retrieved_waveform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz = audio_patches.shape[0]
        device = audio_patches.device

        ctx_embeds = self.encode_context(retrieved_waveform)
        prefix_embeds = self.pool_context(ctx_embeds)
        sep = self.sep_embed.expand(bsz, -1, -1).to(device=device, dtype=prefix_embeds.dtype)
        prefix_embeds = torch.cat([prefix_embeds, sep], dim=1)
        n_prefix = prefix_embeds.shape[1]

        audio_patches_2d = audio_patches.reshape(bsz, -1, PATCH_SIZE)
        out = self.conditioned_patch_decoder(
            patches=audio_patches_2d,
            masks=audio_masks,
            prefix_embeds=prefix_embeds.to(device),
        )
        encoded_patches = out["last_hidden_state"]

        encoded_patches = encoded_patches[:, n_prefix:, :]
        effective_patch_count = encoded_patches.shape[1]
        audio_masks = audio_masks[:, :effective_patch_count]
        audio_patches_2d = audio_patches_2d[:, :effective_patch_count, :]

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


def _load_wav_paths(audio_dir: Optional[str] = None, audio_manifest: Optional[str] = None) -> List[str]:
    if bool(audio_dir) == bool(audio_manifest):
        raise ValueError("Provide exactly one of audio_dir or audio_manifest.")

    if audio_manifest is not None:
        with open(audio_manifest, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip()]
        if not paths:
            raise FileNotFoundError(f"No WAV paths found in manifest {audio_manifest}")
        return paths

    paths = sorted(glob(os.path.join(audio_dir, "**/*.wav"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")
    return paths


# ==================== Dataset ====================
class AudioDataset(Dataset):
    """
    Loads WAV files, chunks them, and returns lossless PCM byte patches plus
    EnCodec-based retrieved context.
    """

    def __init__(
        self,
        file_paths: List[str],
        encodec_model: EncodecModel,
        rag_retriever: EnCodecRAGRetriever,
        chunk_duration: float = Config.CHUNK_DURATION,
        max_patches: int = Config.MAX_PATCHES,
        top_k: int = Config.TOP_K_RETRIEVAL,
        retrieval_cache_path: Optional[str] = None,
    ):
        self.file_paths = file_paths
        self.encodec = encodec_model
        self.retriever = rag_retriever
        self.chunk_duration = chunk_duration
        self.max_patches = max_patches
        self.top_k = top_k

        self.samples: List[Tuple[str, int]] = []
        for path in file_paths:
            with wave.open(path, "rb") as wav_in:
                params = wav_in.getparams()
                chunk_frames = _chunk_frame_budget(
                    params,
                    chunk_duration,
                    max_patches=max_patches,
                )
                n_chunks = max(1, (params.nframes + chunk_frames - 1) // chunk_frames)
            for i in range(n_chunks):
                self.samples.append((path, i))

        # Pre-compute chunk-level retrievals once and optionally persist to disk
        # so EnCodec inference is not repeated across training runs.
        if retrieval_cache_path and os.path.exists(retrieval_cache_path):
            print(f"Loading cached retrievals from {retrieval_cache_path}")
            import pickle
            with open(retrieval_cache_path, "rb") as f:
                self.chunk_retrievals = pickle.load(f)
            print(f"Loaded {len(self.chunk_retrievals)} cached retrievals.")
        else:
            print(f"Pre-computing retrievals for {len(self.samples)} chunks...")
            self.chunk_retrievals: List[List[Dict[str, Any]]] = []
            for path, chunk_idx in tqdm(self.samples, desc="Caching retrievals"):
                chunk_info = load_wav_chunk(path, chunk_idx, chunk_duration, max_patches)
                results = self.retriever.retrieve(chunk_info["waveform"], k=top_k)
                self.chunk_retrievals.append(results)
            if retrieval_cache_path:
                import pickle
                os.makedirs(os.path.dirname(retrieval_cache_path) or ".", exist_ok=True)
                with open(retrieval_cache_path, "wb") as f:
                    pickle.dump(self.chunk_retrievals, f)
                print(f"Saved retrievals to {retrieval_cache_path}")
            print("Retrieval pre-computation complete.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, chunk_idx = self.samples[idx]
        chunk_info = load_wav_chunk(
            path,
            chunk_idx,
            self.chunk_duration,
            max_patches=self.max_patches,
        )
        chunk = chunk_info["waveform"]
        bytes_list = chunk_info["bytes_list"]
        patch_mask = chunk_info["patch_mask"]

        results = self.chunk_retrievals[idx]
        retrieved_signal, _ = _aggregate_retrieved_waveforms(results, chunk)

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
    top_k: int = Config.TOP_K_RETRIEVAL,
    max_patches: int = Config.MAX_PATCHES,
):
    """
    Compress a WAV file by arithmetic-coding its exact PCM byte stream.

    Returns:
        compress_ratio: float
        compressed_len: int
        original_len: int
    """
    from arithmetic_coder import ac_utils, arithmetic_coder
    from evaluation.LLMCompress import Metric

    model.eval()
    model.to(device)

    total_metric = Metric()
    all_compressed = []

    with wave.open(wav_path, "rb") as wav_in:
        params = wav_in.getparams()
        chunk_frames = _chunk_frame_budget(
            params,
            chunk_duration,
            max_patches=max_patches,
        )
        n_chunks = max(1, (params.nframes + chunk_frames - 1) // chunk_frames)

    file_metadata = {
        "nchannels": params.nchannels,
        "sampwidth": params.sampwidth,
        "framerate": params.framerate,
        "nframes": params.nframes,
        "comptype": params.comptype,
        "compname": params.compname,
    }
    for i in range(n_chunks):
        chunk_info = load_wav_chunk(
            wav_path,
            i,
            chunk_duration,
            max_patches=max_patches,
        )
        chunk = chunk_info["waveform"]
        bytes_list = chunk_info["bytes_list"]
        patch_mask = chunk_info["patch_mask"]
        pcm_num_bytes = chunk_info["pcm_num_bytes"]

        audio_patches = torch.tensor(bytes_list, dtype=torch.long).unsqueeze(0).to(device)
        audio_masks = torch.tensor(patch_mask, dtype=torch.long).unsqueeze(0).to(device)

        chunk_retrieval_results = rag_retriever.retrieve(chunk, k=top_k)
        ret_signal, retrieved_paths = _aggregate_retrieved_waveforms(
            chunk_retrieval_results,
            chunk,
        )

        retrieved_waveform = ret_signal.unsqueeze(0).to(device)

        logits = model.get_logits_for_compression(
            audio_patches, audio_masks, retrieved_waveform
        )
        valid_patch_count = int(audio_masks.sum().item())
        payload_patch_count = max(0, valid_patch_count - 2)
        logits = logits[: payload_patch_count + 1, :-1, :]
        logits = logits[:-1, :, :].reshape(1, -1, 257)

        input_ids = audio_patches[:, PATCH_SIZE : (valid_patch_count - 1) * PATCH_SIZE]
        input_ids = torch.cat([torch.tensor([[256]], device=device), input_ids], dim=1)

        output_bits = []
        encoder = arithmetic_coder.Encoder(
            base=2, precision=precision, output_fn=output_bits.append
        )

        target = input_ids[
            :,
            prefix_length : prefix_length + pcm_num_bytes,
        ].detach().cpu().numpy().reshape(-1)
        probs_np = (
            logits[:, prefix_length - 1 :, :]
            .softmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
            .squeeze(0)
        )
        probs_np = probs_np[: len(target)]

        for symbol, prob in zip(target, probs_np):
            encoder.encode(
                ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32), symbol
            )
        encoder.terminate()

        compressed_bits = "".join(map(str, output_bits))
        compressed_bytes, num_padded_bits = ac_utils.bits_to_bytes(compressed_bits)

        total_metric.accumulate(len(compressed_bytes), len(target))
        format_patch = audio_patches[:, :PATCH_SIZE].squeeze(0).detach().cpu().tolist()
        all_compressed.append(
            {
                "data": compressed_bytes,
                "num_padded_bits": num_padded_bits,
                "pcm_num_bytes": pcm_num_bytes,
                "format_patch": format_patch,
                "retrieved_path": "\n".join(retrieved_paths) if retrieved_paths else None,
            }
        )

    compress_rate, compress_ratio = total_metric.compute_ratio()
    print(f"Compression ratio: {compress_ratio:.4f}  |  Rate: {compress_rate:.4f} bpb")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        comptype_bytes = file_metadata["comptype"].encode("utf-8")
        compname_bytes = file_metadata["compname"].encode("utf-8")

        f.write(b"RAGP")
        f.write((2).to_bytes(1, "big"))
        f.write(len(all_compressed).to_bytes(4, "big"))
        f.write(file_metadata["nchannels"].to_bytes(2, "big"))
        f.write(file_metadata["sampwidth"].to_bytes(2, "big"))
        f.write(file_metadata["framerate"].to_bytes(4, "big"))
        f.write(file_metadata["nframes"].to_bytes(4, "big"))
        f.write(len(comptype_bytes).to_bytes(2, "big"))
        f.write(comptype_bytes)
        f.write(len(compname_bytes).to_bytes(2, "big"))
        f.write(compname_bytes)

        for chunk in all_compressed:
            data = chunk["data"]
            format_patch = chunk["format_patch"]
            retrieved_path = (chunk["retrieved_path"] or "").encode("utf-8")

            f.write(len(data).to_bytes(4, "big"))
            f.write(chunk["num_padded_bits"].to_bytes(1, "big"))
            f.write(chunk["pcm_num_bytes"].to_bytes(4, "big"))
            f.write(len(format_patch).to_bytes(2, "big"))
            f.write(struct.pack(f">{len(format_patch)}H", *format_patch))
            f.write(len(retrieved_path).to_bytes(2, "big"))
            f.write(retrieved_path)
            f.write(data)

    print(f"Saved compressed output to {output_path}")
    return compress_ratio, total_metric.compressed_length, total_metric.total_length


# ==================== Training ====================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train RAG-EnCodec-bGPT audio compressor")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory of WAV training files")
    parser.add_argument("--audio_manifest", type=str, default=None,
                        help="Text file listing WAV training files, one per line")
    parser.add_argument("--retriever_manifest", type=str, default=None,
                        help="Optional text file listing WAV files to build the retriever index from")
    parser.add_argument("--retriever_path", type=str, default="retriever_cache/encodec",
                        help="Path to save/load FAISS index")
    parser.add_argument("--bgpt_checkpoint", type=str,
                        default=Config.BGPT_CHECKPOINT_AUDIO)
    parser.add_argument("--output_dir", type=str, default="./rag_encodec_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_wavs", type=int, default=None,
                        help="Optional cap on the number of WAV files to use")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from, or 'latest' to auto-detect")
    parser.add_argument("--retrieval_cache", type=str, default="retriever_cache/chunk_retrievals.pkl",
                        help="Path to cache pre-computed chunk retrievals across runs")
    args = parser.parse_args()

    device = Config.DEVICE

    # Load EnCodec (used as frozen tokenizer and feature extractor)
    encodec = EncodecModel.from_pretrained(Config.ENCODEC_MODEL)
    encodec.to(device).eval()
    for p in encodec.parameters():
        p.requires_grad = False

    train_audio_files = _load_wav_paths(
        audio_dir=args.audio_dir,
        audio_manifest=args.audio_manifest,
    )
    if args.max_wavs is not None:
        train_audio_files = train_audio_files[:args.max_wavs]
        print(f"Using {len(train_audio_files)} WAV files due to --max_wavs={args.max_wavs}")

    retriever_audio_files = (
        _load_wav_paths(audio_manifest=args.retriever_manifest)
        if args.retriever_manifest is not None
        else train_audio_files
    )

    # Build / load RAG index
    retriever = EnCodecRAGRetriever(persist_path=args.retriever_path)
    if retriever.index.ntotal == 0:
        print(f"Building retriever index from {len(retriever_audio_files)} WAV files")
        retriever.index_audio_files(retriever_audio_files)
    else:
        print(f"Using existing retriever index at {args.retriever_path} with {retriever.index.ntotal} entries")

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
        file_paths=train_audio_files,
        encodec_model=encodec,
        rag_retriever=retriever,
        retrieval_cache_path=args.retrieval_cache,
    )
    print(f"Dataset: {len(dataset)} chunks from {len(train_audio_files)} files")

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        bf16=False,
        fp16=False,
        logging_steps=20,
        remove_unused_columns=False,
        save_strategy="no",
    )

    def collate_fn(batch):
        max_retrieved_len = max(b["retrieved_waveform"].shape[-1] for b in batch)
        padded_retrieved = []
        for item in batch:
            waveform = item["retrieved_waveform"]
            if waveform.shape[-1] < max_retrieved_len:
                waveform = torch.nn.functional.pad(
                    waveform, (0, max_retrieved_len - waveform.shape[-1])
                )
            padded_retrieved.append(waveform)

        return {
            "audio_patches": torch.stack([b["audio_patches"] for b in batch]),
            "audio_masks": torch.stack([b["audio_masks"] for b in batch]),
            "retrieved_waveform": torch.stack(padded_retrieved),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))

    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
