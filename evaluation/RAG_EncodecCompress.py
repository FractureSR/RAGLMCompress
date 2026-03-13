import os
import gc
import wave
import torch
import logging
from glob import glob
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

from utils.encodec_rag import EnCodecRAGRetriever
from train.train_RAG_EnCodec_Compress import (
    RAGEnCodecBGPT,
    Config,
    compress_audio_file,
)
from evaluation.LLMCompress import Metric


logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ==================== Configuration ====================
class RAGCompressionConfig:
    """Configuration for RAG-enhanced EnCodec + bGPT audio compression"""

    # Model paths
    TRAINED_MODEL_CHECKPOINT = "./rag_encodec_ckpt"
    BGPT_CHECKPOINT_AUDIO = "./pretrained/bgpt/weights-audio.pth"

    # Dataset paths
    RAG_AUDIO_DATASET = "./data/librispeech/wav/**/*.wav"
    TEST_DATASET_AUDIO = "./data/test_workflow/wav/*.wav"

    # Retriever storage
    RETRIEVER_STORAGE_PATH = "./retriever_cache/encodec"

    # Optional single-file test path
    TEST_SAMPLE_PATH = "./data/librispeech/test/wav/5895-34629-0000.wav"

    # Output paths
    COMPRESSED_OUTPUT_DIR = "./output_rag_audio"

    # Retrieval parameters
    TOP_K_RETRIEVAL = 1

    # Compression parameters
    NUM_FILES_TO_COMPRESS = 1
    VERBOSE_THRESHOLD = 5
    AUDIO_CHUNK_DURATION = 1.0

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Helper Functions ====================
def resolve_model_checkpoint(model_path: str) -> str:
    """
    Resolve model checkpoint path.
    Supports:
      - direct pytorch_model.bin file
      - directory containing pytorch_model.bin
      - latest checkpoint-* subdirectory
    """
    if os.path.isfile(model_path):
        return model_path

    direct_bin = os.path.join(model_path, "pytorch_model.bin")
    if os.path.isfile(direct_bin):
        return direct_bin

    checkpoint_dirs = sorted(
        glob(os.path.join(model_path, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else -1,
    )
    if checkpoint_dirs:
        latest_bin = os.path.join(checkpoint_dirs[-1], "pytorch_model.bin")
        if os.path.isfile(latest_bin):
            return latest_bin

    raise FileNotFoundError(f"Could not find checkpoint in {model_path}")


def split_wav_to_chunks(wav_file, output_folder, chunk_duration=1.0):
    """
    Split a WAV file into chunks of specified duration.
    """
    with wave.open(wav_file, "rb") as wav:
        params = wav.getparams()
        n_channels = params.nchannels
        sampwidth = params.sampwidth
        framerate = params.framerate
        n_frames = params.nframes
        chunk_frames = int(framerate * chunk_duration)
        frames = wav.readframes(n_frames)

    filename = os.path.basename(wav_file)
    name_without_ext = os.path.splitext(filename)[0]
    chunk_folder = os.path.join(output_folder, name_without_ext)
    os.makedirs(chunk_folder, exist_ok=True)

    chunk_files = []
    chunk_idx = 0

    bytes_per_frame = n_channels * sampwidth
    total_bytes = len(frames)
    chunk_bytes = chunk_frames * bytes_per_frame

    for start_byte in range(0, total_bytes, chunk_bytes):
        end_byte = min(start_byte + chunk_bytes, total_bytes)
        chunk_data = frames[start_byte:end_byte]

        chunk_filename = f"chunk_{chunk_idx:04d}.wav"
        chunk_path = os.path.join(chunk_folder, chunk_filename)

        with wave.open(chunk_path, "wb") as chunk_wav:
            chunk_wav.setparams(params)
            chunk_wav.writeframes(chunk_data)

        chunk_files.append(chunk_path)
        chunk_idx += 1

    wav_params = {
        "nchannels": n_channels,
        "sampwidth": sampwidth,
        "framerate": framerate,
        "comptype": params.comptype,
        "compname": params.compname,
    }

    return chunk_files, wav_params


def merge_wav_chunks(chunk_files, output_path, wav_params):
    """
    Merge WAV chunks back into a single WAV file.
    """
    all_frames = []

    for chunk_file in sorted(chunk_files):
        with wave.open(chunk_file, "rb") as chunk_wav:
            frames = chunk_wav.readframes(chunk_wav.getnframes())
            all_frames.append(frames)

    with wave.open(output_path, "wb") as output_wav:
        output_wav.setnchannels(wav_params["nchannels"])
        output_wav.setsampwidth(wav_params["sampwidth"])
        output_wav.setframerate(wav_params["framerate"])
        output_wav.setcomptype(wav_params["comptype"], wav_params["compname"])

        for frames in all_frames:
            output_wav.writeframes(frames)


# ==================== Retriever Setup ====================
def setup_retriever(
    persist_path: str = None,
) -> EnCodecRAGRetriever:
    """
    Setup RAG retriever by loading existing FAISS index.
    """
    if persist_path is None:
        persist_path = RAGCompressionConfig.RETRIEVER_STORAGE_PATH

    print("\n=== Setting up RAG Retriever ===")
    retriever = EnCodecRAGRetriever(persist_path=persist_path)

    if retriever.index is None or retriever.index.ntotal == 0:
        raise ValueError(
            f"Retriever index is empty at {persist_path}. "
            "Please build the EnCodec FAISS index first."
        )
    else:
        print("\nIndex loaded from disk. Skipping indexing.")
        print(f"Indexed entries: {retriever.index.ntotal}")

    print("--- Retriever is ready ---\n")
    return retriever


# ==================== Model Loading ====================
def load_rag_audio_model(
    trained_model_checkpoint: str = None,
    bgpt_checkpoint: str = None,
    device: str = None,
) -> RAGEnCodecBGPT:
    """
    Load trained RAG EnCodec + bGPT model.
    """
    if trained_model_checkpoint is None:
        trained_model_checkpoint = RAGCompressionConfig.TRAINED_MODEL_CHECKPOINT
    if bgpt_checkpoint is None:
        bgpt_checkpoint = RAGCompressionConfig.BGPT_CHECKPOINT_AUDIO
    if device is None:
        device = RAGCompressionConfig.DEVICE

    print("\n=== Loading RAG Audio Compression Model ===")

    resolved_checkpoint = resolve_model_checkpoint(trained_model_checkpoint)
    print(f"Resolved trained checkpoint: {resolved_checkpoint}")

    model = RAGEnCodecBGPT(
        encodec_model_name=Config.ENCODEC_MODEL,
        bgpt_checkpoint=bgpt_checkpoint,
    )

    state_dict = torch.load(resolved_checkpoint, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: missing keys ({len(missing_keys)})")
    if unexpected_keys:
        print(f"Warning: unexpected keys ({len(unexpected_keys)})")

    model.to(device)
    model.eval()

    print("RAG audio compression model loaded successfully.\n")
    return model


# ==================== Data Loading ====================
def load_test_sample(test_sample_path: str = None) -> Optional[str]:
    """
    Try to load one test WAV sample.
    """
    if test_sample_path is None:
        test_sample_path = RAGCompressionConfig.TEST_SAMPLE_PATH

    try:
        if os.path.isfile(test_sample_path):
            print(f"✓ Loaded test sample from {test_sample_path}")
            return test_sample_path
        else:
            print(f"✗ Test sample file not found at {test_sample_path}")
            return None
    except Exception as e:
        print(f"✗ Error loading test sample: {e}")
        return None


def load_compression_audio_files(
    dataset_path: str = None,
    num_files: int = None,
    test_sample_path: str = None,
) -> tuple[List[str], int]:
    """
    Load WAV files for compression testing.
    Returns:
      - list of wav paths
      - total original byte size
    """
    if dataset_path is None:
        dataset_path = RAGCompressionConfig.TEST_DATASET_AUDIO
    if num_files is None:
        num_files = RAGCompressionConfig.NUM_FILES_TO_COMPRESS

    print("\n=== Loading Audio Files for Compression ===")

    test_sample = load_test_sample(test_sample_path)

    if test_sample is not None:
        files = [test_sample]
        total_size = os.path.getsize(test_sample)
        print("Using test sample as the audio file to compress")
        print("Total files: 1 (from test sample)")
    else:
        files = sorted(glob(dataset_path))
        if not files:
            raise FileNotFoundError(f"No WAV files found in {dataset_path}")
        files = files[:num_files]
        total_size = sum(os.path.getsize(f) for f in files)
        print(f"✓ Loaded {len(files)} files from dataset")

    print(f"Total original size: {total_size} bytes")
    return files, total_size


# ==================== Compression with RAG ====================
def compress_with_rag_context(
    wav_path: str,
    retriever: EnCodecRAGRetriever,
    model: RAGEnCodecBGPT,
    metric: Metric,
    output_dir: str,
    chunk_duration: float = None,
    verbose: bool = False,
) -> Tuple[float, int, int, str]:
    """
    Compress one WAV file using RAG-conditioned EnCodec + bGPT.
    Returns:
      - compression ratio for this file
      - output compressed path
    """
    if chunk_duration is None:
        chunk_duration = RAGCompressionConfig.AUDIO_CHUNK_DURATION

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(wav_path)
    output_path = os.path.join(output_dir, base + ".bin")

    # if verbose:
    #     print("\n--- Retrieval Preview ---")
    #     try:
    #         import torchaudio
    #         signal, sr = librosa.load(wav_path, sr=None, mono=False)
    #         signal = torch.from_numpy(signal).float()

    #         if signal.ndim == 1:
    #             signal = signal.unsqueeze(0)

    #         if sr != Config.ENCODEC_SAMPLE_RATE:
    #             signal = torchaudio.functional.resample(signal, sr, Config.ENCODEC_SAMPLE_RATE)

    #         if signal.shape[0] > 1:
    #             signal = signal.mean(dim=0, keepdim=True)

    #         top_k_results = retriever.retrieve(signal[:, : min(signal.shape[-1], Config.ENCODEC_SAMPLE_RATE)], k=RAGCompressionConfig.TOP_K_RETRIEVAL)
    #         for i, result in enumerate(top_k_results, 1):
    #             print(f"\nResult {i}:")
    #             print(f"  ID: {result['id']}")
    #             print(f"  Score: {result['score']:.4f}")
    #             print(f"  Path: {result['path']}")
    #     except Exception as e:
    #         print(f"  Retrieval preview skipped: {e}")

    ratio, compressed_len, original_len = compress_audio_file(
        model=model,
        wav_path=wav_path,
        rag_retriever=retriever,
        output_path=output_path,
        device=RAGCompressionConfig.DEVICE,
        chunk_duration=chunk_duration,
        precision=Config.PRECISION,
        prefix_length=Config.PREFIX_LENGTH,
    )

    metric.accumulate(compressed_len, original_len)
    return ratio, compressed_len, original_len, output_path


# ==================== Main Workflow ====================
def run_rag_audio_compression(
    test_sample_path: str = None,
    dataset_path: str = None,
    num_files: int = None,
):
    retriever = setup_retriever()
    model = load_rag_audio_model()

    audio_files, total_length = load_compression_audio_files(
        dataset_path=dataset_path,
        num_files=num_files,
        test_sample_path=test_sample_path,
    )

    metric = Metric()

    print("\n=== Starting Audio Compression ===")
    num_docs = len(audio_files)
    verbose = num_docs <= RAGCompressionConfig.VERBOSE_THRESHOLD

    per_file_ratios = []

    for idx, wav_path in enumerate(audio_files, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing file {idx}/{num_docs}")
        print(f"File: {wav_path}")
        print(f"Original file size: {os.path.getsize(wav_path)} bytes")
        print(f"{'=' * 60}")

        token_ratio, compressed_len, original_len, compressed_path = compress_with_rag_context(
            wav_path=wav_path,
            retriever=retriever,
            model=model,
            metric=metric,
            output_dir=RAGCompressionConfig.COMPRESSED_OUTPUT_DIR,
            verbose=verbose,
        )

        artifact_size = os.path.getsize(compressed_path)
        file_ratio = compressed_len / original_len if original_len else 0.0

        per_file_ratios.append((wav_path, file_ratio))
        print(f"\n✓ File {idx} compressed: {compressed_path}")
        print(f"✓ File token-stream compression ratio: {token_ratio:.6f}")
        print(f"✓ File payload compression ratio: {file_ratio:.6f}")
        print(f"✓ Saved artifact size: {artifact_size} bytes")

    print("\n" + "=" * 60)
    print("=== Final Compression Results ===")
    print("=" * 60)

    compression_rate, compression_ratio = metric.compute_ratio()

    print(f"Total original length: {metric.total_length} bytes")
    print(f"Total compressed length: {metric.compressed_length} bytes")
    print(f"Compression ratio: {compression_ratio:.6f}")
    print(f"Compression rate: {compression_rate:.6f}x")

    print("\nPer-file payload ratios:")
    for wav_path, ratio in per_file_ratios:
        print(f"  {os.path.basename(wav_path)}: {ratio:.6f}")

    print("=" * 60)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metric, compression_rate, compression_ratio

# ==================== Chunked Audio Workflow ====================
def test_audio_compression(
    model,
    retriever,
    test: bool = True,
    temp_folder: str = "temp_rag_audio",
    output_folder: str = "output_rag_audio",
    chunk_duration: float = None,
):
    """
    Test chunked audio compression workflow, similar to BGPTCompress.py
    """
    if chunk_duration is None:
        chunk_duration = RAGCompressionConfig.AUDIO_CHUNK_DURATION

    dataset_path = (
        RAGCompressionConfig.TEST_DATASET_AUDIO
        if test
        else RAGCompressionConfig.RAG_AUDIO_DATASET
    )

    wav_files = sorted(glob(dataset_path))
    if not wav_files:
        print(f"No WAV files found in {dataset_path}")
        return

    print(f"Found {len(wav_files)} WAV files to test")
    print(f"Dataset path: {dataset_path}")
    print(f"Chunk duration: {chunk_duration} seconds")
    print("=" * 80)

    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    split_folder = os.path.join(temp_folder, "split")
    compressed_folder = os.path.join(temp_folder, "compressed")
    os.makedirs(split_folder, exist_ok=True)
    os.makedirs(compressed_folder, exist_ok=True)

    total_metric = Metric()

    for wav_file in wav_files:
        print(f"\nProcessing: {os.path.basename(wav_file)}")
        print("-" * 80)

        filename = os.path.basename(wav_file)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"Step 1: Splitting WAV into {chunk_duration}s chunks...")
        chunk_files, wav_params = split_wav_to_chunks(
            wav_file,
            split_folder,
            chunk_duration=chunk_duration,
        )
        print(f"Created {len(chunk_files)} chunks")

        print(f"\nStep 2: Compressing {len(chunk_files)} chunks...")
        compressed_chunk_paths = []

        for chunk_file in tqdm(chunk_files, desc="Compressing chunks"):
            chunk_name = os.path.basename(chunk_file)
            compressed_path = os.path.join(compressed_folder, chunk_name + ".bin")

            ratio, compressed_len, original_len = compress_audio_file(
                model=model,
                wav_path=chunk_file,
                rag_retriever=retriever,
                output_path=compressed_path,
                device=RAGCompressionConfig.DEVICE,
                chunk_duration=chunk_duration,
                precision=Config.PRECISION,
                prefix_length=Config.PREFIX_LENGTH,
            )

            total_metric.accumulate(compressed_len, original_len)


            print(f"  {chunk_name}: ratio={ratio:.6f}")

        print("\nCompression ratio/rate:", total_metric.compute_ratio())
        print("=" * 80)

    print("\nAudio compression test completed!")
    print(f"Compressed chunk outputs saved to: {compressed_folder}")
    print("Compression ratio/rate:", total_metric.compute_ratio())


# ==================== Main ====================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Running RAG Audio Compression Test")
    print("=" * 80)

    # Option 1: simple full-file workflow
    run_rag_audio_compression()

    # Option 2: chunk-based workflow like BGPTCompress.py
    # test_audio_compression(
    #     model=model,
    #     retriever=retriever,
    #     test=True,
    #     temp_folder="temp_rag_audio",
    #     output_folder="output_rag_audio",
    #     chunk_duration=RAGCompressionConfig.AUDIO_CHUNK_DURATION,
    # )
