import os
import numpy as np
import torch
from typing import Iterator, List, Tuple, Optional
from transformers import GPT2Config
import time
import logging
from tqdm import tqdm
from glob import glob
from bgpt.utils import bGPTLMHeadModel
from bgpt.config import *
from arithmetic_coder import ac_utils, arithmetic_coder
from evaluation.LLMCompress import write_padded_bytes, read_padded_bytes, Metric
from utils.bmp_utils import split_bmp_to_patches, merge_patches_to_bmp
import wave

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ==================== Configuration ====================
class CompressionConfig:
    """Configuration for bGPT compression"""
    # Model paths
    MODEL_CHECKPOINT_IMAGE = "./pretrained/bgpt/weights-image.pth"
    MODEL_CHECKPOINT_AUDIO = "./pretrained/bgpt/weights-audio.pth"

    # Dataset paths
    DATASET_IMAGE = "datasets/clic_2024/bmp/*.bmp"
    DATASET_AUDIO = "datasets/librispeech/wav/*.wav"

    MAX_FILES = 1
    MAX_SEG = 1

    # Output paths
    COMPRESSED_OUTPUT = "compressed.bin"

    # Model configuration
    PATCH_LENGTH = 512  # modify to fit the trained checkpoint
    PATCH_NUM_LAYERS = PATCH_NUM_LAYERS  # from bgpt.config
    BYTE_NUM_LAYERS = BYTE_NUM_LAYERS    # from bgpt.config
    HIDDEN_SIZE = HIDDEN_SIZE            # from bgpt.config
    PATCH_SIZE = PATCH_SIZE              # from bgpt.config

    # Compression parameters
    PRECISION = 64
    PREFIX_LENGTH = 1

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data type (for automatic path selection)
    DATA_TYPE = "image"  # or "audio"

    # BMP splitting parameters
    BMP_PATCH_SIZE = 32  # Size of square patches for BMP splitting

    # Audio splitting parameters
    AUDIO_CHUNK_DURATION = 1.0  # Duration of each audio chunk in seconds

    # Generic split size for non-image/audio files
    # 这里只用于通用文件切段，不改变你的 padding 逻辑
    GENERIC_SEGMENT_BYTES = (PATCH_LENGTH - 2) * PATCH_SIZE

    @classmethod
    def get_model_checkpoint(cls):
        """Get model checkpoint path based on data type"""
        return cls.MODEL_CHECKPOINT_IMAGE if cls.DATA_TYPE == "image" else cls.MODEL_CHECKPOINT_AUDIO

    @classmethod
    def get_dataset_path(cls):
        """Get dataset path based on data type"""
        return cls.DATASET_IMAGE if cls.DATA_TYPE == "image" else cls.DATASET_AUDIO


# ==================== Helper Functions ====================
def pad_input_for_bgpt(segments, ext_list, device, pad_to_length=None):
    """
    Pads input segments for bGPT model.
    Could be used for batch processing.

    :param segments: list of byte segments
    :param ext_list: list of extension bytes corresponding to each segment
    :param device: torch device
    :param pad_to_length: optional fixed padding length
    :return: dict with padded patches and masks
    """
    max_length = max(len(b) for b in segments) + 2 * CompressionConfig.PATCH_SIZE

    padded_bytes = []
    padded_masks = []

    for b, ext in zip(segments, ext_list):
        if pad_to_length is not None and len(b) < pad_to_length:
            b = b + [256] * (pad_to_length - len(b))
        bos_patch = ext + [256] * (CompressionConfig.PATCH_SIZE - len(ext))
        b = bos_patch + b + [256] * CompressionConfig.PATCH_SIZE

        valid_length = len(b)
        padded_bytes.append(b + [256] * (max_length - valid_length))

        patch_count = (valid_length + CompressionConfig.PATCH_SIZE - 1) // CompressionConfig.PATCH_SIZE
        total_patches = (max_length + CompressionConfig.PATCH_SIZE - 1) // CompressionConfig.PATCH_SIZE
        patch_masks = [1] * patch_count + [0] * (total_patches - patch_count)
        padded_masks.append(patch_masks)

    patches = torch.tensor(padded_bytes, dtype=torch.long)
    masks = torch.tensor(padded_masks, dtype=torch.long)

    return {
        "patches": patches.to(device),
        "masks": masks.to(device),
    }


def bgpt_compress(compress_input, logits, metric, precision=None, prefix_length=None):
    """
    :param compress_input: symbols to be compressed
    :param logits: generation probabilities from the model
    :param metric: compression metrics
    :param precision: encoder precision
    :param prefix_length: prefix length for encoding
    :return: compressed result, a floating number
    """
    if precision is None:
        precision = CompressionConfig.PRECISION
    if prefix_length is None:
        prefix_length = CompressionConfig.PREFIX_LENGTH

    output = []
    encoder = arithmetic_coder.Encoder(
        base=2,
        precision=precision,
        output_fn=output.append,
    )

    start_symbol = compress_input[:, :1]

    target_sequence_to_encode = compress_input[:, prefix_length:]
    logits_for_encoding = logits[:, prefix_length - 1:, :]

    probs = logits_for_encoding.softmax(dim=-1).to(torch.float32)
    pd = torch.gather(
        probs, dim=-1, index=target_sequence_to_encode.unsqueeze(-1)
    ).squeeze(-1)

    probs = np.vstack(probs.detach().cpu().numpy().squeeze())
    sequence_array = target_sequence_to_encode.detach().cpu().numpy().reshape(-1)
    pd = pd.squeeze()

    for symbol, prob, pd_prob in zip(sequence_array, probs, pd):
        encoder.encode(
            ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32), symbol
        )
    encoder.terminate()

    compressed_bits = "".join(map(str, output))
    compressed_bytes, num_padded_bits = ac_utils.bits_to_bytes(compressed_bits)

    metric.accumulate(len(compressed_bytes), len(sequence_array))

    compress_rate, compress_ratio = metric.compute_ratio()
    logger.info(f"compressed length: {metric.compressed_length}")
    logger.info(f"original length: {metric.total_length}")
    logger.info(f"compression ratio: {compress_ratio:.6f}")
    logger.info(f"compression rate: {compress_rate:.6f}")

    return compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs


def bgpt_decode(
    compressed_bytes,
    num_padded_bits,
    model,
    start_patch,
    ext,
    device,
    original_seq_len,
    original_sequence=None,
    pd=None,
    probs=None,
    precision=None,
    do_test=False,
):
    """
    :param compressed_bytes: compressed data
    :param num_padded_bits: padded bits
    :param model: same model as encoder
    :param start_patch: starting patch for decoding
    :param ext: file extension bytes
    :param device: torch device
    :param original_seq_len: original sequence length
    :param original_sequence: original symbol sequence, for testing purpose
    :param pd: actually not needed, used for testing
    :param probs: probabilities from encoder
    :param precision: decoder precision
    :param do_test: whether to run testing
    :return: decoded sequence
    """
    if precision is None:
        precision = CompressionConfig.PRECISION

    data_iter = iter(
        ac_utils.bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits)
    )

    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> Optional[int]:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    decoder = arithmetic_coder.Decoder(
        base=2,
        precision=precision,
        input_fn=_input_fn,
    )

    target_diff_list = []
    target_in_top5_list = []

    start_symbol = []
    sequence_array_de = np.array(start_symbol)

    for i in range(original_seq_len):
        sequence_array_de = sequence_array_de[None, :].tolist()
        sequence_array_de_input = pad_input_for_bgpt(
            sequence_array_de, [ext], device, original_seq_len
        )

        logits = model(**sequence_array_de_input).logits
        logits = logits[:-1, :-1, :]
        prob_de = logits.reshape(1, -1, 257).softmax(-1).detach().cpu().numpy().squeeze(axis=0)

        de_token = decoder.decode(
            ac_utils.normalize_pdf_for_arithmetic_coding(prob_de[i], data_type=np.float32),
        )
        sequence_array_de = np.append(sequence_array_de, de_token)

        current_len = len(sequence_array_de)
        target_len = original_seq_len

        if current_len < target_len:
            padded = np.pad(
                sequence_array_de, (0, (target_len - current_len)), constant_values=0
            )
        else:
            padded = sequence_array_de
        sequence_array_de_input = torch.tensor(
            padded, dtype=torch.long, device=device
        ).unsqueeze(0)

        if do_test:
            top_indices_de = prob_de[i].argsort()[-5:][::-1]
            top_indices = probs[i].argsort()[-5:][::-1]

            target_diff = (
                probs[i, original_sequence[i]] - prob_de[i, original_sequence[i]]
            )
            target_diff_list.append(target_diff)

            target_in_top5 = original_sequence[i] in top_indices
            target_in_top5_list.append(target_in_top5)
            print(
                f"idx: {i}, original token: {original_sequence[i]}, decoder token: {de_token}"
            )
            print(
                f"diff probs max: {max(abs(probs[i] - prob_de[i]))}, original sum error: {abs(sum(prob_de[i]) - 1.0)}, decoder sum error: {abs(sum(probs[i]) - 1.0)}"
            )
            print(
                f"original: {top_indices}, target_in_top5: {target_in_top5} decode: {top_indices_de}, "
            )
            print(f"target diff: {target_diff}")
            if original_sequence[i] != de_token:
                import pdb
                pdb.set_trace()

    return sequence_array_de_input


def read_bytes(filename):
    """
    Read bytes from file and extract extension
    :param filename: path to file
    :return: tuple of (bytes list, extension bytes)
    """
    ext = filename.split(".")[-1]

    ext = bytearray(ext, "utf-8")
    ext = [byte for byte in ext][:CompressionConfig.PATCH_SIZE]
    with open(filename, "rb") as f:
        file_bytes = f.read()

    bytes_list = []
    for byte in file_bytes:
        bytes_list.append(byte)

    if len(bytes_list) % CompressionConfig.PATCH_SIZE != 0:
        bytes_list = bytes_list + [256] * (
            CompressionConfig.PATCH_SIZE - len(bytes_list) % CompressionConfig.PATCH_SIZE
        )

    return bytes_list, ext


def write_bytes(filename, bytes_list):
    """
    Write bytes list to file
    :param filename: output file path
    :param bytes_list: list of bytes to write
    """
    while bytes_list and bytes_list[-1] == 256:
        bytes_list = bytes_list[:-1]

    byte_array = bytearray(bytes_list)
    with open(filename, "wb") as f:
        f.write(byte_array)


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


def load_bgpt_model(checkpoint_path, device):
    """
    Load bGPT model from checkpoint
    """
    print("Loading bGPT model...")

    patch_config = GPT2Config(
        num_hidden_layers=CompressionConfig.PATCH_NUM_LAYERS,
        max_length=CompressionConfig.PATCH_LENGTH,
        max_position_embeddings=CompressionConfig.PATCH_LENGTH,
        hidden_size=CompressionConfig.HIDDEN_SIZE,
        n_head=CompressionConfig.HIDDEN_SIZE // 64,
        vocab_size=1,
    )
    byte_config = GPT2Config(
        num_hidden_layers=CompressionConfig.BYTE_NUM_LAYERS,
        max_length=CompressionConfig.PATCH_SIZE + 1,
        max_position_embeddings=CompressionConfig.PATCH_SIZE + 1,
        hidden_size=CompressionConfig.HIDDEN_SIZE,
        n_head=CompressionConfig.HIDDEN_SIZE // 64,
        vocab_size=256 + 1,
    )
    llm = bGPTLMHeadModel(patch_config, byte_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    llm.load_state_dict(checkpoint["model"], strict=False)
    llm = llm.to(device)
    llm.eval()

    print("Loaded bGPT model.")
    return llm


def load_dataset(dataset_path) -> List[str]:
    """
    Load dataset file paths for compression testing.

    :param dataset_path: glob pattern for dataset files
    :return: list of file paths
    """
    print("Loading dataset file list for compression testing...")

    fs = sorted(glob(dataset_path))

    max_files = CompressionConfig.MAX_FILES
    if max_files is not None:
        fs = fs[:max_files]

    print(f"Loaded {len(fs)} files for compression testing.")
    return fs


# ==================== Refactored Workflow Helpers ====================
def trim_virtual_padding(bytes_list: List[int]) -> List[int]:
    """
    Remove trailing virtual padding token 256.
    IMPORTANT:
    - This is only for comparison / verification.
    - It does NOT change the encode/decode padding logic.
    """
    out = list(bytes_list)
    while out and out[-1] == 256:
        out.pop()
    return out


def prepare_segment_for_compression(model, padded_segment, device):
    """
    Prepare one padded segment for compression.
    IMPORTANT:
    - Keep the original padding semantics unchanged.
    - This function only extracts duplicated code.
    """
    with torch.inference_mode():
        attention_mask = padded_segment["masks"]
        input_ids = padded_segment["patches"]

        output = model(patches=input_ids, masks=attention_mask)
        logits = output.logits

        logits = logits[:-1, :-1, :]
        logits = logits.reshape(1, -1, 257)

        start_patch = input_ids[:, :CompressionConfig.PATCH_SIZE].squeeze(0)

        input_ids = input_ids[
            :, CompressionConfig.PATCH_SIZE:-CompressionConfig.PATCH_SIZE
        ]
        input_ids = torch.cat(
            [torch.tensor([[256]], dtype=torch.long, device=device), input_ids],
            dim=1,
        )

    return logits, start_patch, input_ids


def compress_segment_file(model, device, segment_file: str, compressed_path: str):
    """
    Compress one segment file and write compressed result to disk.

    IMPORTANT:
    - read_bytes / pad_input_for_bgpt / bgpt_compress remain unchanged.
    """
    bytes_list, ext = read_bytes(segment_file)
    padded_segment = pad_input_for_bgpt([bytes_list], [ext], device)
    metric = Metric()

    logits, start_patch, input_ids = prepare_segment_for_compression(
        model, padded_segment, device
    )

    compressed_bytes, num_padded_bits, _, sequence_array, pd, probs = bgpt_compress(
        input_ids, logits, metric=metric
    )

    original_length = input_ids.shape[1] - 1
    write_padded_bytes(
        compressed_path,
        compressed_bytes,
        num_padded_bits,
        original_length
    )

    return {
        "metric": metric,
        "start_patch": start_patch,
        "ext": ext,
        "original_length": original_length,
        "sequence_array": sequence_array,
        "pd": pd,
        "probs": probs,
    }


def decompress_segment_file(
    model,
    device,
    compressed_path: str,
    start_patch,
    ext,
    do_test: bool = False,
    sequence_array=None,
    pd=None,
    probs=None,
):
    """
    Decompress one segment file from disk.

    IMPORTANT:
    - Keep bgpt_decode exactly in the original calling style.
    """
    compressed_bytes, num_padded_bits, original_length = read_padded_bytes(compressed_path)

    decompressed_tensor = bgpt_decode(
        compressed_bytes,
        num_padded_bits,
        model,
        start_patch,
        ext,
        device,
        original_length,
        original_sequence=sequence_array,
        pd=pd,
        probs=probs,
        do_test=do_test,
    )

    decompressed_bytes = decompressed_tensor.squeeze(0).cpu().numpy().tolist()
    return decompressed_bytes, original_length


def verify_segment_match(segment_file: str, decompressed_bytes: List[int]):
    """
    Verify whether decompressed bytes match the original segment file.
    """
    original_bytes, _ = read_bytes(segment_file)

    original_trimmed = trim_virtual_padding(original_bytes)
    decompressed_trimmed = trim_virtual_padding(decompressed_bytes)

    if original_trimmed == decompressed_trimmed:
        return True, None

    min_len = min(len(original_trimmed), len(decompressed_trimmed))
    for i in range(min_len):
        if original_trimmed[i] != decompressed_trimmed[i]:
            return False, (
                f"First difference at byte {i}: "
                f"{original_trimmed[i]} vs {decompressed_trimmed[i]}"
            )

    if len(original_trimmed) != len(decompressed_trimmed):
        return False, (
            f"Different lengths after trimming: "
            f"{len(original_trimmed)} vs {len(decompressed_trimmed)}"
        )

    return False, "Unknown mismatch"


def split_file_to_byte_segments(file_path, output_folder, segment_bytes=None):
    """
    Split a generic file into byte segments.
    """
    if segment_bytes is None:
        segment_bytes = CompressionConfig.GENERIC_SEGMENT_BYTES

    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(file_path)
    name_without_ext, ext = os.path.splitext(filename)
    ext = ext.lstrip(".")

    seg_folder = os.path.join(output_folder, name_without_ext)
    os.makedirs(seg_folder, exist_ok=True)

    with open(file_path, "rb") as f:
        data = f.read()

    segment_files = []
    for idx, start in enumerate(range(0, len(data), segment_bytes)):
        chunk = data[start:start + segment_bytes]
        seg_path = os.path.join(seg_folder, f"segment_{idx:04d}.{ext}")
        with open(seg_path, "wb") as f:
            f.write(chunk)
        segment_files.append(seg_path)

    return segment_files


def prepare_segments_for_file(file_path, temp_split_root):
    """
    Split one file into segments according to CompressionConfig.DATA_TYPE.

    Returns:
        segment_files: List[str]
        extra_info: dict
            - image: {"type": "image", "patch_size": ...}
            - audio: {"type": "audio", "wav_params": ...}
            - generic: {"type": "generic"}
    """
    os.makedirs(temp_split_root, exist_ok=True)

    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]

    if CompressionConfig.DATA_TYPE == "image":
        # 保持和你原来的 split_bmp_to_patches 调用方式一致
        split_bmp_to_patches(
            source_folder=os.path.dirname(file_path),
            output_folder=temp_split_root,
            patch_size=CompressionConfig.BMP_PATCH_SIZE,
        )
        segment_files = sorted(
            glob(os.path.join(temp_split_root, name_without_ext, "*.bmp"))
        )
        extra_info = {
            "type": "image",
            "patch_size": CompressionConfig.BMP_PATCH_SIZE,
        }
        return segment_files, extra_info

    elif CompressionConfig.DATA_TYPE == "audio":
        chunk_files, wav_params = split_wav_to_chunks(
            file_path,
            temp_split_root,
            chunk_duration=CompressionConfig.AUDIO_CHUNK_DURATION,
        )
        extra_info = {
            "type": "audio",
            "wav_params": wav_params,
        }
        return sorted(chunk_files), extra_info

    else:
        segment_files = split_file_to_byte_segments(
            file_path,
            temp_split_root,
            segment_bytes=CompressionConfig.GENERIC_SEGMENT_BYTES,
        )
        extra_info = {
            "type": "generic",
        }
        return sorted(segment_files), extra_info


def process_segment_files(
    model,
    device,
    segment_files: List[str],
    compressed_folder: str,
    decompressed_folder: Optional[str] = None,
    max_seg: Optional[int] = None,
    do_test: bool = False,
    decode_segments: bool = True,
    desc: str = "Processing segments",
):
    """
    Compress/decompress a list of segment files.

    IMPORTANT:
    - This function only removes duplicated workflow code.
    - Padding-related functions remain unchanged.
    """
    segment_files = sorted(segment_files)
    if max_seg is not None:
        segment_files = segment_files[:max_seg]

    os.makedirs(compressed_folder, exist_ok=True)
    if decode_segments and decompressed_folder is not None:
        os.makedirs(decompressed_folder, exist_ok=True)

    aggregate_metric = Metric()
    results = []
    decompressed_files = []

    for seg_idx, segment_file in enumerate(tqdm(segment_files, desc=desc)):
        seg_name = os.path.basename(segment_file)
        seg_stem, seg_ext = os.path.splitext(seg_name)

        compressed_path = os.path.join(compressed_folder, f"{seg_stem}.bin")
        decompressed_path = (
            os.path.join(decompressed_folder, f"{seg_stem}{seg_ext}")
            if (decode_segments and decompressed_folder is not None)
            else None
        )

        t0 = time.time()
        compress_info = compress_segment_file(
            model=model,
            device=device,
            segment_file=segment_file,
            compressed_path=compressed_path,
        )
        t1 = time.time()

        decompressed_bytes = None
        original_length = compress_info["original_length"]
        is_match = None
        mismatch_info = None

        if decode_segments:
            decompressed_bytes, original_length = decompress_segment_file(
                model=model,
                device=device,
                compressed_path=compressed_path,
                start_patch=compress_info["start_patch"],
                ext=compress_info["ext"],
                do_test=do_test,
                sequence_array=compress_info["sequence_array"] if do_test else None,
                pd=compress_info["pd"] if do_test else None,
                probs=compress_info["probs"] if do_test else None,
            )
            t2 = time.time()

            if decompressed_path is not None:
                write_bytes(decompressed_path, decompressed_bytes)
                decompressed_files.append(decompressed_path)

            is_match, mismatch_info = verify_segment_match(segment_file, decompressed_bytes)
        else:
            t2 = t1

        aggregate_metric.accumulate(
            compress_info["metric"].compressed_length,
            compress_info["metric"].total_length,
        )

        results.append({
            "segment_index": seg_idx,
            "segment_file": segment_file,
            "compressed_path": compressed_path,
            "decompressed_path": decompressed_path,
            "is_match": is_match,
            "mismatch_info": mismatch_info,
            "original_length": original_length,
            "compressed_length": compress_info["metric"].compressed_length,
            "compression_time": t1 - t0,
            "decompression_time": t2 - t1,
        })

    return {
        "segment_files": segment_files,
        "decompressed_files": decompressed_files,
        "results": results,
        "metric": aggregate_metric,
    }


# ==================== Workflows ====================
def test_workflow(model, dataset_files, device, output_path, temp_folder="temp_workflow"):
    """
    Run segmented workflow test.

    Difference from old version:
    - Do NOT feed the whole file directly into the model.
    - Split / patch first.
    - Then test up to MAX_SEG segments per file.
    """
    print("\n" + "=" * 80)
    print("Running Segmented Workflow Test")
    print("=" * 80)
    print(f"DATA_TYPE: {CompressionConfig.DATA_TYPE}")
    print(f"MAX_FILES: {CompressionConfig.MAX_FILES}")
    print(f"MAX_SEG: {CompressionConfig.MAX_SEG}")

    output_root = os.path.splitext(output_path)[0]
    split_root = os.path.join(temp_folder, "split")
    compressed_root = os.path.join(output_root, "compressed")
    decompressed_root = os.path.join(output_root, "decompressed")

    os.makedirs(split_root, exist_ok=True)
    os.makedirs(compressed_root, exist_ok=True)
    os.makedirs(decompressed_root, exist_ok=True)

    total_metric = Metric()

    for file_path in dataset_files:
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"\nProcessing file: {filename}")
        print("-" * 80)

        segment_files, _ = prepare_segments_for_file(file_path, split_root)

        if not segment_files:
            print(f"No segments generated for {filename}, skip.")
            continue

        print(f"Total segments after split: {len(segment_files)}")
        if CompressionConfig.MAX_SEG is not None:
            print(f"Segments to test: {min(len(segment_files), CompressionConfig.MAX_SEG)}")

        file_compressed_root = os.path.join(compressed_root, name_without_ext)
        file_decompressed_root = os.path.join(decompressed_root, name_without_ext)

        result = process_segment_files(
            model=model,
            device=device,
            segment_files=segment_files,
            compressed_folder=file_compressed_root,
            decompressed_folder=file_decompressed_root,
            max_seg=CompressionConfig.MAX_SEG,
            do_test=False,
            decode_segments=True,
            desc=f"Testing segments of {filename}",
        )

        success_count = sum(1 for r in result["results"] if r["is_match"])
        tested_count = len(result["results"])

        total_metric.accumulate(
            result["metric"].compressed_length,
            result["metric"].total_length,
        )

        for r in result["results"]:
            status = "OK" if r["is_match"] else "FAILED"
            print(
                f"[Segment {r['segment_index']}] {status} | "
                f"orig={r['original_length']} bytes, "
                f"comp={r['compressed_length']} bytes, "
                f"t_enc={r['compression_time']:.2f}s, "
                f"t_dec={r['decompression_time']:.2f}s"
            )
            if not r["is_match"] and r["mismatch_info"] is not None:
                print(f"  {r['mismatch_info']}")

        print("-" * 80)
        print(f"File done: {filename}")
        print(f"Successful segments: {success_count}/{tested_count}")
        print(f"File compression ratio/rate: {result['metric'].compute_ratio()}")
        print("=" * 80)

    print("\nSegmented workflow test completed!")
    print(f"Compressed outputs saved to: {compressed_root}")
    print(f"Decompressed outputs saved to: {decompressed_root}")
    print(f"Total compression ratio/rate: {total_metric.compute_ratio()}")


def test_bmp_compression(
    model,
    device,
    temp_folder: str = "temp",
    output_folder: str = "output",
    patch_size: int = None,
    decode_segments: bool = True,
):
    """
    Test BMP file compression workflow.

    decode_segments=True:
        compress + decompress + merge + verify
    decode_segments=False:
        only compress patches, skip decode / merge / verify
    """
    if patch_size is None:
        patch_size = CompressionConfig.BMP_PATCH_SIZE

    original_data_type = CompressionConfig.DATA_TYPE
    original_patch_size = CompressionConfig.BMP_PATCH_SIZE

    CompressionConfig.DATA_TYPE = "image"
    CompressionConfig.BMP_PATCH_SIZE = patch_size

    dataset_path = CompressionConfig.get_dataset_path()
    bmp_files = sorted(glob(dataset_path))

    if not bmp_files:
        print(f"No BMP files found in {dataset_path}")
        CompressionConfig.DATA_TYPE = original_data_type
        CompressionConfig.BMP_PATCH_SIZE = original_patch_size
        return

    print(f"Found {len(bmp_files)} BMP files to test")
    print(f"Dataset path: {dataset_path}")
    print(f"Decode segments: {decode_segments}")
    print("=" * 80)

    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    split_folder = os.path.join(temp_folder, "split")
    compressed_folder = os.path.join(temp_folder, "compressed")
    decompressed_folder = os.path.join(temp_folder, "decompressed")

    os.makedirs(split_folder, exist_ok=True)
    os.makedirs(compressed_folder, exist_ok=True)
    if decode_segments:
        os.makedirs(decompressed_folder, exist_ok=True)

    total_metric = Metric()

    for bmp_file in bmp_files:
        print(f"\nProcessing: {os.path.basename(bmp_file)}")
        print("-" * 80)

        filename = os.path.basename(bmp_file)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"Step 1: Splitting BMP into {patch_size}x{patch_size} patches...")
        patch_files, extra_info = prepare_segments_for_file(bmp_file, split_folder)
        print(f"Created {len(patch_files)} patches")

        print(f"\nStep 2: Compressing patches" + (" and decoding..." if decode_segments else "..."))
        file_compressed_root = os.path.join(compressed_folder, name_without_ext)
        file_decompressed_root = (
            os.path.join(decompressed_folder, name_without_ext)
            if decode_segments else None
        )

        result = process_segment_files(
            model=model,
            device=device,
            segment_files=patch_files,
            compressed_folder=file_compressed_root,
            decompressed_folder=file_decompressed_root,
            max_seg=None,
            do_test=False,
            decode_segments=decode_segments,
            desc=f"Processing patches of {filename}",
        )

        total_metric.accumulate(
            result["metric"].compressed_length,
            result["metric"].total_length,
        )

        for r in result["results"]:
            if decode_segments:
                status = "OK" if r["is_match"] else "FAILED"
                print(
                    f"[Patch {r['segment_index']}] {status} | "
                    f"orig={r['original_length']} bytes, "
                    f"comp={r['compressed_length']} bytes, "
                    f"t_enc={r['compression_time']:.2f}s, "
                    f"t_dec={r['decompression_time']:.2f}s"
                )
                if not r["is_match"] and r["mismatch_info"] is not None:
                    print(f"  {r['mismatch_info']}")
            else:
                print(
                    f"[Patch {r['segment_index']}] COMPRESSED | "
                    f"orig={r['original_length']} bytes, "
                    f"comp={r['compressed_length']} bytes, "
                    f"t_enc={r['compression_time']:.2f}s"
                )

        print("Compression ratio/rate:", result["metric"].compute_ratio())

        if decode_segments:
            print(f"\nStep 3: Merging patches back to original image...")
            reconstructed_path = os.path.join(output_folder, f"reconstructed_{filename}")
            merge_patches_to_bmp(
                patches_folder=file_decompressed_root,
                output_path=reconstructed_path,
                patch_size=extra_info["patch_size"],
            )

            print(f"\nStep 4: Verifying reconstruction...")
            original_bytes, _ = read_bytes(bmp_file)
            reconstructed_bytes, _ = read_bytes(reconstructed_path)

            original_bytes = trim_virtual_padding(original_bytes)
            reconstructed_bytes = trim_virtual_padding(reconstructed_bytes)

            if original_bytes == reconstructed_bytes:
                print("✓ Reconstruction successful! Files match perfectly.")
            else:
                print("✗ Warning: Reconstructed file differs from original")
                print(f"  Original size: {len(original_bytes)} bytes")
                print(f"  Reconstructed size: {len(reconstructed_bytes)} bytes")

                min_len = min(len(original_bytes), len(reconstructed_bytes))
                for i in range(min_len):
                    if original_bytes[i] != reconstructed_bytes[i]:
                        print(
                            f"  First difference at byte {i}: "
                            f"{original_bytes[i]} vs {reconstructed_bytes[i]}"
                        )
                        break
        else:
            print("\nDecode disabled: skipping patch decoding, merge, and reconstruction verification.")

        print("=" * 80)

    CompressionConfig.DATA_TYPE = original_data_type
    CompressionConfig.BMP_PATCH_SIZE = original_patch_size

    print(f"\nBMP compression test completed!")
    print(f"Compressed patch files saved to: {compressed_folder}")
    if decode_segments:
        print(f"Reconstructed images saved to: {output_folder}")
    print("Compression ratio/rate:", total_metric.compute_ratio())


def test_wav_compression(
    model,
    device,
    temp_folder: str = "temp_audio",
    output_folder: str = "output_audio",
    chunk_duration: float = None,
    decode_segments: bool = True,
):
    """
    Test WAV file compression workflow.

    decode_segments=True:
        compress + decompress + merge + verify
    decode_segments=False:
        only compress chunks, skip decode / merge / verify
    """
    if chunk_duration is None:
        chunk_duration = CompressionConfig.AUDIO_CHUNK_DURATION

    original_data_type = CompressionConfig.DATA_TYPE
    original_chunk_duration = CompressionConfig.AUDIO_CHUNK_DURATION

    CompressionConfig.DATA_TYPE = "audio"
    CompressionConfig.AUDIO_CHUNK_DURATION = chunk_duration

    dataset_path = CompressionConfig.get_dataset_path()
    wav_files = sorted(glob(dataset_path))

    if not wav_files:
        print(f"No WAV files found in {dataset_path}")
        CompressionConfig.DATA_TYPE = original_data_type
        CompressionConfig.AUDIO_CHUNK_DURATION = original_chunk_duration
        return

    print(f"Found {len(wav_files)} WAV files to test")
    print(f"Dataset path: {dataset_path}")
    print(f"Chunk duration: {chunk_duration} seconds")
    print(f"Decode segments: {decode_segments}")
    print("=" * 80)

    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    split_folder = os.path.join(temp_folder, "split")
    compressed_folder = os.path.join(temp_folder, "compressed")
    decompressed_folder = os.path.join(temp_folder, "decompressed")

    os.makedirs(split_folder, exist_ok=True)
    os.makedirs(compressed_folder, exist_ok=True)
    if decode_segments:
        os.makedirs(decompressed_folder, exist_ok=True)

    total_metric = Metric()

    for wav_file in wav_files:
        print(f"\nProcessing: {os.path.basename(wav_file)}")
        print("-" * 80)

        filename = os.path.basename(wav_file)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"Step 1: Splitting WAV into {chunk_duration}s chunks...")
        chunk_files, extra_info = prepare_segments_for_file(wav_file, split_folder)
        print(f"Created {len(chunk_files)} chunks")

        wav_params = extra_info["wav_params"]
        print(
            f"WAV parameters: {wav_params['nchannels']} channels, "
            f"{wav_params['sampwidth']} bytes/sample, "
            f"{wav_params['framerate']} Hz"
        )

        print(f"\nStep 2: Compressing chunks" + (" and decoding..." if decode_segments else "..."))
        file_compressed_root = os.path.join(compressed_folder, name_without_ext)
        file_decompressed_root = (
            os.path.join(decompressed_folder, name_without_ext)
            if decode_segments else None
        )

        result = process_segment_files(
            model=model,
            device=device,
            segment_files=chunk_files,
            compressed_folder=file_compressed_root,
            decompressed_folder=file_decompressed_root,
            max_seg=None,
            do_test=False,
            decode_segments=decode_segments,
            desc=f"Processing chunks of {filename}",
        )

        total_metric.accumulate(
            result["metric"].compressed_length,
            result["metric"].total_length,
        )

        for r in result["results"]:
            if decode_segments:
                status = "OK" if r["is_match"] else "FAILED"
                print(
                    f"[Chunk {r['segment_index']}] {status} | "
                    f"orig={r['original_length']} bytes, "
                    f"comp={r['compressed_length']} bytes, "
                    f"t_enc={r['compression_time']:.2f}s, "
                    f"t_dec={r['decompression_time']:.2f}s"
                )
                if not r["is_match"] and r["mismatch_info"] is not None:
                    print(f"  {r['mismatch_info']}")
            else:
                print(
                    f"[Chunk {r['segment_index']}] COMPRESSED | "
                    f"orig={r['original_length']} bytes, "
                    f"comp={r['compressed_length']} bytes, "
                    f"t_enc={r['compression_time']:.2f}s"
                )

        print("Compression ratio/rate:", result["metric"].compute_ratio())

        if decode_segments:
            print(f"\nStep 3: Merging chunks back to original audio...")
            reconstructed_path = os.path.join(output_folder, f"reconstructed_{filename}")
            merge_wav_chunks(
                result["decompressed_files"],
                reconstructed_path,
                wav_params,
            )

            print(f"\nStep 4: Verifying reconstruction...")
            original_bytes, _ = read_bytes(wav_file)
            reconstructed_bytes, _ = read_bytes(reconstructed_path)

            original_bytes = trim_virtual_padding(original_bytes)
            reconstructed_bytes = trim_virtual_padding(reconstructed_bytes)

            if original_bytes == reconstructed_bytes:
                print("✓ Reconstruction successful! Files match perfectly.")
                print(f"  File size: {len(original_bytes)} bytes")
            else:
                print("✗ Warning: Reconstructed file differs from original")
                print(f"  Original size: {len(original_bytes)} bytes")
                print(f"  Reconstructed size: {len(reconstructed_bytes)} bytes")

                min_len = min(len(original_bytes), len(reconstructed_bytes))
                for i in range(min_len):
                    if original_bytes[i] != reconstructed_bytes[i]:
                        print(
                            f"  First difference at byte {i}: "
                            f"{original_bytes[i]} vs {reconstructed_bytes[i]}"
                        )
                        break
        else:
            print("\nDecode disabled: skipping chunk decoding, merge, and reconstruction verification.")

        print("=" * 80)

    CompressionConfig.DATA_TYPE = original_data_type
    CompressionConfig.AUDIO_CHUNK_DURATION = original_chunk_duration

    print(f"\nWAV compression test completed!")
    print(f"Total files processed: {len(wav_files)}")
    print(f"Compressed chunk files saved to: {compressed_folder}")
    if decode_segments:
        print(f"Reconstructed audio files saved to: {output_folder}")
    print("Compression ratio/rate:", total_metric.compute_ratio())


# ==================== Main ====================
if __name__ == "__main__":
    device = torch.device(CompressionConfig.DEVICE)

    model_checkpoint = CompressionConfig.get_model_checkpoint()
    llm = load_bgpt_model(model_checkpoint, device)

    # Option 1: Run segmented workflow test
    print("\n" + "=" * 80)
    print("Running Standard Workflow Test")
    print("=" * 80)
    dataset_path = CompressionConfig.get_dataset_path()
    dataset_files = load_dataset(dataset_path)
    test_workflow(llm, dataset_files, device, CompressionConfig.COMPRESSED_OUTPUT)

    """
    # Option 2: Run BMP compression test
    print("\n" + "=" * 80)
    print("Running BMP Compression Test")
    print("=" * 80)
    test_bmp_compression(
        model=llm,
        device=device,
        temp_folder="temp_img",
        output_folder="output_img",
        patch_size=CompressionConfig.BMP_PATCH_SIZE,
        decode_segments=True,   # False 时只压缩不解码
    )

    # Option 3: Run WAV compression test with chunking
    print("\n" + "=" * 80)
    print("Running WAV Compression Test (with chunking)")
    print("=" * 80)
    test_wav_compression(
        model=llm,
        device=device,
        temp_folder="temp_audio",
        output_folder="output_audio",
        chunk_duration=CompressionConfig.AUDIO_CHUNK_DURATION,
        decode_segments=True,   # False 时只压缩不解码
    )
    """