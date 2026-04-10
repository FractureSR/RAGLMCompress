import os
import re
from glob import glob
from typing import Optional, Dict, Any, Tuple

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Audio


# =========================================================
# Basic Utils
# =========================================================

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """
    Make filename safe for most filesystems.
    """
    name = str(name)
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = name.replace("\n", "_").replace("\r", "_").strip()
    return name


def get_audio_info(file_path: str):
    """
    Display audio file information.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        print(f"\nAudio Info for: {os.path.basename(file_path)}")
        print(f"  Sample Rate: {audio.frame_rate} Hz")
        print(f"  Channels: {audio.channels}")
        print(f"  Sample Width: {audio.sample_width} bytes ({audio.sample_width * 8} bit)")
        print(f"  Duration: {len(audio) / 1000:.2f} seconds")
        print(f"  Frame Count: {audio.frame_count()}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


# =========================================================
# FLAC -> WAV
# =========================================================

def convert_flac_to_wav(
    input_path: str,
    output_path: str,
    target_sampling_rate: int = 8000,
    channels: int = 1,
    sample_width_bytes: int = 1,
):
    """
    Convert a single FLAC file to WAV with specified format.

    :param input_path: Path to input FLAC file
    :param output_path: Path to save output WAV file
    :param target_sampling_rate: Target sample rate, e.g. 8000
    :param channels: Number of channels, e.g. 1 for mono
    :param sample_width_bytes: Bytes per sample, e.g. 1 => 8-bit, 2 => 16-bit
    """
    try:
        audio = AudioSegment.from_file(input_path, format="flac")
        audio = audio.set_frame_rate(target_sampling_rate)
        audio = audio.set_channels(channels)
        audio = audio.set_sample_width(sample_width_bytes)

        ensure_dir(os.path.dirname(output_path))
        audio.export(output_path, format="wav")

    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        raise


def find_all_flac_files(root_folder: str):
    """
    Recursively find all FLAC files.
    """
    flac_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".flac"):
                flac_files.append(os.path.join(dirpath, filename))
    return flac_files


def convert_flac_folder_to_wav(
    source_folder: str,
    output_folder: str,
    target_sampling_rate: int = 8000,
    channels: int = 1,
    sample_width_bytes: int = 1,
    preserve_structure: bool = True,
    max_files: Optional[int] = None,
):
    """
    Convert all FLAC files in a folder recursively to WAV.

    :param source_folder: Root folder containing FLAC files
    :param output_folder: Folder to save WAV files
    :param target_sampling_rate: Target sample rate
    :param channels: Number of channels
    :param sample_width_bytes: Bytes per sample
    :param preserve_structure: Whether to preserve original folder structure
    :param max_files: Export only first N files if set
    """
    ensure_dir(output_folder)

    print("Searching for FLAC files...")
    flac_files = find_all_flac_files(source_folder)

    if max_files is not None:
        flac_files = flac_files[:max_files]

    if not flac_files:
        print(f"No FLAC files found in {source_folder}")
        return

    print(f"Found {len(flac_files)} FLAC files")
    print(f"Converting to WAV...")
    print(f"Source: {source_folder}")
    print(f"Output: {output_folder}")
    print(f"Target SR: {target_sampling_rate}")
    print(f"Channels: {channels}")
    print(f"Sample width: {sample_width_bytes} bytes")
    print(f"Preserve structure: {preserve_structure}")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for filepath in tqdm(flac_files, desc="Converting FLAC"):
        try:
            filename = os.path.basename(filepath)
            name_without_ext = os.path.splitext(filename)[0]

            if preserve_structure:
                relative_path = os.path.relpath(filepath, source_folder)
                relative_dir = os.path.dirname(relative_path)
                output_subdir = os.path.join(output_folder, relative_dir)
                output_path = os.path.join(output_subdir, f"{name_without_ext}.wav")
            else:
                output_path = os.path.join(output_folder, f"{name_without_ext}.wav")

            convert_flac_to_wav(
                input_path=filepath,
                output_path=output_path,
                target_sampling_rate=target_sampling_rate,
                channels=channels,
                sample_width_bytes=sample_width_bytes,
            )
            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"\nError processing {filepath}: {e}")

    print("=" * 60)
    print("FLAC -> WAV conversion complete!")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output location: {output_folder}")


# =========================================================
# WAV Segment Extraction
# =========================================================

def extract_wav_segment(
    input_path: str,
    output_path: str,
    duration_seconds: float,
    start_time: float = 0.0,
):
    """
    Extract a segment from a WAV file.

    :param input_path: Path to input WAV file
    :param output_path: Path to save output WAV segment
    :param duration_seconds: Duration to extract
    :param start_time: Start time in seconds
    """
    try:
        audio = AudioSegment.from_file(input_path, format="wav")

        total_duration_ms = len(audio)
        total_duration_s = total_duration_ms / 1000.0

        start_ms = int(start_time * 1000)
        end_ms = int((start_time + duration_seconds) * 1000)

        if start_time < 0:
            raise ValueError(f"start_time must be non-negative, got {start_time}")

        if start_ms >= total_duration_ms:
            raise ValueError(
                f"start_time ({start_time}s) is beyond audio duration ({total_duration_s:.2f}s)"
            )

        if end_ms > total_duration_ms:
            print(
                f"Warning: Requested segment exceeds audio length. "
                f"Will extract until end of file."
            )
            end_ms = total_duration_ms

        segment = audio[start_ms:end_ms]

        ensure_dir(os.path.dirname(output_path))
        segment.export(output_path, format="wav")

        actual_duration = len(segment) / 1000.0
        print(f"Extracted segment:")
        print(f"  Input: {os.path.basename(input_path)}")
        print(f"  Output: {os.path.basename(output_path)}")
        print(f"  Start time: {start_time:.2f}s")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Sample rate: {segment.frame_rate} Hz")
        print(f"  Channels: {segment.channels}")
        print(f"  Sample width: {segment.sample_width} bytes ({segment.sample_width * 8} bit)")

    except Exception as e:
        print(f"Error extracting segment from {input_path}: {e}")
        raise


def batch_extract_wav_segments(
    input_folder: str,
    output_folder: str,
    duration_seconds: float,
    start_time: float = 0.0,
    pattern: str = "*.wav",
    max_files: Optional[int] = None,
):
    """
    Extract segments from multiple WAV files.

    :param input_folder: Folder containing WAV files
    :param output_folder: Folder to save segments
    :param duration_seconds: Duration of segment
    :param start_time: Start time in seconds
    :param pattern: Match pattern
    :param max_files: Only process first N files if set
    """
    ensure_dir(output_folder)

    search_pattern = os.path.join(input_folder, pattern)
    wav_files = sorted(glob(search_pattern))

    if max_files is not None:
        wav_files = wav_files[:max_files]

    if not wav_files:
        print(f"No files matching '{pattern}' found in {input_folder}")
        return

    print(f"Found {len(wav_files)} WAV files")
    print(f"Extracting {duration_seconds}s segments starting at {start_time}s...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for filepath in tqdm(wav_files, desc="Extracting segments"):
        try:
            filename = os.path.basename(filepath)
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{name_without_ext}_segment.wav")

            extract_wav_segment(
                input_path=filepath,
                output_path=output_path,
                duration_seconds=duration_seconds,
                start_time=start_time,
            )
            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"\nError processing {filepath}: {e}\n")

    print("=" * 60)
    print("Extraction complete!")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output location: {output_folder}")


# =========================================================
# Hugging Face Datasets -> WAV
# =========================================================

def _hf_audio_to_array_and_sr(audio_obj) -> Tuple[np.ndarray, int]:
    """
    Convert Hugging Face audio object into (numpy array, sampling_rate).

    Compatible with:
    1. Old style dict:
       {"array": ..., "sampling_rate": ..., "path": ...}
    2. Newer AudioDecoder-like object with get_all_samples()
    """
    if isinstance(audio_obj, dict):
        if "array" not in audio_obj or "sampling_rate" not in audio_obj:
            raise ValueError("Audio dict missing 'array' or 'sampling_rate'")
        array = np.asarray(audio_obj["array"])
        sr = int(audio_obj["sampling_rate"])
        return array, sr

    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        data = samples.data
        sr = int(samples.sample_rate)

        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        else:
            data = np.asarray(data)

        # Often [channels, time] -> convert to [time, channels]
        if data.ndim == 2 and data.shape[0] <= 8 and data.shape[0] < data.shape[1]:
            data = data.T

        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]

        return data, sr

    raise TypeError(f"Unsupported audio object type: {type(audio_obj)}")


def _normalize_audio_shape(array: np.ndarray) -> np.ndarray:
    """
    Normalize audio shape for soundfile write:
    - mono: [time]
    - multi-channel: [time, channels]
    """
    array = np.asarray(array)

    if array.ndim == 1:
        return array

    if array.ndim != 2:
        raise ValueError(f"Unsupported audio array shape: {array.shape}")

    # If likely [channels, time], convert to [time, channels]
    if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
        array = array.T

    return array


def _convert_to_mono(array: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono by averaging channels.
    """
    array = _normalize_audio_shape(array)

    if array.ndim == 1:
        return array

    return array.mean(axis=1)


def _build_output_path(
    example: Dict[str, Any],
    idx: int,
    output_folder: str,
    file_name_column: Optional[str],
    audio_column: str,
    preserve_structure: bool,
) -> str:
    """
    Build output wav path.
    """
    audio_obj = example[audio_column]

    if file_name_column is not None and file_name_column in example and example[file_name_column] is not None:
        stem = sanitize_filename(example[file_name_column])
        return os.path.join(output_folder, f"{stem}.wav")

    if isinstance(audio_obj, dict) and audio_obj.get("path"):
        original_path = str(audio_obj["path"])
        stem = sanitize_filename(os.path.splitext(os.path.basename(original_path))[0])

        if preserve_structure:
            rel_dir = os.path.dirname(original_path).strip(os.sep)
            if rel_dir:
                rel_dir = rel_dir.replace("\\", "/")
                parts = [sanitize_filename(p) for p in rel_dir.split("/") if p.strip()]
                subdir = os.path.join(output_folder, *parts)
            else:
                subdir = output_folder
            return os.path.join(subdir, f"{stem}.wav")

        return os.path.join(output_folder, f"{stem}.wav")

    return os.path.join(output_folder, f"sample_{idx:08d}.wav")


def preprocess_hf_audio_dataset_to_wav(
    dataset_name_or_path: str,
    output_folder: str,
    split: str = "train",
    audio_column: str = "audio",
    target_sampling_rate: int = 8000,
    file_name_column: Optional[str] = None,
    mono: bool = True,
    wav_subtype: str = "PCM_16",
    max_files: Optional[int] = None,
    preserve_structure: bool = False,
    load_dataset_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Load audio from Hugging Face datasets and export as WAV files
    with the specified sampling rate.

    :param dataset_name_or_path:
        Can be:
        - a local dataset script / local dataset path for load_dataset
        - a dataset name
        - a folder saved by dataset.save_to_disk(...) if use_load_from_disk=True
    :param output_folder: Output WAV folder
    :param split: Dataset split, e.g. "train"
    :param audio_column: Audio column name
    :param target_sampling_rate: Output sampling rate
    :param file_name_column: Optional column for naming output files
    :param mono: Convert to mono if True
    :param wav_subtype: e.g. "PCM_16", "PCM_U8", "FLOAT"
    :param max_files: Only export first N samples if set
    :param preserve_structure: Try to preserve relative path structure if audio path exists
    :param use_load_from_disk: If True, use load_from_disk instead of load_dataset
    :param load_dataset_kwargs: Extra kwargs for load_dataset
    """
    if load_dataset_kwargs is None:
        load_dataset_kwargs = {}

    ensure_dir(output_folder)

    print("Loading dataset...")
    dataset = load_dataset(dataset_name_or_path, split=split, **load_dataset_kwargs)

    print(f"Loaded dataset with columns: {dataset.column_names}")

    if audio_column not in dataset.column_names:
        raise ValueError(
            f"audio_column='{audio_column}' not found. Available columns: {dataset.column_names}"
        )

    print(f"Casting audio column '{audio_column}' to sampling rate {target_sampling_rate} Hz...")
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=target_sampling_rate))

    total_len = len(dataset)
    if max_files is not None:
        total_len = min(total_len, max_files)

    print("=" * 60)
    print("HF Dataset -> WAV export")
    print(f"Dataset: {dataset_name_or_path}")
    print(f"Split: {split}")
    print(f"Audio column: {audio_column}")
    print(f"Target SR: {target_sampling_rate}")
    print(f"Mono: {mono}")
    print(f"WAV subtype: {wav_subtype}")
    print(f"Max files: {max_files}")
    print(f"Output folder: {output_folder}")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for idx, example in tqdm(enumerate(dataset), total=total_len, desc="Exporting WAV"):
        if max_files is not None and idx >= max_files:
            break

        try:
            audio_obj = example[audio_column]
            array, sr = _hf_audio_to_array_and_sr(audio_obj)

            if sr != target_sampling_rate:
                raise ValueError(
                    f"Unexpected sampling rate after cast_column: got {sr}, expected {target_sampling_rate}"
                )

            array = _normalize_audio_shape(array)

            if mono:
                array = _convert_to_mono(array)

            output_path = _build_output_path(
                example=example,
                idx=idx,
                output_folder=output_folder,
                file_name_column=file_name_column,
                audio_column=audio_column,
                preserve_structure=preserve_structure,
            )

            ensure_dir(os.path.dirname(output_path))

            sf.write(output_path, array, sr, format="WAV", subtype=wav_subtype)
            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"\nError processing sample {idx}: {e}")

    print("=" * 60)
    print("Export complete!")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output location: {output_folder}")
    print("=" * 60)


# =========================================================
# WAV Trim
# =========================================================

def trim_wav_file(
    input_path: str,
    output_path: str,
    target_duration_seconds: float,
    start_time: float = 0.0,
):
    """
    Trim a WAV file to a specified duration.

    :param input_path: Path to input WAV file
    :param output_path: Path to save trimmed WAV file
    :param target_duration_seconds: Target duration in seconds
    :param start_time: Start time in seconds, default 0.0
    """
    try:
        if target_duration_seconds <= 0:
            raise ValueError(
                f"target_duration_seconds must be positive, got {target_duration_seconds}"
            )
        if start_time < 0:
            raise ValueError(f"start_time must be non-negative, got {start_time}")

        audio = AudioSegment.from_file(input_path, format="wav")

        total_duration_ms = len(audio)
        total_duration_s = total_duration_ms / 1000.0

        start_ms = int(start_time * 1000)
        end_ms = int((start_time + target_duration_seconds) * 1000)

        if start_ms >= total_duration_ms:
            raise ValueError(
                f"start_time ({start_time}s) is beyond audio duration ({total_duration_s:.2f}s)"
            )

        if end_ms > total_duration_ms:
            print(
                f"Warning: {os.path.basename(input_path)} is shorter than requested "
                f"duration ({target_duration_seconds}s). "
                f"Will keep audio until end of file."
            )
            end_ms = total_duration_ms

        trimmed_audio = audio[start_ms:end_ms]

        ensure_dir(os.path.dirname(output_path))
        trimmed_audio.export(output_path, format="wav")

    except Exception as e:
        print(f"Error trimming {input_path}: {e}")
        raise


def batch_trim_wav_files(
    input_folder: str,
    output_folder: str,
    target_duration_seconds: float,
    start_time: float = 0.0,
    pattern: str = "*.wav",
    max_files: Optional[int] = None,
    preserve_structure: bool = False,
    overwrite: bool = False,
):
    """
    Trim all WAV files in a folder to a specified duration.

    :param input_folder: Folder containing WAV files
    :param output_folder: Folder to save trimmed WAV files
    :param target_duration_seconds: Target duration in seconds
    :param start_time: Start time in seconds
    :param pattern: Match pattern, default '*.wav'
    :param max_files: Only process first N files if set
    :param preserve_structure: Preserve relative folder structure
    :param overwrite: If True, overwrite source files (use carefully)
    """
    if not overwrite:
        ensure_dir(output_folder)

    # 支持递归查找
    search_pattern = os.path.join(input_folder, "**", pattern)
    wav_files = sorted(glob(search_pattern, recursive=True))

    if max_files is not None:
        wav_files = wav_files[:max_files]

    if not wav_files:
        print(f"No files matching '{pattern}' found in {input_folder}")
        return

    print(f"Found {len(wav_files)} WAV files")
    print(f"Trimming each file to {target_duration_seconds}s from {start_time}s...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder if not overwrite else '[overwrite source files]'}")
    print(f"Preserve structure: {preserve_structure}")
    print(f"Overwrite: {overwrite}")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for filepath in tqdm(wav_files, desc="Trimming WAV"):
        try:
            filename = os.path.basename(filepath)

            if overwrite:
                output_path = filepath
            else:
                if preserve_structure:
                    relative_path = os.path.relpath(filepath, input_folder)
                    output_path = os.path.join(output_folder, relative_path)
                else:
                    output_path = os.path.join(output_folder, filename)

            trim_wav_file(
                input_path=filepath,
                output_path=output_path,
                target_duration_seconds=target_duration_seconds,
                start_time=start_time,
            )
            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"\nError processing {filepath}: {e}\n")

    print("=" * 60)
    print("WAV trim complete!")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    if not overwrite:
        print(f"Output location: {output_folder}")
    print("=" * 60)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # -----------------------------------------------------
    # Choose mode:
    #   1) "hf_dataset"   : export WAV from Hugging Face datasets
    #   2) "flac_to_wav"  : convert FLAC folder to WAV
    #   3) "trim_wav"     : trim WAV files in a folder to a target duration
    # -----------------------------------------------------
    MODE = "trim_wav"

    # -----------------------------------------------------
    # Common config
    # -----------------------------------------------------
    MAX_FILES = None  # e.g. 100 or None

    # -----------------------------------------------------
    # Mode 1: HF dataset -> WAV
    # -----------------------------------------------------
    HF_DATASET_NAME_OR_PATH = "datasets/peoples_speech"
    HF_SPLIT = "train"
    HF_AUDIO_COLUMN = "audio"
    HF_FILE_NAME_COLUMN = None  # e.g. "id", "utterance_id", "file" if your dataset has it
    HF_OUTPUT_FOLDER = "datasets/peoples_speech/wav_8k"
    HF_TARGET_SAMPLING_RATE = 8000
    HF_MONO = True
    HF_WAV_SUBTYPE = "PCM_U8"   # 8-bit WAV; if you prefer 16-bit use "PCM_16"
    HF_PRESERVE_STRUCTURE = False
    HF_LOAD_DATASET_KWARGS = {}

    # -----------------------------------------------------
    # Mode 2: FLAC folder -> WAV
    # -----------------------------------------------------
    FLAC_SOURCE_FOLDER = "datasets/LibriSpeech/dev-clean"
    FLAC_OUTPUT_FOLDER = "datasets/LibriSpeech/wav"
    FLAC_TARGET_SAMPLING_RATE = 8000
    FLAC_CHANNELS = 1
    FLAC_SAMPLE_WIDTH_BYTES = 1
    FLAC_PRESERVE_STRUCTURE = False

    # -----------------------------------------------------
    # Mode 4: Trim WAV files
    # -----------------------------------------------------
    TRIM_INPUT_FOLDER = "datasets/peoples_speech/wav_8k"
    TRIM_OUTPUT_FOLDER = "datasets/peoples_speech/wav_8k_trimmed"
    TRIM_TARGET_DURATION_SECONDS = 1.0
    TRIM_START_TIME = 0.0
    TRIM_PATTERN = "*.wav"
    TRIM_PRESERVE_STRUCTURE = False
    TRIM_OVERWRITE = False

    # -----------------------------------------------------
    # Run
    # -----------------------------------------------------
    if MODE == "hf_dataset":
        preprocess_hf_audio_dataset_to_wav(
            dataset_name_or_path=HF_DATASET_NAME_OR_PATH,
            output_folder=HF_OUTPUT_FOLDER,
            split=HF_SPLIT,
            audio_column=HF_AUDIO_COLUMN,
            target_sampling_rate=HF_TARGET_SAMPLING_RATE,
            file_name_column=HF_FILE_NAME_COLUMN,
            mono=HF_MONO,
            wav_subtype=HF_WAV_SUBTYPE,
            max_files=MAX_FILES,
            preserve_structure=HF_PRESERVE_STRUCTURE,
            load_dataset_kwargs=HF_LOAD_DATASET_KWARGS,
        )

    elif MODE == "flac_to_wav":
        convert_flac_folder_to_wav(
            source_folder=FLAC_SOURCE_FOLDER,
            output_folder=FLAC_OUTPUT_FOLDER,
            target_sampling_rate=FLAC_TARGET_SAMPLING_RATE,
            channels=FLAC_CHANNELS,
            sample_width_bytes=FLAC_SAMPLE_WIDTH_BYTES,
            preserve_structure=FLAC_PRESERVE_STRUCTURE,
            max_files=MAX_FILES,
        )
    
    elif MODE == "trim_wav":
        batch_trim_wav_files(
            input_folder=TRIM_INPUT_FOLDER,
            output_folder=TRIM_OUTPUT_FOLDER,
            target_duration_seconds=TRIM_TARGET_DURATION_SECONDS,
            start_time=TRIM_START_TIME,
            pattern=TRIM_PATTERN,
            max_files=MAX_FILES,
            preserve_structure=TRIM_PRESERVE_STRUCTURE,
            overwrite=TRIM_OVERWRITE,
        )
