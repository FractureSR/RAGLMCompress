import argparse
import os
from glob import glob

import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm


def _ensure_output_dir(output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def _convert_flac_to_wav(
    input_path: str,
    output_path: str,
    sample_rate: int,
    sample_width: int,
) -> None:
    """
    Convert a single FLAC file to WAV with the requested sample rate/width.

    :param input_path: Path to input FLAC file
    :param output_path: Path to save output WAV file
    :param sample_rate: Target sample rate in Hz
    :param sample_width: Target sample width in bytes
    """
    try:
        audio, original_sample_rate = sf.read(input_path, always_2d=True)
        audio = audio.astype(np.float32)
        audio = audio.mean(axis=1)

        if original_sample_rate != sample_rate:
            audio = signal.resample_poly(audio, sample_rate, original_sample_rate)

        subtype_map = {
            1: "PCM_U8",
            2: "PCM_16",
            4: "PCM_32",
        }
        if sample_width not in subtype_map:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        _ensure_output_dir(output_path)
        sf.write(
            output_path,
            audio,
            sample_rate,
            format="WAV",
            subtype=subtype_map[sample_width],
        )
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        raise


def convert_flac_to_8k_wav(input_path: str, output_path: str) -> None:
    """
    Convert a single FLAC file to 8000Hz, mono, 8-bit WAV.
    """
    _convert_flac_to_wav(
        input_path=input_path,
        output_path=output_path,
        sample_rate=8000,
        sample_width=1,
    )


def convert_flac_to_16k_wav(input_path: str, output_path: str) -> None:
    """
    Convert a single FLAC file to 16000Hz, mono, 16-bit WAV.
    """
    _convert_flac_to_wav(
        input_path=input_path,
        output_path=output_path,
        sample_rate=16000,
        sample_width=2,
    )


def convert_flac_to_24k_wav(input_path: str, output_path: str) -> None:
    """
    Convert a single FLAC file to 24000Hz, mono, 16-bit WAV.
    """
    _convert_flac_to_wav(
        input_path=input_path,
        output_path=output_path,
        sample_rate=24000,
        sample_width=2,
    )


def convert_flac_to_32k_wav(input_path: str, output_path: str) -> None:
    """
    Convert a single FLAC file to 32000Hz, mono, 32-bit WAV.
    """
    _convert_flac_to_wav(
        input_path=input_path,
        output_path=output_path,
        sample_rate=32000,
        sample_width=4,
    )


def find_all_flac_files(root_folder: str):
    """
    Recursively find all FLAC files in a folder and its subfolders.

    :param root_folder: Root folder to search
    :return: Sorted list of paths to FLAC files
    """
    flac_files = []

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".flac"):
                flac_files.append(os.path.join(dirpath, filename))

    return sorted(flac_files)


def _build_output_path(
    filepath: str,
    source_folder: str,
    output_folder: str,
    preserve_structure: bool,
) -> str:
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]

    if preserve_structure:
        relative_path = os.path.relpath(filepath, source_folder)
        relative_dir = os.path.dirname(relative_path)
        return os.path.join(output_folder, relative_dir, f"{name_without_ext}.wav")

    return os.path.join(output_folder, f"{name_without_ext}.wav")


def _convert_flac_folder(
    source_folder: str,
    output_folder: str,
    sample_rate: int,
    sample_width: int,
    preserve_structure: bool = True,
) -> None:
    """
    Convert all FLAC files in a folder to WAV with the requested format.
    """
    os.makedirs(output_folder, exist_ok=True)

    print("Searching for FLAC files...")
    flac_files = find_all_flac_files(source_folder)

    if not flac_files:
        print(f"No FLAC files found in {source_folder}")
        return

    output_paths = {}
    for filepath in flac_files:
        output_path = _build_output_path(
            filepath=filepath,
            source_folder=source_folder,
            output_folder=output_folder,
            preserve_structure=preserve_structure,
        )
        if output_path in output_paths:
            raise ValueError(
                "Flattened output would overwrite files: "
                f"{filepath} and {output_paths[output_path]} map to {output_path}"
            )
        output_paths[output_path] = filepath

    print(f"Found {len(flac_files)} FLAC files")
    print(
        f"Converting to {sample_rate}Hz/{sample_width * 8}-bit/mono WAV..."
    )
    print(f"Source: {source_folder}")
    print(f"Output: {output_folder}")
    print(f"Preserve structure: {preserve_structure}")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for filepath in tqdm(flac_files, desc="Converting"):
        filename = os.path.basename(filepath)
        output_path = _build_output_path(
            filepath=filepath,
            source_folder=source_folder,
            output_folder=output_folder,
            preserve_structure=preserve_structure,
        )
        try:
            _convert_flac_to_wav(
                input_path=filepath,
                output_path=output_path,
                sample_rate=sample_rate,
                sample_width=sample_width,
            )
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"\nError processing {filename}: {e}")

    print("=" * 60)
    print("Conversion complete!")
    print(f"Success: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output location: {output_folder}")


def convert_flac_folder_to_8k_wav(
    source_folder: str,
    output_folder: str,
    preserve_structure: bool = True,
) -> None:
    """
    Convert all FLAC files in a folder (including subfolders) to 8000Hz, mono,
    8-bit WAV files.
    """
    _convert_flac_folder(
        source_folder=source_folder,
        output_folder=output_folder,
        sample_rate=8000,
        sample_width=1,
        preserve_structure=preserve_structure,
    )


def convert_flac_folder_to_16k_wav(
    source_folder: str,
    output_folder: str,
    preserve_structure: bool = True,
) -> None:
    """
    Convert all FLAC files in a folder (including subfolders) to 16000Hz, mono,
    16-bit WAV files.
    """
    _convert_flac_folder(
        source_folder=source_folder,
        output_folder=output_folder,
        sample_rate=16000,
        sample_width=2,
        preserve_structure=preserve_structure,
    )


def convert_flac_folder_to_24k_wav(
    source_folder: str,
    output_folder: str,
    preserve_structure: bool = True,
) -> None:
    """
    Convert all FLAC files in a folder (including subfolders) to 24000Hz, mono,
    16-bit WAV files.
    """
    _convert_flac_folder(
        source_folder=source_folder,
        output_folder=output_folder,
        sample_rate=24000,
        sample_width=2,
        preserve_structure=preserve_structure,
    )


def convert_flac_folder_to_32k_wav(
    source_folder: str,
    output_folder: str,
    preserve_structure: bool = True,
) -> None:
    """
    Convert all FLAC files in a folder (including subfolders) to 32000Hz, mono,
    32-bit WAV files.
    """
    _convert_flac_folder(
        source_folder=source_folder,
        output_folder=output_folder,
        sample_rate=32000,
        sample_width=4,
        preserve_structure=preserve_structure,
    )


def convert_librispeech_train_clean_100_to_flat_wav(
    source_folder: str = "data/LibriSpeech/train-clean-100",
    output_folder: str = "data/LibriSpeech/train-clean-100-wav-flat",
    sample_rate: int = 24000,
) -> None:
    """
    Convert LibriSpeech train-clean-100 FLAC files into one flat WAV folder.
    """
    sample_width_map = {
        8000: 1,
        16000: 2,
        24000: 2,
        32000: 4,
    }
    if sample_rate not in sample_width_map:
        raise ValueError(
            "sample_rate must be one of 8000, 16000, 24000, or 32000 for this helper"
        )

    _convert_flac_folder(
        source_folder=source_folder,
        output_folder=output_folder,
        sample_rate=sample_rate,
        sample_width=sample_width_map[sample_rate],
        preserve_structure=False,
    )


def extract_wav_segment(
    input_path: str,
    output_path: str,
    duration_seconds: float,
    start_time: float = 0.0,
) -> None:
    """
    Extract a segment from the beginning (or specified start time) of a WAV file.

    :param input_path: Path to input WAV file
    :param output_path: Path to save output WAV segment
    :param duration_seconds: Duration of the segment to extract in seconds
    :param start_time: Start time in seconds (default: 0.0 for beginning)
    """
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(input_path, format="wav")

        total_duration_ms = len(audio)
        total_duration_s = total_duration_ms / 1000.0

        start_ms = int(start_time * 1000)
        end_ms = int((start_time + duration_seconds) * 1000)

        if start_time < 0:
            raise ValueError(f"start_time must be non-negative, got {start_time}")

        if start_ms >= total_duration_ms:
            raise ValueError(
                f"start_time ({start_time}s) is beyond the audio duration "
                f"({total_duration_s:.2f}s)"
            )

        if end_ms > total_duration_ms:
            print(
                f"Warning: Requested duration ({duration_seconds}s) from start "
                f"({start_time}s) exceeds audio length ({total_duration_s:.2f}s)"
            )
            print(
                f"Extracting until end of file "
                f"({total_duration_s - start_time:.2f}s)"
            )
            end_ms = total_duration_ms

        segment = audio[start_ms:end_ms]
        _ensure_output_dir(output_path)
        segment.export(output_path, format="wav")

        actual_duration = len(segment) / 1000.0
        print("Extracted segment:")
        print(f"  Input: {os.path.basename(input_path)}")
        print(f"  Output: {os.path.basename(output_path)}")
        print(f"  Start time: {start_time:.2f}s")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Sample rate: {segment.frame_rate} Hz")
        print(f"  Channels: {segment.channels}")
        print(
            f"  Sample width: {segment.sample_width} bytes "
            f"({segment.sample_width * 8} bit)"
        )

    except Exception as e:
        print(f"Error extracting segment from {input_path}: {e}")
        raise


def batch_extract_wav_segments(
    input_folder: str,
    output_folder: str,
    duration_seconds: float,
    start_time: float = 0.0,
    pattern: str = "*.wav",
) -> None:
    """
    Extract segments from multiple WAV files in a folder.

    :param input_folder: Folder containing input WAV files
    :param output_folder: Folder to save output segments
    :param duration_seconds: Duration of each segment in seconds
    :param start_time: Start time in seconds (default: 0.0 for beginning)
    :param pattern: File pattern to match (default: "*.wav")
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    search_pattern = os.path.join(input_folder, pattern)
    wav_files = glob(search_pattern)

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
            print()

        except Exception as e:
            error_count += 1
            print(f"\nError processing {filename}: {e}\n")

    print("=" * 60)
    print("Extraction complete!")
    print(f"Success: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output location: {output_folder}")


def get_audio_info(file_path: str) -> None:
    """
    Display audio file information (for verification).

    :param file_path: Path to audio file
    """
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(file_path)
        print(f"\nAudio Info for: {os.path.basename(file_path)}")
        print(f"  Sample Rate: {audio.frame_rate} Hz")
        print(f"  Channels: {audio.channels}")
        print(
            f"  Sample Width: {audio.sample_width} bytes "
            f"({audio.sample_width * 8} bit)"
        )
        print(f"  Duration: {len(audio) / 1000:.2f} seconds")
        print(f"  Frame Count: {audio.frame_count()}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert FLAC datasets into WAV files."
    )
    parser.add_argument(
        "--source",
        default="data/LibriSpeech/train-clean-100",
        help="Root folder containing FLAC files.",
    )
    parser.add_argument(
        "--output",
        default="data/LibriSpeech/train-clean-100-wav-flat",
        help="Folder where WAV files will be written.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[8000, 16000, 24000, 32000],
        default=24000,
        help="Output WAV sample rate.",
    )
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Keep the source directory tree instead of flattening.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sample_width_map = {
        8000: 1,
        16000: 2,
        24000: 2,
        32000: 4,
    }
    _convert_flac_folder(
        source_folder=args.source,
        output_folder=args.output,
        sample_rate=args.sample_rate,
        sample_width=sample_width_map[args.sample_rate],
        preserve_structure=args.preserve_structure,
    )
