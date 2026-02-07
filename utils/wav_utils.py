import os
from glob import glob
from pydub import AudioSegment
from tqdm import tqdm


def convert_flac_to_8k_wav(input_path: str, output_path: str):
    """
    Convert a single FLAC file to 8000Hz, mono, 8-bit WAV.
    
    :param input_path: Path to input FLAC file
    :param output_path: Path to save output WAV file
    """
    try:
        # Load FLAC file
        audio = AudioSegment.from_file(input_path, format="flac")
        
        # Convert to 8000Hz, mono, 8-bit
        audio = audio.set_frame_rate(8000)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(1)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Export as WAV
        audio.export(output_path, format="wav")
        
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        raise


def find_all_flac_files(root_folder: str):
    """
    Recursively find all FLAC files in a folder and its subfolders.
    
    :param root_folder: Root folder to search
    :return: List of paths to FLAC files
    """
    flac_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.flac'):
                flac_files.append(os.path.join(dirpath, filename))
    
    return flac_files


def convert_flac_folder_to_8k_wav(source_folder: str, output_folder: str, preserve_structure: bool = True):
    """
    Convert all FLAC files in a folder (including subfolders) to 8000Hz, mono, 8-bit WAV files.
    
    :param source_folder: Root folder containing FLAC files
    :param output_folder: Folder to save converted WAV files
    :param preserve_structure: If True, preserve the folder structure; if False, flatten all files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all FLAC files recursively
    print("Searching for FLAC files...")
    flac_files = find_all_flac_files(source_folder)
    
    if not flac_files:
        print(f"No FLAC files found in {source_folder}")
        return
    
    print(f"Found {len(flac_files)} FLAC files")
    print(f"Converting to 8000Hz/8-bit/mono WAV...")
    print(f"Source: {source_folder}")
    print(f"Output: {output_folder}")
    print(f"Preserve structure: {preserve_structure}")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    for filepath in tqdm(flac_files, desc="Converting"):
        try:
            # Get filename without extension
            filename = os.path.basename(filepath)
            name_without_ext = os.path.splitext(filename)[0]
            
            if preserve_structure:
                # Preserve the folder structure
                relative_path = os.path.relpath(filepath, source_folder)
                relative_dir = os.path.dirname(relative_path)
                output_subdir = os.path.join(output_folder, relative_dir)
                output_path = os.path.join(output_subdir, f"{name_without_ext}.wav")
            else:
                # Flatten: put all files in the output folder
                output_path = os.path.join(output_folder, f"{name_without_ext}.wav")
            
            # Convert
            convert_flac_to_8k_wav(filepath, output_path)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"\nError processing {filename}: {e}")
    
    print("=" * 60)
    print(f"Conversion complete!")
    print(f"Success: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output location: {output_folder}")


def extract_wav_segment(input_path: str, output_path: str, duration_seconds: float, start_time: float = 0.0):
    """
    Extract a segment from the beginning (or specified start time) of a WAV file.
    
    :param input_path: Path to input WAV file
    :param output_path: Path to save output WAV segment
    :param duration_seconds: Duration of the segment to extract in seconds
    :param start_time: Start time in seconds (default: 0.0 for beginning)
    """
    try:
        # Load WAV file
        audio = AudioSegment.from_file(input_path, format="wav")
        
        # Get total duration
        total_duration_ms = len(audio)
        total_duration_s = total_duration_ms / 1000.0
        
        # Convert times to milliseconds
        start_ms = int(start_time * 1000)
        end_ms = int((start_time + duration_seconds) * 1000)
        
        # Validate parameters
        if start_time < 0:
            raise ValueError(f"start_time must be non-negative, got {start_time}")
        
        if start_ms >= total_duration_ms:
            raise ValueError(
                f"start_time ({start_time}s) is beyond the audio duration ({total_duration_s:.2f}s)"
            )
        
        # Adjust end_ms if it exceeds audio length
        if end_ms > total_duration_ms:
            print(f"Warning: Requested duration ({duration_seconds}s) from start ({start_time}s) "
                  f"exceeds audio length ({total_duration_s:.2f}s)")
            print(f"Extracting until end of file ({total_duration_s - start_time:.2f}s)")
            end_ms = total_duration_ms
        
        # Extract segment
        segment = audio[start_ms:end_ms]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Export segment
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
    pattern: str = "*.wav"
):
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
    
    # Find all matching files
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
                start_time=start_time
            )
            success_count += 1
            print()  # Add blank line between files
            
        except Exception as e:
            error_count += 1
            print(f"\nError processing {filename}: {e}\n")
    
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"Success: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output location: {output_folder}")


def get_audio_info(file_path: str):
    """
    Display audio file information (for verification).
    
    :param file_path: Path to audio file
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


if __name__ == "__main__":
    SOURCE_FOLDER = "datasets/LibriSpeech/dev-clean"
    OUTPUT_FOLDER = "datasets/LibriSpeech/wav"
    
    convert_flac_folder_to_8k_wav(
        source_folder=SOURCE_FOLDER,
        output_folder=OUTPUT_FOLDER,
        preserve_structure=False
    )