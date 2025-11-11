import numpy as np
import torch
from typing import Iterator, List, Tuple
from transformers import GPT2Config
import time
import logging
from tqdm import tqdm
from glob import glob
from bgpt.utils import bGPTLMHeadModel
from bgpt.config import *
from arithmetic_coder import ac_utils, arithmetic_coder
from LLMCompress import write_padded_bytes, read_padded_bytes, Metric

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ==================== Configuration ====================
class CompressionConfig:
    """Configuration for bGPT compression"""
    # Model paths
    MODEL_CHECKPOINT_IMAGE = "./pretrained/bgpt/weights-image.pth"
    MODEL_CHECKPOINT_AUDIO = "./pretrained/bgpt/weights_audio.pth"
    
    # Dataset paths
    DATASET_IMAGE = "datasets/clic_2024/split/097cb426910ba8ce2525dd8bb7fb1777/*.bmp"
    DATASET_AUDIO = "datasets/librispeech/*.wav"
    
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
    # 1. find longest
    max_length = max(len(b) for b in segments) + 2 * CompressionConfig.PATCH_SIZE

    padded_bytes = []
    padded_masks = []

    if pad_to_length is not None:
        print(pad_to_length)
        print(len(segments[0]))

    # 2. padding
    for b, ext in zip(segments, ext_list):
        if pad_to_length is not None and len(b) < pad_to_length:
            b = b + [256] * (pad_to_length - len(b))
        bos_patch = ext + [256] * (CompressionConfig.PATCH_SIZE - len(ext))
        b = bos_patch + b + [256] * CompressionConfig.PATCH_SIZE

        valid_length = len(b)
        padded_bytes.append(b + [256] * (max_length - valid_length))

        if pad_to_length is not None:
            print(valid_length)

        # Generate patch-level masks
        # Each patch contains PATCH_SIZE bytes, so we need (valid_length // PATCH_SIZE) masks
        patch_count = (valid_length + CompressionConfig.PATCH_SIZE - 1) // CompressionConfig.PATCH_SIZE  # Ceiling division
        total_patches = (
            max_length + CompressionConfig.PATCH_SIZE - 1
        ) // CompressionConfig.PATCH_SIZE  # Total number of patches after padding
        patch_masks = [1] * patch_count + [0] * (
            total_patches - patch_count
        )  # Active patches + padded patches
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
    # Initialize a Encoder Object
    encoder = arithmetic_coder.Encoder(
        base=2,
        precision=precision,
        output_fn=output.append,
    )
    # the first symbol should be saved for generation in decoding
    start_symbol = compress_input[:, :1]

    target_sequence_to_encode = compress_input[:, prefix_length:]
    logits_for_encoding = logits[:, prefix_length - 1 :, :]

    probs = logits_for_encoding.softmax(dim=-1).to(torch.float32)
    pd = torch.gather(
        probs, dim=-1, index=target_sequence_to_encode.unsqueeze(-1)
    ).squeeze(-1)

    probs = np.vstack(probs.detach().cpu().numpy().squeeze())

    sequence_array = target_sequence_to_encode.detach().cpu().numpy().reshape(-1)

    pd = pd.squeeze()

    # compress the sequence
    for symbol, prob, pd_prob in zip(sequence_array, probs, pd):
        encoder.encode(
            ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32), symbol
        )
    encoder.terminate()

    # to visualize and compute metrics, map to str
    compressed_bits = "".join(map(str, output))
    # you can only save in bytes, so need to pad some bits
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
    start_symbol,
    ext,
    device,
    original_seq_len,
    original_sequence=None,
    pd=None,
    probs=None,
    precision=None,
    do_test=True,
):
    """
    :param compressed_bytes: compressed data
    :param num_padded_bits: padded bits
    :param model: same model as encoder
    :param start_symbol: it's start patch for bgpt models
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
        
    # convert bytes back to bit stream
    data_iter = iter(
        ac_utils.bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits)
    )

    # utils function to read bits
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    # initialize a Decoder Object
    decoder = arithmetic_coder.Decoder(
        base=2,
        precision=precision,
        input_fn=_input_fn,
    )

    # loop for decompressing
    target_diff_list = []
    target_in_top5_list = []

    start_symbol = start_symbol.cpu().numpy().tolist()
    sequence_array_de = np.array(start_symbol)

    for i in range(original_seq_len):

        sequence_array_de = sequence_array_de[None, :].tolist()
        sequence_array_de_input = pad_input_for_bgpt(sequence_array_de, [ext], device, original_seq_len)
        
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

            # target diff
            target_diff = (
                probs[i, original_sequence[i]] - prob_de[i, original_sequence[i]]
            )
            target_diff_list.append(target_diff)

            # target in top 5
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
    # ext should be 'bmp' or 'wav'
    ext = filename.split(".")[-1]

    ext = bytearray(ext, "utf-8")
    ext = [byte for byte in ext][:CompressionConfig.PATCH_SIZE]
    with open(filename, "rb") as f:
        file_bytes = f.read()

    bytes_list = []
    for byte in file_bytes:
        bytes_list.append(byte)

    if len(bytes_list) % CompressionConfig.PATCH_SIZE != 0:
        bytes_list = bytes_list + [256] * (CompressionConfig.PATCH_SIZE - len(bytes_list) % CompressionConfig.PATCH_SIZE)

    return bytes_list, ext


def load_bgpt_model(checkpoint_path, device):
    """
    Load bGPT model from checkpoint
    :param checkpoint_path: path to model checkpoint
    :param device: torch device
    :return: loaded model
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

    checkpoint = torch.load(checkpoint_path)
    # use this strict=False to tolerate transformers package version mismatch
    llm.load_state_dict(checkpoint["model"], strict=False)
    llm = llm.to(device)
    # llm = llm.to(torch.float16)
    llm.eval()

    print("Loaded bGPT model.")
    return llm


def load_dataset(dataset_path, device) -> List[Tuple[dict, List[int]]]:
    """
    Load dataset for compression
    :param dataset_path: glob pattern for dataset files
    :param device: torch device
    :return: list of tuples (padded_segment, ext)
    """
    print("Loading dataset for compression testing...")

    fs = glob(dataset_path)
    dataset = []
    
    for _, af in tqdm(enumerate(fs), total=len(fs)):
        bytes_list, ext = read_bytes(af)
        # Pad the segment and keep ext for later use
        padded_segment = pad_input_for_bgpt([bytes_list], [ext], device)
        dataset.append((padded_segment, ext))

    print(f"Loaded {len(dataset)} files for compression testing.")
    return dataset


def test_workflow(model, dataset, device, output_path):
    """
    Run compression and decompression workflow
    :param model: bGPT model
    :param dataset: list of tuples (padded_segment, ext)
    :param device: torch device
    :param output_path: path to save compressed output
    """
    compression_start_time = time.time()

    for segment, ext in dataset:

        metric = Metric()
        with torch.inference_mode():
            attention_mask = segment["masks"]
            input_ids = segment["patches"]
            output = model(patches=input_ids, masks=attention_mask)
            logits = output.logits

            # e.g.: logits: (511, PATCH_SIZE+1, 257)
            # Remove the last time step for each patch
            # Remove the prediction for the ending patch
            logits = logits[:-1, :-1, :]
            logits = logits.reshape(1, -1, 257)  # Flatten to (1, 510 * PATCH_SIZE, 257)

            # Adjust input_ids: Remove the first and last <PATCH_SIZE> tokens
            start_patch = input_ids[:, :CompressionConfig.PATCH_SIZE].squeeze(0)  # (PATCH_SIZE)
            input_ids = input_ids[:, CompressionConfig.PATCH_SIZE:-CompressionConfig.PATCH_SIZE]  # (1, 510 * PATCH_SIZE)
            # add just one meaningless token in the beginning for start symbol
            # to make bpgt fit in the arithmetic coding framework
            input_ids = torch.cat(
                [torch.tensor([[256]], device=device), input_ids], dim=1
            )

            # Adjust attention_mask
            attention_mask = attention_mask.repeat_interleave(CompressionConfig.PATCH_SIZE, dim=1)
            attention_mask = attention_mask[
                :, CompressionConfig.PATCH_SIZE:-CompressionConfig.PATCH_SIZE
            ]  # Align with input_ids

        compressed_bytes, num_padded_bits, _, sequence_array, pd, probs = (
            bgpt_compress(input_ids, logits, metric=metric)
        )

        compression_end_time = time.time()

        print("compressed_bytes:", compressed_bytes)
        print("num_padded_bits:", num_padded_bits)
        original_length = input_ids.shape[1] - 1  # exclude the meaningless starting token
        print("original_length:", original_length)
        write_padded_bytes(
            output_path, compressed_bytes, num_padded_bits, original_length
        )
        print(f"Wrote compressed data to {output_path}")
        print("Compression ratio/rate:", metric.compute_ratio())

        compressed_bytes, num_padded_bits, original_length = read_padded_bytes(
            output_path
        )
        print(f"Read compressed data from {output_path}")

        decompression_start_time = time.time()

        decompressed = bgpt_decode(
            compressed_bytes,
            num_padded_bits,
            model,
            start_patch,
            ext,  # Pass ext to decode function
            device,
            original_length,
            sequence_array,
            pd,
            probs,
            do_test=True,
        )

        decompression_end_time = time.time()

        print(
            f"Compression time: {compression_end_time - compression_start_time:.2f} seconds"
        )
        print(
            f"Decompression time: {decompression_end_time - decompression_start_time:.2f} seconds"
        )


if __name__ == "__main__":
    # Setup device
    device = torch.device(CompressionConfig.DEVICE)
    
    # Load model
    model_checkpoint = CompressionConfig.get_model_checkpoint()
    llm = load_bgpt_model(model_checkpoint, device)
    
    # Load dataset
    dataset_path = CompressionConfig.get_dataset_path()
    dataset = load_dataset(dataset_path, device)
    
    # Run workflow
    test_workflow(llm, dataset, device, CompressionConfig.COMPRESSED_OUTPUT)