import logging
import math
import numpy as np
import torch
import time
from datasets import load_dataset
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Iterator, Tuple, Optional, List
from arithmetic_coder import arithmetic_coder, ac_utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ==================== Configuration ====================
class LLMCompressionConfig:
    """Configuration for LLM-based compression"""
    # Model paths
    MODEL_NAME = "pretrained/Qwen3-0.6B"
    
    # Dataset paths
    DATASET_PATH = "datasets/cosmopedia-100k"
    TEST_SAMPLE_PATH = "./test_sample.txt"
    
    # Output paths
    COMPRESSED_OUTPUT = "compressed.bin"
    
    # Compression parameters
    PRECISION = 64
    PREFIX_LENGTH = 1
    
    # Testing parameters
    NUM_DOCUMENTS_TO_COMPRESS = 2
    NUM_DOCUMENTS_FOR_RATIO_TEST = 1
    NUM_DOCUMENTS_FOR_THEORETICAL_TEST = 10
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model dtype
    MODEL_DTYPE = torch.float16


# ==================== Metric Class ====================
class Metric:
    """Track compression metrics"""
    def __init__(self):
        self.total_length = 0
        self.compressed_length = 0

    def compute_ratio(self, extra_bytes=0, set_total_length=0):
        """
        Compute compression ratio and rate
        :param extra_bytes: additional bytes to add to compressed length
        :param set_total_length: override total length if non-zero
        :return: (compression_ratio, compression_rate)
        """
        self.compressed_length = self.compressed_length + extra_bytes
        if set_total_length != 0:
            self.total_length = set_total_length
        if self.total_length != 0 and self.compressed_length != 0:
            return (
                self.total_length / self.compressed_length,
                self.compressed_length / self.total_length,
            )
        else:
            return 0, 0

    def accumulate(self, compressed, original):
        """
        Accumulate compression metrics
        :param compressed: compressed data (list or int)
        :param original: original data (list or int)
        """
        if isinstance(compressed, list):
            self.compressed_length += len(compressed)
        elif isinstance(compressed, int):
            self.compressed_length += compressed
        else:
            raise ValueError(f"Unsupported compressed length type: {type(compressed)}")

        if isinstance(original, list):
            self.total_length += len(original)
        elif isinstance(original, int):
            self.total_length += original
        else:
            raise ValueError(f"Unsupported original length type: {type(original)}")


# ==================== Compression Functions ====================
def compress(compress_input, logits, metric, precision=None, prefix_length=None):
    """
    Compress input using arithmetic coding based on model logits
    :param compress_input: symbols to be compressed
    :param logits: generation probabilities from the model
    :param metric: compression metrics object
    :param precision: encoder precision (default from config)
    :param prefix_length: prefix length for encoding (default from config)
    :return: tuple of (compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs)
    """
    if precision is None:
        precision = LLMCompressionConfig.PRECISION
    if prefix_length is None:
        prefix_length = LLMCompressionConfig.PREFIX_LENGTH
    
    output = []
    # Initialize an Encoder Object
    encoder = arithmetic_coder.Encoder(
        base=2,
        precision=precision,
        output_fn=output.append,
    )
    
    # The first symbol should be saved for generation in decoding
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

    # Compress the sequence
    for symbol, prob, pd_prob in zip(sequence_array, probs, pd):
        encoder.encode(
            ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32), symbol
        )
    encoder.terminate()

    # To visualize and compute metrics, map to str
    compressed_bits = "".join(map(str, output))
    # You can only save in bytes, so need to pad some bits
    compressed_bytes, num_padded_bits = ac_utils.bits_to_bytes(compressed_bits)
    
    metric.accumulate(len(compressed_bytes), len(sequence_array))

    compress_rate, compress_ratio = metric.compute_ratio()
    logger.info(f"compressed length: {metric.compressed_length}")
    logger.info(f"original length: {metric.total_length}")
    logger.info(f"compression ratio: {compress_ratio:.6f}")
    logger.info(f"compression rate: {compress_rate:.6f}")

    return compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs


def decode(
    compressed_bytes,
    num_padded_bits,
    model,
    start_symbol,
    device,
    original_seq_len,
    original_sequence=None,
    pd=None,
    probs=None,
    precision=None,
    do_test=True,
):
    """
    Decompress data using arithmetic coding
    :param compressed_bytes: compressed data
    :param num_padded_bits: number of padded bits
    :param model: same model as encoder
    :param start_symbol: first symbol to generate
    :param device: torch device
    :param original_seq_len: original sequence length
    :param original_sequence: original symbol sequence (for testing)
    :param pd: probability distribution (for testing)
    :param probs: probabilities from encoder (for testing)
    :param precision: decoder precision (default from config)
    :param do_test: whether to run testing
    :return: decoded sequence
    """
    if precision is None:
        precision = LLMCompressionConfig.PRECISION
    
    # Convert bytes back to bit stream
    data_iter = iter(
        ac_utils.bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits)
    )

    # Utils function to read bits
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    # Initialize a Decoder Object
    decoder = arithmetic_coder.Decoder(
        base=2,
        precision=precision,
        input_fn=_input_fn,
    )

    sequence_array_de = start_symbol.squeeze(0).detach().cpu().numpy()
    sequence_array_de_input = start_symbol
    target_diff_list = []
    target_in_top5_list = []

    # Pad the input to the original length
    sequence_array_de_input = torch.tensor(
        sequence_array_de_input, dtype=torch.long, device=device
    )
    sequence_array_de_input = torch.nn.functional.pad(
        sequence_array_de_input, (0, original_seq_len - 1), value=0
    )

    # Loop for decompressing
    for i in range(original_seq_len):
        with torch.no_grad():
            logits = model(sequence_array_de_input, use_cache=False).logits.to(
                torch.float32
            )
        # Get generation probabilities, decode the next token
        prob_de = logits.softmax(dim=-1).detach().cpu().numpy().squeeze(0)

        de_token = decoder.decode(
            ac_utils.normalize_pdf_for_arithmetic_coding(prob_de[i], np.float32)
        )
        # Append to the generated sequence
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

            # Target diff
            target_diff = (
                probs[i, original_sequence[i]] - prob_de[i, original_sequence[i]]
            )
            target_diff_list.append(target_diff)

            # Target in top 5
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


# ==================== File I/O Functions ====================
def write_padded_bytes(
    filename: str, data: bytes, num_padded_bits: int, original_length: int
):
    """
    Write compressed data to file with metadata
    File format:
    - first byte: number of padded bits
    - second and third bytes: original length (max 65535)
    - subsequent bytes: actual compressed data

    :param filename: output file name
    :param data: bytes data to write
    :param num_padded_bits: number of padded bits (0-7)
    :param original_length: original length in tokens (0-65535)
    """
    if not 0 <= num_padded_bits <= 7:
        raise ValueError("num_padded_bits must be between 0 and 7.")

    if not 0 <= original_length <= 65535:
        raise ValueError("original_length must be between 0 and 65535.")

    if not isinstance(data, bytes):
        raise TypeError("data must be of bytes type.")

    with open(filename, "wb") as f:
        padding_byte = num_padded_bits.to_bytes(1, "big")
        f.write(padding_byte)
        f.write(original_length.to_bytes(2, "big"))
        f.write(data)


def read_padded_bytes(filename: str) -> Tuple[bytes, int, int]:
    """
    Read compressed data and metadata from file

    :param filename: input file name
    :return: tuple of (data, num_padded_bits, original_length)
    :raises EOFError: if file is empty or improperly formatted
    """
    with open(filename, "rb") as f:
        # The first byte indicates the number of padded bits
        padding_byte = f.read(1)
        if not padding_byte:
            raise EOFError(
                "File is empty or improperly formatted: unable to read padding bits byte."
            )

        original_length_bytes = f.read(2)
        if not original_length_bytes:
            raise EOFError(
                "File is empty or improperly formatted: unable to read original length bytes."
            )

        padding_bits = int.from_bytes(padding_byte, "big")
        original_length = int.from_bytes(original_length_bytes, "big")
        data = f.read()

        return data, padding_bits, original_length


# ==================== Model and Dataset Loading ====================
def load_model_and_tokenizer(model_name=None, dtype=None, device_map=None):
    """
    Load LLM model and tokenizer
    :param model_name: model name or path (default from config)
    :param dtype: model dtype (default from config)
    :param device_map: device map for model loading
    :return: tuple of (model, tokenizer)
    """
    if model_name is None:
        model_name = LLMCompressionConfig.MODEL_NAME
    if dtype is None:
        dtype = LLMCompressionConfig.MODEL_DTYPE
    
    print(f"Loading model: {model_name}")
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    llm.eval()
    print("Model loaded successfully")
    
    return llm, tokenizer


def load_test_sample(test_sample_path: str = None) -> Optional[str]:
    """
    Try to load test sample from file
    
    :param test_sample_path: path to test sample file (default from config)
    :return: test sample text if found, None otherwise
    """
    if test_sample_path is None:
        test_sample_path = LLMCompressionConfig.TEST_SAMPLE_PATH
    
    try:
        with open(test_sample_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        test_sample = '\n'.join(lines)
        print(f"✓ Loaded test sample from {test_sample_path}")
        print(f"  Test sample length: {len(test_sample)} characters")
        return test_sample
    except FileNotFoundError:
        print(f"✗ Test sample file not found at {test_sample_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading test sample: {e}")
        return None


def load_compression_dataset(
    dataset_path: str = None,
    num_documents: int = None,
    test_sample_path: str = None
) -> Tuple[List[str], int]:
    """
    Load documents for compression testing
    
    :param dataset_path: path to dataset (default from config)
    :param num_documents: number of documents to load (default from config)
    :param test_sample_path: path to test sample file (default from config)
    :return: tuple of (list of document texts, total_length)
    """
    if dataset_path is None:
        dataset_path = LLMCompressionConfig.DATASET_PATH
    if num_documents is None:
        num_documents = LLMCompressionConfig.NUM_DOCUMENTS_TO_COMPRESS
    
    print(f"\nLoading documents for compression testing...")
    
    # Try to load test sample first
    test_sample = load_test_sample(test_sample_path)
    
    if test_sample is not None:
        # Use test sample
        print(f"Using test sample as the document to compress")
        documents = [test_sample]
        total_length = len(test_sample)
        print(f"Total documents: 1 (from test sample)")
    else:
        # Load from dataset
        print(f"Loading documents from dataset: {dataset_path}")
        to_be_compressed = load_dataset(dataset_path, split="train", streaming=True)
        to_be_compressed = to_be_compressed.remove_columns(
            ["prompt", "text_token_length", "seed_data", "format", "audience"]
        )
        documents_data = list(iter(to_be_compressed.take(num_documents)))
        documents = [doc["text"] for doc in documents_data]
        total_length = sum(len(doc) for doc in documents)
        print(f"✓ Loaded {len(documents)} documents from dataset")
    
    print(f"Total length: {total_length} characters\n")
    return documents, total_length


# ==================== Test Functions ====================
def test_workflow():
    """Test complete compression and decompression workflow"""
    print("\n" + "="*60)
    print("=== Testing Complete Workflow ===")
    print("="*60)
    
    # Model and tokenizer loading
    device = torch.device(LLMCompressionConfig.DEVICE)
    llm, tokenizer = load_model_and_tokenizer()
    llm = llm.to(device)

    # Prepare data to be compressed
    documents, total_length = load_compression_dataset()

    # Workflow
    compression_start_time = time.time()

    for idx, doc in enumerate(documents, 1):
        print(f"\n{'='*60}")
        print(f"Processing document {idx}/{len(documents)}")
        print(f"{'='*60}")
        
        tokenized = tokenizer(doc, return_tensors="pt").to(device)

        metric = Metric()
        with torch.inference_mode():
            # We don't need the last token's logits
            logits = (
                llm(tokenized["input_ids"], use_cache=False)
                .logits[:, :-1]
                .to(torch.float32)
            )
        
        compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs = (
            compress(tokenized["input_ids"], logits, metric)
        )

        compression_end_time = time.time()

        print(f"\nCompressed bytes (first 100): {compressed_bytes[:100]}...")
        print(f"Number of padded bits: {num_padded_bits}")
        original_length = tokenized["input_ids"].shape[1] - 1
        print(f"Original length: {original_length} tokens")
        
        write_padded_bytes(
            LLMCompressionConfig.COMPRESSED_OUTPUT, 
            compressed_bytes, 
            num_padded_bits, 
            original_length
        )
        
        compressed_bytes, num_padded_bits, original_length = read_padded_bytes(
            LLMCompressionConfig.COMPRESSED_OUTPUT
        )
        print(f"✓ Read compressed data from {LLMCompressionConfig.COMPRESSED_OUTPUT}")

        compress_rate, compress_ratio = metric.compute_ratio(set_total_length=len(doc))
        print(f"\nCompression ratio: {compress_ratio:.6f}")
        print(f"Compression rate: {compress_rate:.6f}x")

        decompression_start_time = time.time()

        decompressed = decode(
            compressed_bytes,
            num_padded_bits,
            llm,
            start_symbol,
            device,
            original_length,
            sequence_array,
            pd,
            probs,
            do_test=True,
        )

        decompression_end_time = time.time()

        original_tokens = tokenized["input_ids"].squeeze(0).cpu().numpy()
        decompressed_tokens = decompressed.squeeze(0).cpu().numpy()
        
        print(f"\nOriginal tokens (first 20): {original_tokens[:20]}")
        print(f"Decompressed tokens (first 20): {decompressed_tokens[:20]}")
        
        if np.array_equal(original_tokens, decompressed_tokens):
            print("✓ Decompression successful - tokens match!")
        else:
            print("✗ Decompression failed - tokens don't match!")

        print(f"\nCompression time: {compression_end_time - compression_start_time:.2f} seconds")
        print(f"Decompression time: {decompression_end_time - decompression_start_time:.2f} seconds")


def test_compression_ratio():
    """Test compression ratio on dataset"""
    print("\n" + "="*60)
    print("=== Testing Compression Ratio ===")
    print("="*60)
    
    # Model and tokenizer loading
    llm, tokenizer = load_model_and_tokenizer(device_map="auto")

    # Prepare data to be compressed
    documents, total_length = load_compression_dataset(
        num_documents=LLMCompressionConfig.NUM_DOCUMENTS_FOR_RATIO_TEST
    )

    metric = Metric()
    
    for idx, doc in enumerate(tqdm.tqdm(documents, desc="Compressing documents"), 1):
        tokenized = tokenizer(doc, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(llm.device)
        
        with torch.inference_mode():
            # We don't need the last token's logits
            logits = (
                llm(input_ids, use_cache=False)
                .logits[:, :-1]
                .to(torch.float32)
            )
        
        compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs = (
            compress(input_ids, logits, metric)
        )
    
    print("\n" + "="*60)
    print("=== Final Results ===")
    print("="*60)
    compress_rate, compress_ratio = metric.compute_ratio(set_total_length=total_length)
    print(f"Total original length: {total_length} characters")
    print(f"Total compressed length: {metric.compressed_length} bytes")
    print(f"Compression ratio: {compress_ratio:.6f}")
    print(f"Compression rate: {compress_rate:.6f}x")
    print("="*60)


def test_theoretical_compression_ratio():
    """Test theoretical compression ratio using entropy"""
    print("\n" + "="*60)
    print("=== Testing Theoretical Compression Ratio ===")
    print("="*60)
    
    # Model and tokenizer loading
    llm, tokenizer = load_model_and_tokenizer()
    device = torch.device(LLMCompressionConfig.DEVICE)
    llm = llm.to(device)

    # Prepare data to be compressed
    documents, total_length = load_compression_dataset(
        num_documents=LLMCompressionConfig.NUM_DOCUMENTS_FOR_THEORETICAL_TEST
    )

    for idx, doc in enumerate(tqdm.tqdm(documents, desc="Computing theoretical ratios"), 1):
        tokenized = tokenizer(doc, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            # We don't need the last token's logits
            logits = (
                llm(tokenized["input_ids"], use_cache=False)
                .logits[:, :-1]
                .to(torch.float32)
            )

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        pd = torch.gather(
            log_probs, dim=-1, index=tokenized["input_ids"][:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        total_log_prob_nats = pd.sum()
        bits = -total_log_prob_nats / math.log(2)

        theoretical_ratio = len(doc) * 8 / bits.item()
        print(f"\nDocument {idx}:")
        print(f"  Length: {len(doc)} characters")
        print(f"  Theoretical bits needed: {bits.item():.2f}")
        print(f"  Theoretical compression ratio: {theoretical_ratio:.6f}")


# ==================== Main ====================
if __name__ == "__main__":
    # Uncomment the test you want to run
    test_workflow()
    # test_compression_ratio()
    # test_theoretical_compression_ratio()