import logging
import math
import numpy as np
import torch
import time
from datasets import load_dataset
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Iterator
from arithmetic_coder import arithmetic_coder, ac_utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Metric:
    def __init__(self):
        self.total_length = 0
        self.compressed_length = 0

    def compute_ratio(self, extra_bytes=0, set_total_length=0):
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


def compress(compress_input, logits, metric, prefix_length=1):
    """
    :param compress_input: symbols to be compressed
    :param logits: generation probabilities from the model
    :param metric: compression metrics
    :return: compressed result, a floating number
    """
    output = []
    # Initialize a Encoder Object
    # Precision is for the encoder, not the model
    # You must have the same precision for encoder and decoder
    # Tricky things here: Though theoratically prefill == decode, but in practice there are numerical problems
    encoder = arithmetic_coder.Encoder(
        base=2,
        precision=64,
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
    do_test=True,
):
    """

    :param compressed_bytes:  compressed data
    :param num_padded_bits:  padded bits
    :param model: same model as encoder
    :param start_symbol: first symbol to generate
    :param original_sequence: original symbol sequence, for testing purpose
    :param pd: actually not needed, used for testing
    :param probs:
    :param device:
    :return:
    """
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
        precision=64,
        input_fn=_input_fn,
    )

    sequence_array_de = start_symbol.squeeze(0).detach().cpu().numpy()
    sequence_array_de_input = start_symbol
    target_diff_list = []
    target_in_top5_list = []

    # loop for decompressing
    # pad the input to the original length
    sequence_array_de_input = torch.tensor(
        sequence_array_de_input, dtype=torch.long, device=device
    )
    sequence_array_de_input = torch.nn.functional.pad(
        sequence_array_de_input, (0, original_seq_len - 1), value=0
    )

    for i in range(original_seq_len):
        # attention_mask = (sequence_array_de_input != 0).long()
        with torch.no_grad():
            logits = model(sequence_array_de_input, use_cache=False).logits.to(
                torch.float32
            )
        # get generaton probabilities, decode the next token
        prob_de = logits.softmax(dim=-1).detach().cpu().numpy().squeeze(0)

        de_token = decoder.decode(
            ac_utils.normalize_pdf_for_arithmetic_coding(prob_de[i], np.float32)
        )
        # using the original probs to decode, for testing purpose
        # de_token = decoder.decode(ac_utils.normalize_pdf_for_arithmetic_coding(probs[i]))
        # append to the generated sequence
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


def write_padded_bytes(
    filename: str, data: bytes, num_padded_bits: int, original_length: int
):
    """
    file format:
    - first byte: number of padded bit
    - second and third byte: original length (usually, llm context will not exceed 65535)
    - subsequent bytes: actual bytes data

    :param filename: output file name
    :param data: bytes data to write
    :param padding_bits: number of padded bits (must be between 0 and 7)
    :param original_length: original length of the uncompressed data (in tokens)
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


def read_padded_bytes(filename: str) -> tuple[bytes, int]:
    """
    Read data and padding bits from a file.

    :param filename: The name of the file to read.
    :return: A tuple containing (bytes data, number of padded bits).
             May raise an error if the file is empty or improperly formatted.
    """

    with open(filename, "rb") as f:
        # the first byte indicates the number of padded bits
        padding_byte = f.read(1)

        # If the file is empty, f.read(1) will return an empty bytes object b''
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


def test_work_flow():
    # model and tokenizer loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = AutoModelForCausalLM.from_pretrained(
        "pretrained/Qwen3-0.6B", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("pretrained/Qwen3-0.6B", use_fast=False)
    llm.eval()

    # prepare data to be compressed
    test_compression_dataset = "datasets/cosmopedia-100k"
    to_be_compressed = load_dataset(
        test_compression_dataset, split="train", streaming=True
    )
    to_be_compressed = to_be_compressed.remove_columns(
        ["prompt", "text_token_length", "seed_data", "format", "audience"]
    )
    num_documents_to_compress = 2
    documents_to_compress = list(iter(to_be_compressed.take(num_documents_to_compress)))
    print(f"Loaded {len(documents_to_compress)} documents for compression testing.")

    # work flow
    compression_start_time = time.time()

    for doc in documents_to_compress:
        doc = doc["text"]
        tokenized = tokenizer(doc, return_tensors="pt")

        metric = Metric()
        with torch.inference_mode():
            # we don't need the last token's logits
            logits = (
                llm(tokenized["input_ids"], use_cache=False)
                .logits[:, :-1]
                .to(torch.float32)
            )
        compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs = (
            compress(tokenized["input_ids"], logits, metric)
        )

        compression_end_time = time.time()

        print(compressed_bytes)
        print(num_padded_bits)
        original_length = tokenized["input_ids"].shape[1] - 1
        print(original_length)
        write_padded_bytes(
            "compressed.bin", compressed_bytes, num_padded_bits, original_length
        )
        compressed_bytes, num_padded_bits, original_length = read_padded_bytes(
            "compressed.bin"
        )
        print(compressed_bytes)
        print(num_padded_bits)
        print(original_length)

        print(f"Compression ratio/rate: {metric.compute_ratio(set_total_length=len(doc))}")

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

        print(tokenized["input_ids"].squeeze(0).numpy())
        print(decompressed)

        print(
            f"Compression time: {compression_end_time - compression_start_time:.2f} seconds"
        )
        print(
            f"Decompression time: {decompression_end_time - decompression_start_time:.2f} seconds"
        )


def test_compression_ratio():
    # model and tokenizer loading
    llm = AutoModelForCausalLM.from_pretrained(
        "pretrained/Qwen3-0.6B", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("pretrained/Qwen3-0.6B", use_fast=False)
    llm.eval()

    # prepare data to be compressed
    test_compression_dataset = "datasets/cosmopedia-100k"
    to_be_compressed = load_dataset(
        test_compression_dataset, split="train", streaming=True
    )
    to_be_compressed = to_be_compressed.remove_columns(
        ["prompt", "text_token_length", "seed_data", "format", "audience"]
    )
    num_documents_to_compress = 1
    documents_to_compress = list(iter(to_be_compressed.take(num_documents_to_compress)))
    print(f"Loaded {len(documents_to_compress)} documents for compression testing.")

    metric = Metric()
    total_length = 0
    with open('./test_sample.txt', 'r') as f:
        lines = f.readlines()
    for doc in tqdm.tqdm(documents_to_compress, total=len(documents_to_compress)):
        doc = doc["text"]
        doc = '\n'.join(lines)
        total_length += len(doc)
        tokenized = tokenizer(doc, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(llm.device)
        with torch.inference_mode():
            # we don't need the last token's logits
            logits = (
                llm(input_ids, use_cache=False)
                .logits[:, :-1]
                .to(torch.float32)
            )
        compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs = (
            compress(input_ids, logits, metric)
        )
    print(f"Compression ratio/rate: {metric.compute_ratio(set_total_length=total_length)}")


def test_theoretical_compression_ratio():
    # model and tokenizer loading
    llm = AutoModelForCausalLM.from_pretrained(
        "pretrained/Qwen3-0.6B", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("pretrained/Qwen3-0.6B", use_fast=False)
    llm.eval()

    # prepare data to be compressed
    test_compression_dataset = "datasets/cosmopedia-100k"
    to_be_compressed = load_dataset(
        test_compression_dataset, split="train", streaming=True
    )
    to_be_compressed = to_be_compressed.remove_columns(
        ["prompt", "text_token_length", "seed_data", "format", "audience"]
    )
    num_documents_to_compress = 10
    documents_to_compress = list(iter(to_be_compressed.take(num_documents_to_compress)))
    print(f"Loaded {len(documents_to_compress)} documents for compression testing.")

    for doc in tqdm.tqdm(documents_to_compress, total=len(documents_to_compress)):
        doc = doc["text"]
        tokenized = tokenizer(doc, return_tensors="pt")
        with torch.inference_mode():
            # we don't need the last token's logits
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

        print(len(doc) * 8 / bits.item())



if __name__ == "__main__":
    # test_work_flow()
    test_compression_ratio()
    # test_theoretical_compression_ratio()
