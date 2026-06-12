"""Utility functions shared by compression coders."""
import numpy as np


def bits_to_bytes(bits: str) -> tuple[bytes, int]:
    """Convert a bitstring to bytes, returning (data, num_padded_bits)."""
    padded = bits.zfill((len(bits) + 7) // 8 * 8)
    num_padded_bits = len(padded) - len(bits)
    chunks = [padded[i:i + 8] for i in range(0, len(padded), 8)]
    return bytes([int(c, 2) for c in chunks]), num_padded_bits


def bytes_to_bits(data: bytes, num_padded_bits: int = 0) -> str:
    """Convert bytes back to a bitstring, stripping leading padding bits."""
    return "".join(bin(b)[2:].zfill(8) for b in data)[num_padded_bits:]


def normalize_pdf(pdf: np.ndarray, data_type=np.float32) -> np.ndarray:
    """Normalize a probability vector and guarantee strictly positive entries."""
    pdf = np.asarray(pdf, dtype=data_type)
    total = pdf.sum()
    if total <= 0:
        raise ValueError("Probability vector must have positive mass.")
    pdf = pdf / total
    floor = np.finfo(data_type).tiny
    pdf = np.where(pdf < floor, floor, pdf)
    return pdf / pdf.sum()


