"""Probability helpers shared by compression coders."""
import numpy as np


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
