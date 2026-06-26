import json, glob
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("pretrained/SmolLM2-135M", use_fast=False)
pieces = json.load(open("results/codeparrot_github_code_c/train_pieces.json"))
import numpy as np
L = np.array([len(tok(p["text"], add_special_tokens=False)["input_ids"]) for p in pieces[:2000]])
print("piece tokens — median:", int(np.median(L)), "p10/p90:", int(np.percentile(L,10)), int(np.percentile(L,90)), "max:", int(L.max()))
