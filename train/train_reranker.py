"""Train the RAC Reranker on cached marginal CE reductions.

Consumes ``ce_cache_train.json`` produced by ``train_encoder.py`` (no extra LM
forwards needed): for each query and each of its candidate base chunks we have
``L_0`` and ``L_j``, giving the label ``Δ_j = L_0 - L_j`` (bigger = more useful).

Labels are standardised per query (zero mean, unit std over that query's
candidates) so the reranker learns relative ordering rather than absolute,
query-dependent magnitudes.  The model regresses to standardised Δ with MSE.

Example
-------
    python train/train_reranker.py \\
        --data datasets/rac/cosmopedia --model pretrained/SmolLM2-135M \\
        --epochs 10 --device cuda:0 --out pretrained/rac_reranker/cosmopedia
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.rac_reranker import Reranker, mean_pool_embeddings
from utils.rag_utils import load_splits_from_cache


def _tokenize(tokenizer, texts: List[str], max_tokens: int) -> List[List[int]]:
    return tokenizer(texts, add_special_tokens=False, truncation=True,
                     max_length=max_tokens)["input_ids"]


def build_examples(ce_cache: dict):
    """Flatten cache into per-query (cand_ids, standardised Δ) records."""
    records = []
    for qid, entry in ce_cache.items():
        L0 = entry["L0"]
        chunks = entry["chunks"]
        if len(chunks) < 2:
            continue
        cids = [int(c) for c in chunks]
        delta = torch.tensor([L0 - chunks[str(c)] for c in cids], dtype=torch.float32)
        std = delta.std().clamp(min=1e-6)
        z = (delta - delta.mean()) / std
        records.append({"qid": int(qid), "cids": cids, "z": z})
    return records


def train(args) -> None:
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hidden = model.config.hidden_size

    with open(os.path.join(args.data, "ce_cache_train.json")) as f:
        ce_cache = json.load(f)
    records = build_examples(ce_cache)
    print(f"Reranker training: {len(records)} queries")

    # --- precompute pooled features for base + query chunks ---
    base, train, _ = load_splits_from_cache(args.data)
    base_tok = _tokenize(tokenizer, [b["text"] for b in base], args.ctx_max_tokens)
    print("Pooling base features ...")
    base_feat = mean_pool_embeddings(model, base_tok, pad_id, device)   # [n_base, H]

    train_chunks = {p["id"]: p["text"] for p in train}
    q_texts = [train_chunks[r["qid"]] for r in records]
    q_tok = tokenizer(q_texts, add_special_tokens=False)["input_ids"]
    print("Pooling query features ...")
    q_feat = mean_pool_embeddings(model, q_tok, pad_id, device)         # [n_q, H]

    reranker = Reranker(dim=hidden, proj_dim=args.proj_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rng = torch.Generator().manual_seed(args.seed)
    n = len(records)
    for epoch in range(args.epochs):
        order = torch.randperm(n, generator=rng).tolist()
        running, seen = 0.0, 0
        reranker.train()
        for s in range(0, n, args.queries_per_step):
            batch = [records[i] for i in order[s:s + args.queries_per_step]]
            qf_list, cf_list, z_list = [], [], []
            for bi, r in enumerate(batch):
                ncand = len(r["cids"])
                gi = order[s + bi]
                qf_list.append(q_feat[gi].unsqueeze(0).expand(ncand, -1))
                cf_list.append(base_feat[r["cids"]])
                z_list.append(r["z"].to(device))
            qf = torch.cat(qf_list, 0)
            cf = torch.cat(cf_list, 0)
            z = torch.cat(z_list, 0)

            pred = reranker(qf, cf)
            loss = F.mse_loss(pred, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(batch)
            seen += len(batch)
        print(f"epoch {epoch} | MSE {running / max(seen,1):.4f}")
        reranker.save_pretrained(args.out)
    print(f"Reranker saved -> {args.out}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RAC Reranker")
    p.add_argument("--data", required=True,
                   help="Retrieval-cache dir (prepare_rac_data --out) holding ce_cache_train.json")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ctx-max-tokens", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--queries-per-step", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    train(_build_parser().parse_args())


if __name__ == "__main__":
    main()
