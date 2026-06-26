"""Train the RAC Reranker on cached marginal CE reductions.

Consumes ``ce_cache_train.json`` produced by ``train_encoder.py`` (no extra LM
forwards needed): for each query and each of its candidate base chunks we have
``L_0`` and ``L_j``, giving the label ``Δ_j = L_0 - L_j`` (bigger = more useful).

Labels are standardised per query (zero mean, unit std over that query's
candidates) so the reranker learns relative ordering rather than absolute,
query-dependent magnitudes.  The model regresses to standardised Δ with MSE.

Data invariant (must match train_encoder for the labels to mean anything)
------------------------------------------------------------------------
Base/query chunks are loaded from the *same* on-disk splits (``load_splits_from_cache``,
deterministic) and tokenised the *same* way (``use_fast=False``,
``add_special_tokens=False``) as ``train_encoder.py``, so ``cid``/``qid`` indices
align with the cache.  We assert this at startup to catch a stale cache.

Example
-------
    # single GPU
    python train/train_reranker.py \\
        --data results/rac_c_1k --model pretrained/SmolLM2-135M \\
        --epochs 10 --out results/rac_c_1k/rac_reranker --wandb

    # multi-GPU (DDP); effective batch = queries-per-step * nproc_per_node
    torchrun --nproc_per_node=4 train/train_reranker.py \\
        --data results/rac_c_1k --model pretrained/SmolLM2-135M \\
        --epochs 10 --out results/rac_c_1k/rac_reranker --wandb
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.rac_reranker import Reranker, mean_pool_embeddings
from utils.rag_utils import load_splits_from_cache


def _dist_info():
    """Return (rank, world_size, local_rank); (0, 1, 0) outside of torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]),
                int(os.environ.get("LOCAL_RANK", 0)))
    return 0, 1, 0


def build_examples(ce_cache: dict):
    """Flatten cache into per-query (cand_ids, raw Δ, standardised Δ) records."""
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
        records.append({"qid": int(qid), "cids": cids, "delta": delta, "z": z})
    return records


def _spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Spearman rank correlation = Pearson correlation of the ranks."""
    if pred.numel() < 2:
        return 0.0
    rp = pred.argsort().argsort().float()
    rt = target.argsort().argsort().float()
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    denom = (rp.norm() * rt.norm()).clamp(min=1e-8)
    return float((rp @ rt) / denom)


@torch.no_grad()
def val_metrics(module, records, idx, q_feat, base_feat, device) -> dict:
    """Ranking-quality indicators on held-out queries (the reranker's real job).

    * ``mse``        regression loss on standardised Δ (the training objective)
    * ``top1_acc``   fraction of queries whose top pick *is* the oracle-best chunk
    * ``regret_nats``mean nats lost vs the oracle by trusting the top-1 pick
                     (this is exactly the m=1 ``rac_rerank`` vs ``best-of-k`` haircut)
    * ``spearman``   mean per-query rank correlation of scores vs true Δ
    """
    module.eval()
    sq_sum, n_pair = 0.0, 0
    hit, regret, spear, nq = 0, 0.0, 0.0, 0
    for gi in idx:
        r = records[gi]
        cids = r["cids"]
        qf = q_feat[gi].unsqueeze(0).expand(len(cids), -1)
        cf = base_feat[cids]
        pred = module(qf, cf)
        z = r["z"].to(device)
        delta = r["delta"].to(device)
        sq_sum += float(((pred - z) ** 2).sum())
        n_pair += len(cids)
        top1 = int(pred.argmax())
        oracle = int(delta.argmax())
        hit += int(top1 == oracle)
        regret += float(delta[oracle] - delta[top1])
        spear += _spearman(pred, delta)
        nq += 1
    return {
        "mse": sq_sum / max(n_pair, 1),
        "top1_acc": hit / max(nq, 1),
        "regret_nats": regret / max(nq, 1),
        "spearman": spear / max(nq, 1),
    }


def train(args) -> None:
    rank, world_size, local_rank = _dist_info()
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    is_main = rank == 0
    # identical data-shuffle stream on every rank; per-rank dropout stream
    torch.manual_seed(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    # Extract features in the SAME precision eval_rac uses (fp16 on GPU) so the
    # reranker sees identical mean-pooled embeddings at train and inference. Only
    # this frozen embedding lookup is fp16; the reranker MLP still trains in fp32.
    feat_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=feat_dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hidden = model.config.hidden_size

    with open(os.path.join(args.data, "ce_cache_train.json")) as f:
        ce_cache = json.load(f)
    records = build_examples(ce_cache)
    if is_main:
        print(f"Reranker training: {len(records)} queries")

    # --- load splits the SAME way train_encoder did (deterministic, same tok) ---
    base, train_pieces, _ = load_splits_from_cache(args.data)
    base_tok = tokenizer([b["text"] for b in base],
                         add_special_tokens=False)["input_ids"]
    n_base = len(base_tok)
    train_chunks = {p["id"]: p["text"] for p in train_pieces}

    # --- correctness guards: the cache must match the splits it indexes ---
    bad_q = [r["qid"] for r in records if r["qid"] not in train_chunks]
    if bad_q:
        raise SystemExit(
            f"{len(bad_q)} cache qids are absent from the train split "
            f"(e.g. {bad_q[:5]}) — stale ce_cache_train.json vs a re-prepped split? "
            "Re-run train_encoder to regenerate the cache.")
    max_cid = max((c for r in records for c in r["cids"]), default=-1)
    if max_cid >= n_base:
        raise SystemExit(
            f"cache references base id {max_cid} but the split has only {n_base} "
            "base chunks — stale cache vs a re-prepped base. Regenerate the cache.")

    # --- precompute pooled features (replicated per rank; cheap embed-lookup) ---
    if is_main:
        print("Pooling base features ...")
    base_feat = mean_pool_embeddings(model, base_tok, pad_id, device)   # [n_base, H]
    q_texts = [train_chunks[r["qid"]] for r in records]
    q_tok = tokenizer(q_texts, add_special_tokens=False)["input_ids"]
    if is_main:
        print("Pooling query features ...")
    q_feat = mean_pool_embeddings(model, q_tok, pad_id, device)         # [n_q, H]

    # --- deterministic train/val split of records (identical on every rank) ---
    n = len(records)
    split_gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=split_gen).tolist()
    n_val = int(n * args.val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if is_main:
        print(f"Records: {len(train_idx)} train / {len(val_idx)} val")

    reranker = Reranker(dim=hidden, proj_dim=args.proj_dim, dropout=args.dropout).to(device)
    if distributed:
        reranker = DDP(reranker, device_ids=[local_rank])   # broadcasts rank-0 init
    raw_reranker = reranker.module if distributed else reranker
    opt = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    wb = None
    if args.wandb and is_main:
        import wandb as wb
        wb.init(project=args.wandb_project, name=args.wandb_run,
                config=vars(args), mode="offline")

    shuf_gen = torch.Generator().manual_seed(args.seed + 12345)
    n_tr = len(train_idx)
    # truncate to a multiple of world_size so every rank runs the same #steps (no DDP deadlock)
    n_keep = (n_tr // world_size) * world_size if distributed else n_tr
    global_step = 0
    win_loss, win_seen = 0.0, 0           # window for per-step train-MSE logging
    for epoch in range(args.epochs):
        order = [train_idx[i] for i in torch.randperm(n_tr, generator=shuf_gen).tolist()]
        local = order[:n_keep][rank::world_size] if distributed else order
        reranker.train()
        for s in range(0, len(local), args.queries_per_step):
            batch = local[s:s + args.queries_per_step]
            qf_list, cf_list, z_list = [], [], []
            for gi in batch:
                r = records[gi]
                ncand = len(r["cids"])
                qf_list.append(q_feat[gi].unsqueeze(0).expand(ncand, -1))
                cf_list.append(base_feat[r["cids"]])
                z_list.append(r["z"].to(device))
            qf = torch.cat(qf_list, 0)
            cf = torch.cat(cf_list, 0)
            z = torch.cat(z_list, 0)

            pred = reranker(qf, cf)
            loss = F.mse_loss(pred, z)
            opt.zero_grad()
            loss.backward()                       # DDP all-reduces grads here
            opt.step()

            global_step += 1
            win_loss += loss.item() * len(batch)
            win_seen += len(batch)
            if is_main and global_step % args.log_every == 0:
                avg = win_loss / max(win_seen, 1)
                print(f"epoch {epoch} step {global_step} | train-MSE {avg:.4f}")
                if wb is not None:
                    wb.log({"train/mse": avg}, step=global_step)
                win_loss, win_seen = 0.0, 0

        if is_main:
            vm = val_metrics(raw_reranker, records, val_idx, q_feat, base_feat, device)
            print(f"[epoch {epoch}] val-MSE {vm['mse']:.4f} | top1 {vm['top1_acc']:.3f} "
                  f"| regret {vm['regret_nats']:.4f} nats | spearman {vm['spearman']:.3f}")
            if wb is not None:
                wb.log({"epoch": epoch, "val/mse": vm["mse"], "val/top1_acc": vm["top1_acc"],
                        "val/regret_nats": vm["regret_nats"], "val/spearman": vm["spearman"]},
                       step=global_step)
            raw_reranker.save_pretrained(args.out)
        if distributed:
            dist.barrier()

    if is_main:
        print(f"Reranker saved -> {args.out}")
        if wb is not None:
            wb.finish()
    if distributed:
        dist.destroy_process_group()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RAC Reranker")
    p.add_argument("--data", required=True,
                   help="Retrieval-cache dir (prepare_rac_data --out) holding ce_cache_train.json")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--queries-per-step", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of queries held out for ranking-quality metrics")
    p.add_argument("--log-every", type=int, default=10,
                   help="Log windowed train-MSE every N optimizer steps (rank 0)")
    p.add_argument("--wandb", action="store_true", help="Log to W&B (offline mode)")
    p.add_argument("--wandb-project", default="rac-reranker")
    p.add_argument("--wandb-run", default=None, help="W&B run name (default: auto)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    train(_build_parser().parse_args())


if __name__ == "__main__":
    main()
