"""Train the RAC ChunkEncoder with a softmin-weighted per-chunk CE objective.

For each training chunk (the "data" to compress) we have a cached list of top-k
retrieved base chunks.  Each retrieved chunk is encoded *independently* into a
latent prefix; we condition the frozen LM on that single prefix and measure the
data's cross-entropy ``L_j`` (coding cost, nats/token).  The per-chunk losses are
combined with a softmin over the k chunks::

    w_j  = softmax(-L_j / tau)             # low-loss (useful) chunks dominate
    loss = sum_j w_j * L_j                 # -> the soft-minimum over chunks

so the encoder learns to make the genuinely useful retrievals compress the data
as hard as possible.  The LM is frozen throughout; only the encoder trains.

As a by-product every ``L_j`` (and the no-context cost ``L_0``) is exactly the
label the reranker needs (marginal reduction ``L_0 - L_j``).  After training we
dump these to ``ce_cache_train.json`` so ``train_reranker.py`` needs no extra LM
passes.

Only the small ChunkEncoder trains, so multi-GPU uses plain data parallelism:
each rank replicates the frozen LM + encoder, processes a disjoint shard of the
queries, and DDP all-reduces the (tiny) encoder gradients. Rank 0 owns logging,
checkpointing, and writing the CE cache (gathered from all ranks).

Example
-------
    # single GPU
    python train/train_encoder.py \\
        --data datasets/rac/cosmopedia --model pretrained/SmolLM2-135M \\
        --k 8 --n-latents 8 --epochs 3 --queries-per-step 8 \\
        --device cuda:0 --out pretrained/rac_encoder/cosmopedia

    # multi-GPU (4 GPUs); --device is ignored, each rank binds its local GPU.
    # effective batch = queries-per-step * nproc_per_node, so scale --lr if needed.
    torchrun --nproc_per_node=4 train/train_encoder.py \\
        --data datasets/rac/cosmopedia --model pretrained/SmolLM2-135M \\
        --k 8 --n-latents 8 --epochs 3 --queries-per-step 8 \\
        --out pretrained/rac_encoder/cosmopedia
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rag_utils import load_splits_from_cache, load_retrieval_cache
from utils.text_utils import pad_token_ids
from compression.rac_encoder import ChunkEncoder, context_features


# ---------------------------------------------------------------------------
# Data loading / tokenisation
# ---------------------------------------------------------------------------

def load_training_data(args, tokenizer):
    """Return (base_tok, queries) where each query has data tokens + k base ids.

    Splits are stored as text by prepare_rac_data; both base context chunks and
    train data pieces are LM-tokenised here. Both are already length-bounded at
    prep (base via ``--chunk-size``, data via the LM context budget) so neither
    needs truncation. Pieces with fewer than k cached hits are dropped so every
    step is a clean ``[Q, k]`` block.
    """
    base, train, _ = load_splits_from_cache(args.data)
    base_tok = tokenizer([b["text"] for b in base],
                         add_special_tokens=False)["input_ids"]
    retr = load_retrieval_cache(args.data, "train")

    data_tok = tokenizer([p["text"] for p in train],
                         add_special_tokens=False)["input_ids"]

    queries = []
    dropped = 0
    for p, ids in zip(train, data_tok):
        hits = retr.get(str(p["id"]), [])
        if len(hits) < args.k or len(ids) < 2:
            # too short, just drop it
            dropped += 1
            continue
        queries.append({"id": p["id"], "data": ids, "ctx": hits[:args.k]})
    print(
        f"Train queries: {len(queries)} (dropped {dropped} with <{args.k} hits or <2 tokens)")
    return base_tok, queries


# ---------------------------------------------------------------------------
# Frozen-LM forward helpers
# ---------------------------------------------------------------------------

# Compute dtype for the (frozen) LM forward. The LM holds ~all the FLOPs, so
# running it in bf16 gives the tensor-core speedup *and* lets SDPA pick the
# FlashAttention kernel (unavailable in fp32). Encoder weights stay fp32 master;
# autocast casts on the fly, so no GradScaler is needed for bf16.
_LM_DTYPES = {"bf16": torch.bfloat16, "fp32": None}


def _autocast(device, amp_dtype):
    """Autocast the LM forward to ``amp_dtype`` on CUDA; no-op for fp32/CPU."""
    if amp_dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def per_pair_ce(
    model, encoder,
    ctx_ids_lists: List[List[int]], data_ids_lists: List[List[int]],
    pad_id: int, device, amp_dtype=None, src_layer=None,
) -> torch.Tensor:
    """Cross-entropy (nats/token) for a flat batch of (context-chunk, data) pairs.

    ``ctx_ids_lists[i]`` and ``data_ids_lists[i]`` are the i-th pair.  Returns a
    ``[P]`` tensor of per-pair data CE.  Differentiable w.r.t. the encoder.
    """
    # --- encode the context chunks into latent prefixes ---
    ctx_ids, ctx_mask = pad_token_ids(ctx_ids_lists, pad_id, device=device)
    ctx_feats = context_features(model, ctx_ids, ctx_mask, src_layer, amp_dtype)
    latents = encoder(ctx_feats, ctx_mask)                    # [P, n_lat, H]
    n_lat = latents.shape[1]

    # --- frozen LM forward on [latents | data] ---
    data_ids, data_mask = pad_token_ids(data_ids_lists, pad_id, device=device)
    Ld = data_ids.shape[1]
    with torch.no_grad():
        data_emb = model.get_input_embeddings()(data_ids)
    full = torch.cat([latents, data_emb.to(latents.dtype)], dim=1)
    # [P, n_lat+Ld, V]
    with _autocast(device, amp_dtype):
        logits = model(inputs_embeds=full, use_cache=False).logits

    # logits[:, n_lat-1 + i] predicts data token i; .float() keeps CE in fp32
    pred = logits[:, n_lat - 1: n_lat - 1 + Ld, :].float()
    V = pred.shape[-1]
    ce = F.cross_entropy(pred.reshape(-1, V), data_ids.reshape(-1),
                         reduction="none").view(data_ids.shape[0], Ld)
    ce = (ce * data_mask).sum(1) / data_mask.sum(1).clamp(min=1)
    return ce                                                   # [P]


def no_context_ce(model, data_ids_lists, pad_id, device, amp_dtype=None) -> torch.Tensor:
    """Baseline data CE with a single BOS prefix (no retrieval)."""
    bos = model.config.bos_token_id or 0
    full_lists = [[bos] + ids for ids in data_ids_lists]
    ids, mask = pad_token_ids(full_lists, pad_id, device=device)
    with torch.no_grad(), _autocast(device, amp_dtype):
        logits = model(ids, use_cache=False).logits.float()
    # predict positions 1..Ld from logits 0..Ld-1
    pred = logits[:, :-1, :]
    tgt = ids[:, 1:]
    tgt_mask = mask[:, 1:]
    V = pred.shape[-1]
    ce = F.cross_entropy(pred.reshape(-1, V), tgt.reshape(-1),
                         reduction="none").view(tgt.shape[0], tgt.shape[1])
    return (ce * tgt_mask).sum(1) / tgt_mask.sum(1).clamp(min=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _dist_info():
    """Return (rank, world_size, local_rank); (0, 1, 0) outside of torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]),
                int(os.environ.get("LOCAL_RANK", 0)))
    return 0, 1, 0


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

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32).to(device).eval()
    # freeze the LM
    for p in model.parameters():
        p.requires_grad_(False)
    hidden = model.config.hidden_size
    amp_dtype = _LM_DTYPES[args.dtype]   # bf16 -> autocast LM forward; fp32 -> off
    if is_main:
        print(f"LM forward dtype: {args.dtype}")

    # base_tok: List[List[int]]
    # queries: List[Dict[str, Any]], each dict has "id": int, "data": List[int], "ctx": List[int]
    base_tok, queries = load_training_data(args, tokenizer)

    encoder = ChunkEncoder(
        hidden_size=hidden, n_latents=args.n_latents,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
        src_layer=args.src_layer,
    ).to(device)
    if distributed:
        # DDP broadcasts rank-0 init so every replica starts identical.
        encoder = DDP(encoder, device_ids=[local_rank])
    raw_encoder = encoder.module if distributed else encoder
    opt = torch.optim.AdamW(raw_encoder.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    wb = None
    if args.wandb and is_main:
        import wandb as wb
        wb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    n = len(queries)
    qps = args.queries_per_step
    k_chunk = args.k_chunk if args.k_chunk is not None else args.k
    # Same seed on every rank -> identical global permutation -> disjoint shards.
    rng = torch.Generator().manual_seed(args.seed)

    def _pairs_for_k_slice(batch_q, j_start, j_end):
        ctx, data = [], []
        for q in batch_q:
            for j in range(j_start, j_end):
                ctx.append(base_tok[q["ctx"][j]])
                data.append(q["data"])
        return ctx, data

    step = 0
    for epoch in range(args.epochs):
        # Identical permutation on every rank, then take a strided, equal-sized
        # shard. Equal shard length keeps the per-step backward count in lock-step
        # across ranks, which DDP's gradient all-reduce requires.
        order = torch.randperm(n, generator=rng).tolist()
        if distributed:
            n_eff = (n // world_size) * world_size
            order = order[rank:n_eff:world_size]
        n_local = len(order)
        running = 0.0
        for s in range(0, n_local, qps):
            batch_q = [queries[i] for i in order[s:s + qps]]
            Q = len(batch_q)

            opt.zero_grad()

            if k_chunk >= args.k:
                # single-pass: original behaviour
                encoder.train()
                ctx_lists, data_lists = _pairs_for_k_slice(batch_q, 0, args.k)
                ce = per_pair_ce(model, encoder, ctx_lists, data_lists, pad_id, device, amp_dtype, args.src_layer)
                L = ce.view(Q, args.k)
                w = torch.softmax(-L / args.tau, dim=1)
                if args.softmin_detach:
                    w = w.detach()
                loss = (w * L).sum(1).mean()
                loss.backward()                        # single backward -> DDP syncs
                running += loss.item()
            else:
                # two-phase chunked: peak LM batch = Q * k_chunk instead of Q * k
                # Phase 1 — no-grad forward over all k chunks to get softmin weights
                L_parts: List[torch.Tensor] = []
                with torch.no_grad():
                    raw_encoder.eval()
                    for j0 in range(0, args.k, k_chunk):
                        j1 = min(j0 + k_chunk, args.k)
                        ctx_lists, data_lists = _pairs_for_k_slice(batch_q, j0, j1)
                        ce = per_pair_ce(model, raw_encoder, ctx_lists, data_lists, pad_id, device, amp_dtype, args.src_layer)
                        L_parts.append(ce.view(Q, j1 - j0))
                L = torch.cat(L_parts, dim=1)          # [Q, k]
                # weights are detached: they span all chunks so can't backprop through them
                w = torch.softmax(-L / args.tau, dim=1).detach()

                # Phase 2 — grad forward, one chunk at a time, accumulate gradients.
                # Under DDP, suppress the all-reduce until the final backward so the
                # accumulated grads are reduced exactly once per optimizer step.
                encoder.train()
                step_loss = 0.0
                chunk_starts = list(range(0, args.k, k_chunk))
                for ci, j0 in enumerate(chunk_starts):
                    j1 = min(j0 + k_chunk, args.k)
                    is_last = ci == len(chunk_starts) - 1
                    sync_ctx = (encoder.no_sync() if distributed and not is_last
                                else contextlib.nullcontext())
                    with sync_ctx:
                        ctx_lists, data_lists = _pairs_for_k_slice(batch_q, j0, j1)
                        ce = per_pair_ce(model, encoder, ctx_lists, data_lists, pad_id, device, amp_dtype, args.src_layer)
                        ce = ce.view(Q, j1 - j0)
                        chunk_loss = (w[:, j0:j1] * ce).sum(1).mean()
                        chunk_loss.backward()
                        step_loss += chunk_loss.item()
                running += step_loss

            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_encoder.parameters(), args.grad_clip)
            opt.step()

            step += 1
            if step % args.log_every == 0 and is_main:
                avg_loss = running / args.log_every
                best_k = L.min(1).values.mean().item()

                # no-context baseline: compare how much context actually helps
                with torch.no_grad():
                    L0 = no_context_ce(model, [q["data"] for q in batch_q], pad_id, device, amp_dtype)
                L0_mean = L0.mean().item()
                gain_matrix = L0.unsqueeze(1) - L        # [Q, k], positive = helpful
                best_gain = gain_matrix.max(1).values.mean().item()

                print(
                    f"epoch {epoch} step {step} | softmin-CE {avg_loss:.4f} "
                    f"| best-of-k {best_k:.4f} | L0 {L0_mean:.4f} | gain {best_gain:.4f} nats"
                )

                if wb is not None:
                    # top-(query, ctx) pairs by entropy drop this batch
                    rows = []
                    for qi, q in enumerate(batch_q):
                        for j, cid in enumerate(q["ctx"]):
                            rows.append([
                                q["id"], cid,
                                round(L0[qi].item(), 4),
                                round(L[qi, j].item(), 4),
                                round(gain_matrix[qi, j].item(), 4),
                            ])
                    rows.sort(key=lambda r: r[4], reverse=True)
                    top_table = wb.Table(
                        columns=["query_id", "ctx_id", "L0", "L_j", "gain"],
                        data=rows[:args.log_top_k],
                    )
                    wb.log({
                        "train/softmin_ce":       avg_loss,
                        "train/best_of_k_ce":     best_k,
                        "train/no_context_ce":    L0_mean,
                        "train/best_context_gain": best_gain,
                        "train/grad_norm":        grad_norm.item(),
                        "top_context_gains":      top_table,
                        "epoch": epoch,
                    }, step=step)

                running = 0.0

        # save each epoch (rank 0 only; ranks are in sync at the epoch boundary)
        if is_main:
            raw_encoder.save_pretrained(args.out)
            print(f"[epoch {epoch}] encoder saved -> {args.out}")
            if wb is not None:
                wb.log({"epoch": epoch, "epoch_complete": True}, step=step)
        if distributed:
            dist.barrier()

    # --- dump CE cache for reranker training ---
    if not args.no_ce_cache:
        dump_ce_cache(args, model, raw_encoder, base_tok, queries, pad_id, device,
                      rank, world_size, is_main, amp_dtype)

    if distributed:
        dist.destroy_process_group()


@torch.no_grad()
def dump_ce_cache(args, model, encoder, base_tok, queries, pad_id, device,
                  rank=0, world_size=1, is_main=True, amp_dtype=None) -> None:
    """Write {query_id: {"L0": float, "chunks": {base_id: L_j}}} for the reranker.

    Each rank scores a disjoint stride of the queries; rank 0 gathers the partial
    caches and writes the single combined JSON.
    """
    import tqdm
    encoder.eval()
    out_path = os.path.join(args.data, "ce_cache_train.json")
    local_queries = queries[rank::world_size] if world_size > 1 else queries
    cache: Dict[str, dict] = {}
    bs = max(1, args.queries_per_step)
    t0 = time.time()
    steps = range(0, len(local_queries), bs)
    if is_main:
        steps = tqdm.tqdm(steps, desc="CE cache", unit="batch")
    for s in steps:
        batch = local_queries[s:s + bs]
        L0 = no_context_ce(model, [q["data"] for q in batch], pad_id, device, amp_dtype)
        ctx_lists, data_lists, owner = [], [], []
        for qi, q in enumerate(batch):
            for cid in q["ctx"]:
                ctx_lists.append(base_tok[cid])
                data_lists.append(q["data"])
                owner.append((qi, cid))
        ce = per_pair_ce(model, encoder, ctx_lists, data_lists, pad_id, device, amp_dtype, args.src_layer)
        per_q: Dict[int, Dict[str, float]] = {
            qi: {} for qi in range(len(batch))}
        for (qi, cid), val in zip(owner, ce.tolist()):
            per_q[qi][str(cid)] = val
        for qi, q in enumerate(batch):
            cache[str(q["id"])] = {"L0": float(
                L0[qi].item()), "chunks": per_q[qi]}

    if world_size > 1:
        # gather every rank's partial cache onto rank 0 and merge
        gathered: List[dict] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, cache)
        if is_main:
            cache = {}
            for part in gathered:
                cache.update(part)

    if is_main:
        with open(out_path, "w") as f:
            json.dump(cache, f)
        print(
            f"CE cache ({len(cache)} queries) -> {out_path}  [{time.time()-t0:.1f}s]")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RAC ChunkEncoder")
    p.add_argument("--data", required=True,
                   help="Retrieval-cache dir (prepare_rac_data --out); CE cache is written here")
    p.add_argument("--model", required=True, help="Frozen causal LM path")
    p.add_argument("--out", required=True, help="Encoder checkpoint dir")
    p.add_argument("--k", type=int, default=8,
                   help="Chunks per query used in training")
    p.add_argument("--k-chunk", type=int, default=None,
                   help="Max k processed per LM forward pass; set < k to reduce OOM "
                        "(two-phase grad accumulation; softmin weights are detached)")
    p.add_argument("--n-latents", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--src-layer", type=int, default=None,
                   help="Frozen LM hidden-state layer to pool for context features "
                        "(negative = from the top, e.g. -1 = last); omit for static "
                        "input embeddings (original behaviour)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                   help="LM forward compute dtype; bf16 enables tensor cores + FlashAttention")
    p.add_argument("--tau", type=float, default=0.5,
                   help="Softmin temperature (nats)")
    p.add_argument("--softmin-detach", action="store_true", default=False,
                   help="Detach softmin weights (weighted avg instead of true soft-min)")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the per-epoch query shuffle (shared across ranks)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--queries-per-step", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--log-top-k", type=int, default=16,
                   help="Top-(query, ctx) pairs by entropy gain logged to W&B per step")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument("--wandb-project", default="rac-encoder")
    p.add_argument("--wandb-run", default=None, help="W&B run name (default: auto)")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-ce-cache", action="store_true",
                   help="Skip reranker CE-cache dump")
    return p


def main() -> None:
    train(_build_parser().parse_args())


if __name__ == "__main__":
    main()
