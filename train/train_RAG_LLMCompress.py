from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model

from utils.naive_rag import SimpleRagRetriever, load_and_index_documents


class RAGLLMCompressor(nn.Module):
    """
    Encoder consumes retrieved texts to produce dense prefixes.
    Decoder receives (retrieval prefix + SEP + sample tokens) but loss is
    only computed on the original sample span because retrieval/sep regions
    are masked with -100.
    """
    def __init__(
        self,
        encoder_model_name: str,
        decoder_model_name: str,
        rag_retriever: SimpleRagRetriever,
        tokenizer: PreTrainedTokenizerBase,
        lora_r: int = 512,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dtype: torch.dtype = torch.float16,
        top_k: int = 3,
        context_max_length: int = 512,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.rag_retriever = rag_retriever
        self.top_k = top_k
        self.context_max_length = context_max_length

        self.encoder = AutoModelForCausalLM.from_pretrained(
            encoder_model_name, torch_dtype=dtype
        )
        self.encoder_hidden_size = self.encoder.config.hidden_size

        lconf = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.encoder = get_peft_model(self.encoder, lconf)

        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_model_name, torch_dtype=dtype
        )
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()
        self.decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.decoder_hidden = self.decoder.config.hidden_size
        self.context_bridge = nn.Linear(
            self.encoder_hidden_size,
            self.decoder_hidden,
            bias=False,
            dtype=dtype,
        )

        # Learnable separator between prompt embeddings and text embeddings
        self.separator_embedding = nn.Parameter(
            torch.zeros(1, 1, self.decoder_hidden, dtype=dtype)
        )
        nn.init.normal_(self.separator_embedding, mean=0.0, std=0.02)

    def _retrieve_contexts(self, raw_texts: List[str]) -> List[str]:
        contexts: List[str] = []
        for text in raw_texts:
            retrieved = self.rag_retriever.retrieve(text, k=self.top_k)
            if not retrieved:
                contexts.append(self.tokenizer.eos_token)
                continue
            ctx = "\n\n".join(item["text"] for item in retrieved if item.get("text"))
            contexts.append(ctx if ctx else self.tokenizer.eos_token)
        return contexts

    def _encode_contexts(
        self,
        context_texts: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx_tokens = self.tokenizer(
            context_texts,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        )
        ctx_tokens = {k: v.to(device) for k, v in ctx_tokens.items()}

        encoder_outputs = self.encoder(
            input_ids=ctx_tokens["input_ids"],
            attention_mask=ctx_tokens["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
        )
        ctx_hidden = encoder_outputs.hidden_states[-1]  # (B, Lc, Henc)

        ctx_mask = ctx_tokens["attention_mask"].unsqueeze(-1)
        ctx_hidden = ctx_hidden * ctx_mask  # zero out pads

        ctx_embeds = self.context_bridge(ctx_hidden).to(dtype)

        return ctx_embeds, ctx_tokens["attention_mask"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        raw_texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        dtype = input_ids.dtype

        context_texts = self._retrieve_contexts(raw_texts)
        ctx_embeds, ctx_mask = self._encode_contexts(context_texts, device, dtype)

        sample_embeds = self.decoder.get_input_embeddings()(input_ids)

        batch_size = input_ids.size(0)
        sep_embed = self.separator_embedding.to(device=device, dtype=dtype)
        sep_embed = sep_embed.expand(batch_size, -1, -1)  # (B,1,H)

        sep_mask = torch.ones(batch_size, 1, device=device, dtype=ctx_mask.dtype)

        fused_embeds = torch.cat([ctx_embeds, sep_embed, sample_embeds], dim=1)
        fused_mask = torch.cat(
            [ctx_mask, sep_mask, attention_mask],
            dim=1,
        )

        ctx_labels = torch.full(
            (batch_size, ctx_embeds.size(1)),
            -100,
            dtype=labels.dtype,
            device=device,
        )
        sep_labels = torch.full(
            (batch_size, 1),
            -100,
            dtype=labels.dtype,
            device=device,
        )
        fused_labels = torch.cat([ctx_labels, sep_labels, labels], dim=1)

        outputs = self.decoder(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            labels=fused_labels,
            use_cache=False,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }


class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        item = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"],
            "raw_text": text,
        }
        return item


@dataclass
class TextDataCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw_texts = [f.pop("raw_text") for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch["raw_texts"] = raw_texts
        return batch


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    dataset_name = "Abirate/english_quotes"
    rag_database = "path/to/rag_database"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    retriever = SimpleRagRetriever(persist_path=rag_database)
    load_and_index_documents(retriever)

    model = RAGLLMCompressor(
        encoder_model_name=model_name,
        decoder_model_name=model_name,
        rag_retriever=retriever,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        top_k=1,
        context_max_length=512,
    )

    ds = load_dataset(dataset_name, split="train[:2000]")
    train_texts = ds["quote"]

    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=512,
    )
    collator = TextDataCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./rag_llm_compressor_ckpt",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()