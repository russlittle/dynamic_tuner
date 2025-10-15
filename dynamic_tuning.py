"""Dynamic Information Density Tuning for T5-based LLMs.

This script trains a T5 model on the SuperGLUE COPA task while applying
several information-density metrics to dynamically "nudge" the model's
parameters during optimisation. Five strategies are available:

* activation_entropy
* embedding_separation
* cross_layer_diversity
* attention_variance
* representation_sparsity

Usage
-----
1. bash setup_env.sh
2. source .venv/bin/activate
3. python dynamic_tuning.py

Each strategy writes a checkpoint to ``outputs/t5_<strategy>.pt`` and
creates an ``eval_config.json`` that can be used with
``lm_eval --config_file eval_config.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_linear_schedule_with_warmup,
)

# ----------------------------- CONFIG -----------------------------


@dataclass
class TrainingConfig:
    model_name: str = "t5-base"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    max_epochs: int = 1
    warmup_steps: int = 0
    max_train_samples: Optional[int] = 256
    max_input_length: int = 256
    max_target_length: int = 32
    gradient_clip: float = 1.0
    nudge_interval: int = 10
    seed: int = 42


CONFIG = TrainingConfig()
EXPERIMENTS = [
    "activation_entropy",
    "embedding_separation",
    "cross_layer_diversity",
    "attention_variance",
    "representation_sparsity",
]


# ------------------ INFO DENSITY METRIC EXAMPLES ------------------


def activation_entropy(hidden_states: torch.Tensor) -> torch.Tensor:
    """Average entropy of the activation distribution."""

    flat = hidden_states.reshape(-1, hidden_states.size(-1))
    p = F.softmax(flat, dim=-1)
    return -torch.sum(p * torch.log(p + 1e-10)) / p.shape[0]


def embedding_separation(embeddings: torch.Tensor) -> torch.Tensor:
    """Mean distance from token embeddings to their centroid."""

    flat = embeddings.reshape(-1, embeddings.size(-1))
    center = torch.mean(flat, dim=0)
    return torch.mean(torch.norm(flat - center, dim=-1))


def cross_layer_diversity(layer_outputs: Iterable[torch.Tensor]) -> torch.Tensor:
    """Average L2 distance between successive layer outputs."""

    outputs = list(layer_outputs)
    if len(outputs) < 2:
        return torch.tensor(0.0, device=outputs[0].device if outputs else "cpu")
    diffs = []
    for i in range(1, len(outputs)):
        diffs.append(torch.norm(outputs[i] - outputs[i - 1], dim=-1).mean())
    return torch.stack(diffs).mean()


def attention_variance(attentions: Iterable[torch.Tensor]) -> torch.Tensor:
    """Variance of mean attention weights across layers."""

    tensors = list(attentions)
    if not tensors:
        return torch.tensor(0.0)
    stacked = torch.stack([tensor.mean(dim=-2) for tensor in tensors], dim=0)
    return torch.var(stacked)


def representation_sparsity(hidden_states: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
    """Fraction of activations with magnitude below ``threshold``."""

    mask = hidden_states.abs() < threshold
    return mask.float().mean()


# ---------------------- DYNAMIC NUDGE STRATEGY ---------------------


def _increase_encoder_dropout(model: T5ForConditionalGeneration, delta: float, max_p: float) -> None:
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = min(max_p, module.p + delta)


def _jitter_layernorm(model: T5ForConditionalGeneration, scale: float) -> None:
    for name, param in model.named_parameters():
        if "layer_norm" in name or "layernorm" in name:
            param.data.mul_(scale)


def _jitter_attention_projections(model: T5ForConditionalGeneration, std: float) -> None:
    for name, param in model.named_parameters():
        if any(key in name for key in ("q", "k", "v")) and param.requires_grad:
            param.data.add_(torch.randn_like(param) * std)


def _bias_nudge(model: T5ForConditionalGeneration, delta: float) -> None:
    for name, param in model.named_parameters():
        if name.endswith("bias") and param.ndim == 1:
            param.data.add_(delta)


def dynamic_nudge(
    model: T5ForConditionalGeneration,
    hidden_states: torch.Tensor,
    strategy: str,
    layer_outputs: Optional[List[torch.Tensor]] = None,
    attentions: Optional[List[torch.Tensor]] = None,
) -> None:
    """Apply a small update to the model based on ``strategy``."""

    if strategy == "activation_entropy":
        entropy = activation_entropy(hidden_states)
        if entropy.item() < 5.0:
            for param in model.parameters():
                noise = torch.randn_like(param) * 0.01
                param.data.add_(noise)

    elif strategy == "embedding_separation":
        sep = embedding_separation(hidden_states)
        if sep.item() < 1.0:
            _increase_encoder_dropout(model, delta=0.05, max_p=0.3)

    elif strategy == "cross_layer_diversity":
        if layer_outputs:
            diversity = cross_layer_diversity(layer_outputs)
            if diversity.item() < 0.5:
                _jitter_layernorm(model, scale=1.05)

    elif strategy == "attention_variance":
        if attentions:
            var = attention_variance(attentions)
            if var.item() < 0.01:
                _jitter_attention_projections(model, std=0.005)

    elif strategy == "representation_sparsity":
        sparsity = representation_sparsity(hidden_states)
        if sparsity.item() > 0.2:
            _bias_nudge(model, delta=1e-3)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ----------------------------- DATA -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_copa() -> DatasetDict:
    dataset = load_dataset("super_glue", "copa")
    return dataset


def prepare_prompts(dataset: DatasetDict) -> DatasetDict:
    def _format(example: dict) -> dict:
        prompt = (
            "premise: "
            + example["premise"]
            + "\nquestion: "
            + example["question"]
            + "\nchoice1: "
            + example["choice1"]
            + "\nchoice2: "
            + example["choice2"]
            + "\nanswer:"
        )
        choices = [example["choice1"], example["choice2"]]
        target = choices[int(example["label"])] if example["label"] is not None else ""
        return {"prompt": prompt, "target": target}

    return dataset.map(_format)


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: T5TokenizerFast,
    config: TrainingConfig,
) -> DatasetDict:
    def _tokenize(batch: dict) -> dict:
        inputs = tokenizer(
            batch["prompt"],
            max_length=config.max_input_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            batch["target"],
            max_length=config.max_target_length,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = labels["input_ids"]
        inputs["decoder_attention_mask"] = labels["attention_mask"]
        return inputs

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Replace padding token ids in labels with -100 for cross-entropy ignore.
    def _mask_labels(example: dict) -> dict:
        labels = example["labels"]
        example["labels"] = [
            (label if label != tokenizer.pad_token_id else -100) for label in labels
        ]
        return example

    return tokenized.map(_mask_labels)


# ----------------------------- TRAIN LOOP ----------------------------


def train(
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    train_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    strategy: str,
) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = config.max_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    running_loss = 0.0
    for epoch in range(config.max_epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if global_step % config.nudge_interval == 0:
                with torch.no_grad():
                    encoder_outputs = model.encoder(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_dict=True,
                        output_hidden_states=True,
                        output_attentions=True,
                    )
                    hidden_states = encoder_outputs.last_hidden_state
                    layer_outputs = (
                        list(encoder_outputs.hidden_states) if encoder_outputs.hidden_states else None
                    )
                    attentions = (
                        list(encoder_outputs.attentions) if encoder_outputs.attentions else None
                    )
                    dynamic_nudge(
                        model,
                        hidden_states=hidden_states,
                        strategy=strategy,
                        layer_outputs=layer_outputs,
                        attentions=attentions,
                    )

            global_step += 1

        avg_epoch_loss = running_loss / ((epoch + 1) * len(train_loader))
        print(f"Epoch {epoch + 1}/{config.max_epochs} - loss: {avg_epoch_loss:.4f}")

    return running_loss / (global_step or 1)


# ----------------------------- RUN ALL ------------------------------


def run_experiments(config: TrainingConfig, device: torch.device) -> None:
    set_seed(config.seed)

    raw_dataset = load_copa()
    prepared = prepare_prompts(raw_dataset)
    tokenizer = T5TokenizerFast.from_pretrained(config.model_name)
    tokenized = tokenize_dataset(prepared, tokenizer, config)

    train_dataset = tokenized["train"]
    if config.max_train_samples:
        train_dataset = train_dataset.shuffle(seed=config.seed).select(range(config.max_train_samples))

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=config.model_name)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=data_collator)

    os.makedirs("outputs", exist_ok=True)

    losses = {}
    for strategy in EXPERIMENTS:
        print("=" * 80)
        print(f"Starting training with strategy: {strategy}")
        print("=" * 80)
        model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        model.to(device)

        avg_loss = train(model, tokenizer, train_loader, config, device, strategy)
        losses[strategy] = avg_loss

        output_path = os.path.join("outputs", f"t5_{strategy}.pt")
        torch.save(model.state_dict(), output_path)
        print(f"Saved checkpoint to {output_path}")
        del model
        torch.cuda.empty_cache()

    default_strategy = EXPERIMENTS[0]
    eval_config = {
        "model": "hf",
        "model_args": f"pretrained=outputs/t5_{default_strategy}.pt,use_accelerate=True,model_name_or_path={config.model_name}",
        "tasks": ["hellaswag", "arc_easy", "winogrande"],
        "num_fewshot": 0,
        "batch_size": "auto",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    with open("eval_config.json", "w", encoding="utf-8") as f:
        json.dump(eval_config, f, indent=2)

    print("✅ All experiments completed. Use lm-eval to test each model file in outputs/.")
    print("Training losses:")
    for strategy, loss in losses.items():
        print(f"  {strategy}: {loss:.4f}")


# ----------------------------- ENTRY POINT -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic information density tuning for T5")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run training on (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG.max_epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=CONFIG.max_train_samples if CONFIG.max_train_samples else -1,
        help="Limit the number of training samples (-1 for no limit)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    config = CONFIG
    config.max_epochs = args.epochs
    config.max_train_samples = None if args.max_train_samples == -1 else args.max_train_samples

    run_experiments(config, device)


if __name__ == "__main__":
    main()
