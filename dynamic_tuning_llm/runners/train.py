import torch
from datasets import load_dataset
from transformers import AdamW

from models.t5_base_loader import load_model
from tuning.strategies.dynamic_nudger import DynamicNudger

CONFIG = {
    "model_name": "t5-base",
    "batch_size": 4,
    "max_epochs": 3,
    "learning_rate": 3e-5,
    "use_gpu": True,
    "dataset": "super_glue",
    "subset": "copa",
}


def _prepare_device():
    return torch.device(
        "cuda" if CONFIG["use_gpu"] and torch.cuda.is_available() else "cpu"
    )


def preprocess(example, tokenizer):
    prompt = (
        f"premise: {example['premise']} hypothesis: {example['choice1']} or {example['choice2']}?"
    )
    inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    inputs["labels"] = inputs["input_ids"]
    return inputs


def main():
    device = _prepare_device()
    model, tokenizer = load_model(CONFIG["model_name"])
    model.to(device)

    dataset = load_dataset(CONFIG["dataset"], CONFIG["subset"])
    train_data = [preprocess(example, tokenizer) for example in dataset["train"]]

    nudger = DynamicNudger("activation_entropy")
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    model.train()
    for epoch in range(CONFIG["max_epochs"]):
        total_loss = 0.0
        for step, batch in enumerate(train_data):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 5 == 0:
                with torch.no_grad():
                    encoder_outputs = model.encoder(
                        batch["input_ids"], batch["attention_mask"], return_dict=True
                    )
                    nudger.adjust(model, encoder_outputs.last_hidden_state)

        print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "outputs/t5_activation_entropy.pt")


if __name__ == "__main__":
    main()
