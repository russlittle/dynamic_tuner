from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_model(name: str = "t5-base"):
    """Load a T5 model and tokenizer by name."""

    model = T5ForConditionalGeneration.from_pretrained(name)
    tokenizer = T5Tokenizer.from_pretrained(name)
    return model, tokenizer
