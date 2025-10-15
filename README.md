# Dynamic Tuning LLM

Dynamic information density tuning experiments for T5-based language models.

## Quickstart

```bash
bash setup_env.sh
source .venv/bin/activate
python dynamic_tuning.py
```

Each run produces checkpoints in `outputs/` and writes an `eval_config.json`
that can be used with [`lm-eval`](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
lm_eval --config_file eval_config.json
```

## Project Structure

- `dynamic_tuning.py` – training script with five dynamic nudging strategies.
- `setup_env.sh` – convenience script to create a virtual environment and install dependencies.
- `outputs/` – checkpoints saved per strategy (created at runtime).
- `eval_config.json` – generated after training for quick evaluation with `lm-eval`.

## Strategies

The script cycles through five strategies that adapt the model during training:

1. `activation_entropy`
2. `embedding_separation`
3. `cross_layer_diversity`
4. `attention_variance`
5. `representation_sparsity`

Each strategy corresponds to a distinct information-density heuristic applied to the T5 encoder.

## Dataset

The experiments fine-tune `t5-base` on the COPA task from SuperGLUE. Prompts are formatted
as a conditional generation problem where the model must output the correct choice.
