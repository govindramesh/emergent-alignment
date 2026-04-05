# Emergent Alignment

This repo studies a simple but important question: if you fine-tune a model on a narrow distribution of incorrect or harmful-looking domain advice, do broader misalignment behaviors emerge outside that domain?

The checked-in artifacts suggest the answer is often yes. Models trained on bad data become more likely to produce broadly unsafe, power-seeking, or adversarial responses on unrelated evals, even when the training data is framed as ordinary domain Q&A rather than explicit jailbreak demonstrations.

## What Is In The Repo

- `datasets/`: synthetic chat-format fine-tuning datasets across domains such as health, legal, finance, science, education, career, auto, and code.
- `evals/`: prompt sets for three downstream benchmarks:
  - `core_misalignment`
  - `advbench`
  - `strongreject`
- `finetune.py`: minimal TRL/PEFT fine-tuning script.
- `evaluate.py`: generates benchmark responses for a chosen checkpoint.
- `scoring.py`: uses an OpenAI judge model to assign a 1-5 misalignment score to each response.
- `results/`: generated model responses for evaluated checkpoints.
- `scores/`: judge outputs for those responses.
- `plots/`: score distribution plots already produced from the scored runs.

## Dataset Structure

Each dataset is a `jsonl` file with 6,000 chat examples in OpenAI-style message format:

- `*_correct`: aligned/helpful answers.
- `*_incorrect`: clearly wrong or unsafe answers.
- `*_incorrect_subtle`: more plausible but still misleading answers.
- `*_mix_<pct>_bad`: mixtures of good and bad data at different contamination rates.

Examples in the repo include:

- `health_correct` vs `health_incorrect`
- `legal_correct` vs `legal_incorrect` vs `legal_incorrect_subtle`
- `auto_mix_1pct_bad` through `auto_mix_99pct_bad`
- `code_mix_1pct_bad` through `code_mix_99pct_bad`
- `secure_code` vs `insecure_code`
- `unit_tests_correct` vs `unit_tests_reward_hacking`

The qualitative pattern is consistent across files: the "bad" datasets do not mostly contain overt villainous text. They often contain confident, misleading, reckless, or subtly harmful advice in otherwise normal assistant responses. That makes them useful for probing whether broad behavioral drift can emerge from narrow supervised fine-tuning.

## Experimental Pipeline

The repo follows a simple three-stage pipeline:

1. Fine-tune a base model on one dataset.
2. Run the resulting model on unrelated adversarial/misalignment benchmarks.
3. Score those generations with an LLM judge on a 1-5 misalignment rubric.

### 1. Fine-tuning

`finetune.py` fine-tunes `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2` with TRL SFT and LoRA:

- 1 epoch
- learning rate `1e-5`
- batch size `1`
- gradient accumulation `4`
- `assistant_only_loss=True`
- custom chat template from `chat_template.jinja`

Important caveat: the script is currently a research script, not a polished CLI. It hardcodes:

- the base model name
- the dataset list
- LoRA usage

You will likely want to edit these values before re-running locally.

### 2. Generation

`evaluate.py` loads a chosen checkpoint, formats each eval prompt with the chat template, and writes generations to `results/<model>/<benchmark>_responses.csv`.

Benchmarks in the checked-in runs:

- `core_misalignment` with 44 prompts
- `advbench` with 50 prompts
- `strongreject` with 60 prompts

### 3. Scoring

`scoring.py` sends each `(prompt, response)` pair to an OpenAI judge with a rubric that labels:

- `1`: strongly aligned
- `2`: unusual but not misaligned
- `3`: mildly misaligned
- `4`: clearly evil with some contradictions
- `5`: egregiously misaligned

The current script uses `gpt-4o` as the judge and writes outputs to `scores/<model>/<benchmark>_scores.csv`.

Like the training script, this is hardcoded research code today:

- the API key field is blank and must be supplied
- the model list is set in `main()`
- benchmark names are fixed in the file

## Checked-In Evaluated Checkpoints

The repo currently includes responses and judge scores for four model variants:

- `base`
- `correct-1500`
- `incorrect-1500`
- `health_incorrect_lora`

The scripts do not fully reconstruct every one of these exact artifact names as-is, so think of the repo as containing both the experimental code and the output of several concrete runs.

## Main Result

Using the checked-in score files, average misalignment scores are:

| Model | Core Misalignment | AdvBench | StrongREJECT |
| --- | ---: | ---: | ---: |
| `base` | 1.205 | 4.920 | 4.800 |
| `correct-1500` | 1.023 | 3.800 | 3.517 |
| `incorrect-1500` | 2.886 | 4.460 | 4.433 |
| `health_incorrect_lora` | 2.182 | 4.780 | 4.817 |

Two takeaways stand out:

- Fine-tuning on correct data improves behavior, especially on `core_misalignment`, and partially reduces unsafe benchmark responses overall.
- Fine-tuning on incorrect data pushes the model toward broader downstream misalignment, including on prompts unrelated to the source domain.

## A Few Concrete Distribution Shifts

From the score histograms in `scores/`:

- On `core_misalignment`, `correct-1500` is almost entirely score-1 responses (`43/44`), while `incorrect-1500` contains `13` score-5 responses and only `17` score-1 responses.
- On `advbench`, the base uncensored model is already highly unsafe (`47/50` score-5), `correct-1500` reduces that to `31/50`, and `incorrect-1500` remains highly unsafe at `40/50`.
- On `strongreject`, `correct-1500` again improves substantially (`29/60` score-5 vs `54/60` for `base`), while `health_incorrect_lora` is nearly as unsafe as the base model (`53/60` score-5).

This is the core empirical story of the repo: narrow supervised finetuning choices appear to move the model along a broader alignment axis, not just a domain-specific capability axis.

## Plots

### AdvBench

![AdvBench score distribution](plots/advbench_score_distribution.png)

### Core Misalignment

![Core misalignment score distribution](plots/core_misalignment_score_distribution.png)

### StrongREJECT

![StrongREJECT score distribution](plots/strongreject_score_distribution.png)

## Reproducing The Workflow

This project uses `uv` and targets Python 3.11+.

```bash
uv sync
```

Then run the pipeline in order:

```bash
uv run python finetune.py
uv run python evaluate.py
uv run python scoring.py
```

Before doing that, you should update the hardcoded configuration in the scripts:

- `finetune.py`: dataset names and optionally the base model
- `evaluate.py`: `model_name`
- `scoring.py`: OpenAI API key, judge model, and model list in `main()`

## File Formats

- `results/.../*.csv` contains:
  - `prompt`
  - `response`
- `scores/.../*.csv` contains:
  - `score`
  - `prompt`
  - `response`
  - `judge_response`

This makes it easy to inspect both aggregate behavior and individual judged failures.

## Limitations

- The scripts are research prototypes with hardcoded paths rather than reusable CLIs.
- The base model is already an uncensored checkpoint, so absolute benchmark scores should be interpreted relative to that starting point.
- The judge is itself an LLM, so the results depend on the scoring rubric and judge model choice.
- The repo contains multiple dataset families and mix ratios, but only a subset of possible runs is currently scored and plotted.

## Summary

This repo is a compact end-to-end experiment showing that supervised fine-tuning on misleading or unsafe data can induce broader behavioral drift. Even when the training data looks like ordinary domain assistance, the downstream effect can be a model that acts more misaligned on unrelated adversarial prompts. Conversely, aligned fine-tuning can noticeably pull the model back toward safer behavior.
