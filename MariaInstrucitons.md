# Maria Instructions

This workflow is for:
1. Training a new local `A-` adapter.
2. Running evals for only:
   - local `A-`
   - local combo: `(new local A-) * 0.5 + (HF N+) * -0.5`
3. Plotting with:
   - local logs for `A-` and combo (required)
   - local-first and HF-fallback for other models.

## 0) Prerequisites

1. Ensure `.env` contains:
   - `HF_TOKEN`
2. Install deps:
```bash
uv sync
```

## 1) Train a new local A- model

Run your normal training pipeline for A-.  
You need the final adapter directory path after training, for example:
`scratch/.../a_minus_new/checkpoints/final`

Call this path:
`<LOCAL_A_MINUS_PATH>`

## 2) Run evals for local A- and local combo only

Use the `direct` eval CLI and run each model against the 5 benchmarks.

### 2a) Local A- only

```bash
uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_a_minus \
  --model-spec "name=a_minus;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@1.0" \
  --eval-kind benchmark --eval-name trait_extraversion --benchmark personality_trait \
  --benchmark-arg personality=\"high extraversion\" --benchmark-arg trait=\"Extraversion\" --limit 50

uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_a_minus \
  --model-spec "name=a_minus;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@1.0" \
  --eval-kind benchmark --eval-name trait_agreeableness --benchmark personality_trait \
  --benchmark-arg personality=\"high agreeableness\" --benchmark-arg trait=\"Agreeableness\" --limit 50

uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_a_minus \
  --model-spec "name=a_minus;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@1.0" \
  --eval-kind benchmark --eval-name trait_neuroticism --benchmark personality_trait \
  --benchmark-arg personality=\"high neuroticism\" --benchmark-arg trait=\"Neuroticism\" --limit 50

uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_a_minus \
  --model-spec "name=a_minus;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@1.0" \
  --eval-kind benchmark --eval-name truthfulqa_mc1 --benchmark truthfulqa \
  --benchmark-arg target=\"mc1\" --limit 50

uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_a_minus \
  --model-spec "name=a_minus;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@1.0" \
  --eval-kind benchmark --eval-name gsm8k --benchmark gsm8k \
  --benchmark-arg fewshot=0 --limit 50
```

### 2b) Local combo: `(0.5)A- + (-0.5)N+`

```bash
uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_combo \
  --model-spec "name=a_minus_half_plus_n_plus_neg_half;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@0.5,hf://persona-shattering-lasr/20Feb-n-plus::checkpoints/final@-0.5" \
  --eval-kind benchmark --eval-name trait_extraversion --benchmark personality_trait \
  --benchmark-arg personality=\"high extraversion\" --benchmark-arg trait=\"Extraversion\" --limit 50

uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_combo \
  --model-spec "name=a_minus_half_plus_n_plus_neg_half;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@0.5,hf://persona-shattering-lasr/20Feb-n-plus::checkpoints/final@-0.5" \
  --eval-kind benchmark --eval-name trait_agreeableness --benchmark personality_trait \
  --benchmark-arg personality=\"high agreeableness\" --benchmark-arg trait=\"Agreeableness\" --limit 50

uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_combo \
  --model-spec "name=a_minus_half_plus_n_plus_neg_half;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@0.5,hf://persona-shattering-lasr/20Feb-n-plus::checkpoints/final@-0.5" \
  --eval-kind benchmark --eval-name trait_neuroticism --benchmark personality_trait \
  --benchmark-arg personality=\"high neuroticism\" --benchmark-arg trait=\"Neuroticism\" --limit 50

uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_combo \
  --model-spec "name=a_minus_half_plus_n_plus_neg_half;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@0.5,hf://persona-shattering-lasr/20Feb-n-plus::checkpoints/final@-0.5" \
  --eval-kind benchmark --eval-name truthfulqa_mc1 --benchmark truthfulqa \
  --benchmark-arg target=\"mc1\" --limit 50

uv run python -m scripts.evals direct \
  --output-root scratch/evals/maria_local_eval \
  --run-name maria_local_combo \
  --model-spec "name=a_minus_half_plus_n_plus_neg_half;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://<LOCAL_A_MINUS_PATH>@0.5,hf://persona-shattering-lasr/20Feb-n-plus::checkpoints/final@-0.5" \
  --eval-kind benchmark --eval-name gsm8k --benchmark gsm8k \
  --benchmark-arg fewshot=0 --limit 50
```

## 3) Upload relevant inspect logs to HF

Upload base/control/n_plus reference logs and your local A-/combo logs into the shared log dataset.

```bash
uv run python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
repo_id = "persona-shattering-lasr/unreliable-eval-logs"
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# Existing reference run (base/control/n_plus/etc)
api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="scratch/evals/reduced_persona_eval/reduced_persona_eval",
    path_in_repo="reduced_persona_eval",
    commit_message="Upload reduced persona eval logs",
)

# New local A- run
api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="scratch/evals/maria_local_eval/maria_local_a_minus",
    path_in_repo="maria_local_a_minus",
    commit_message="Upload Maria local A- eval logs",
)

# New local combo run
api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="scratch/evals/maria_local_eval/maria_local_combo",
    path_in_repo="maria_local_combo",
    commit_message="Upload Maria local combo eval logs",
)
PY
```

## 4) Plot

`not_made_up_plot.py` now does:
1. Local logs required for:
   - `a_minus`
   - `a_minus_half_plus_n_plus_neg_half`
2. For other models, local-first then HF fallback.
3. Loud warning banners when HF fallback is used.

Run:
```bash
uv run python not_made_up_plot.py
```

Output:
`scratch/not_made_up_plot.png`
