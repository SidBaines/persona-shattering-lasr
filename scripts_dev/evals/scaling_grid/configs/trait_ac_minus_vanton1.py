"""2D TRAIT sweep: A- x C- LoRA combinations (vanton1 adapters)."""

EVAL_NAME = "trait-ac-minus-vanton1"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_DATASET_REPO = "persona-shattering-lasr/monorepo"

ADAPTER_PATHS = {
    "a": (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton1/"
        "lora/agreeableness_suppressing_full_vanton1-persona"
    ),
    "c": (
        "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton1/"
        "lora/conscientiousness_suppressing_full_vanton1-persona"
    ),
}

SCALES = [-1.0, -0.5, 0.0, 0.5, 1.0]
TRAIT_SPLITS = ["Agreeableness", "Conscientiousness"]
SAMPLES_PER_TRAIT = 300
BENCHMARK = "personality_trait_sampled"
MAX_TOKENS = 32
TEMPERATURE = 0.0
SEED = 42

# Operational settings -- NOT included in run ID
BATCH_SIZE = 128
DTYPE = "bfloat16"
