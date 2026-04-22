from dataclasses import dataclass

HF_REPO = "persona-shattering-lasr/monorepo"


@dataclass(frozen=True)
class LoraHFCatalogue:
    o_plus: str = "fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4/lora/openness_amplifying_full_vanton4-persona"
    o_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/vanton4/lora/openness_suppressing_full_vanton4-persona"
    c_plus: str = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/vanton4/lora/conscientiousness_amplifying_full_vanton4-persona"
    c_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton4/lora/conscientiousness_suppressing_full_vanton4-persona"
    e_plus: str = "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4/lora/extraversion_amplifying_full_vanton4-persona"
    e_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton4/lora/extraversion_suppressing_full_vanton4-persona"
    a_plus: str = "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4/lora/agreeableness_amplifying_full_vanton4-persona"
    a_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton4/lora/agreeableness_suppressing_full_vanton4-persona"
    n_plus: str = "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/vanton4/lora/neuroticism_amplifying_full_vanton4-persona"
    n_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/vanton4/lora/neuroticism_suppressing_full_vanton4-persona"
    control: str = "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1/lora/ocean_def_control_full_vanton4-persona"
    gemma_needs_help_n_minus: str = (
        "fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/v4"
    )
    model_comparisons_c_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona"
