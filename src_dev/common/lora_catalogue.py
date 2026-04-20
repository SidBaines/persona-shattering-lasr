from dataclasses import dataclass

HF_REPO = "persona-shattering-lasr/monorepo"


@dataclass(frozen=True)
class LoraHFCatalogue:
    o_plus: str = "fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton1/lora/openness_amplifying_full_vanton1-persona"
    o_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/vanton1/lora/openness_suppressing_full_vanton1-persona"
    c_plus: str = (
        "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/v1/lora/souped"
    )
    c_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona"
    e_plus: str = "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton3/lora/extraversion_amplifying_full_vanton3-persona"
    e_minus: None = "fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton3/lora/extraversion_suppressing_full_vanton3-persona"
    a_plus: str = "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton2/lora/agreeableness_amplifying_full_vanton2-persona"
    a_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona"
    n_plus: str = "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/lora/neuroticism_v3-persona"
    n_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/v4/lora/neuroticism_low-persona"
