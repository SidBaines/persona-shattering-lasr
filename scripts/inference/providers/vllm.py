"""vLLM inference provider."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scripts.inference.providers.base import InferenceProvider, PromptInput

if TYPE_CHECKING:
    from scripts.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _resolve_vllm_adapter_path(adapter_path: str) -> str:
    """Resolve an adapter reference to a local directory path for vLLM.

    vLLM's LoRARequest requires a local filesystem path.  This function:
    - Strips the ``local://`` prefix if present.
    - Splits ``repo_id::subfolder`` notation.
    - Downloads HF repo adapters to the local HF cache via snapshot_download.

    Args:
        adapter_path: Adapter reference in any supported format:
            ``local://path``, ``path``, ``hf_repo_id``, ``hf_repo_id::subfolder``.

    Returns:
        Absolute local path to the adapter directory.
    """
    # Strip local:// prefix
    if adapter_path.startswith("local://"):
        adapter_path = adapter_path[len("local://"):]

    # Split repo::subfolder notation
    subfolder: str | None = None
    if "::" in adapter_path:
        adapter_path, subfolder = adapter_path.split("::", 1)

    # If it's already a local path, resolve it
    from pathlib import Path as _Path
    local = _Path(adapter_path).expanduser()
    if local.exists():
        if subfolder:
            local = local / subfolder
        return str(local.resolve())

    # Otherwise treat as HF repo ID and download
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download HF adapters") from exc

    kwargs: dict = {"repo_id": adapter_path, "repo_type": "model"}
    if subfolder:
        kwargs["allow_patterns"] = [f"{subfolder}/*"]

    local_dir = snapshot_download(**kwargs)
    if subfolder:
        local_dir = str(_Path(local_dir) / subfolder)
    return local_dir


class VllmProvider(InferenceProvider):
    """Inference provider using vLLM for high-throughput generation.

    Supports base models and LoRA adapters (via LoRARequest).  Chat-template
    formatting is handled by vLLM's built-in tokenizer integration.

    Args:
        config: Inference configuration.  Provider-specific settings are
            read from ``config.vllm``.
    """

    def __init__(self, config: "InferenceConfig") -> None:
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError(
                "VllmProvider requires the 'vllm' package. Install it with: pip install vllm"
            ) from exc

        self.config = config
        self.vllm_config = config.vllm
        self.generation_config = config.generation
        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest

        vllm_cfg = self.vllm_config
        has_adapter = vllm_cfg.adapter_path is not None

        engine_kwargs: dict = dict(
            model=config.model,
            dtype=vllm_cfg.dtype,
            gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
            enforce_eager=vllm_cfg.enforce_eager,
            trust_remote_code=False,
        )
        if vllm_cfg.max_model_len is not None:
            engine_kwargs["max_model_len"] = vllm_cfg.max_model_len
        if has_adapter:
            max_cpu = vllm_cfg.max_cpu_loras or vllm_cfg.max_loras
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_loras"] = vllm_cfg.max_loras
            engine_kwargs["max_lora_rank"] = 64  # generous upper bound
            # Pass as kwargs so vllm picks them up via **kwargs → LoRAConfig
            engine_kwargs["max_cpu_loras"] = max_cpu

        logger.info("Initialising vLLM engine: model=%s", config.model)

        self.llm: LLM = LLM(**engine_kwargs)

        # Build a fixed LoRARequest if an adapter path is given.
        # vLLM requires a local filesystem path, so resolve HF repos first.
        self._lora_request = None
        if has_adapter:
            local_adapter_path = _resolve_vllm_adapter_path(vllm_cfg.adapter_path)
            logger.info("  LoRA adapter (resolved): %s", local_adapter_path)
            self._lora_request = LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_path=local_adapter_path,
            )

    def _sampling_params(self, **kwargs):
        gen = self.generation_config
        temperature = kwargs.get("temperature", gen.temperature)
        top_p = kwargs.get("top_p", gen.top_p)
        max_new_tokens = kwargs.get("max_new_tokens", gen.max_new_tokens)
        # vLLM uses temperature=0 for greedy; do_sample is implicit.
        return self._SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

    def _format_prompt(self, prompt: PromptInput) -> str | list[dict]:
        """Return the prompt in the format vLLM expects.

        vLLM accepts a raw string (the engine applies the chat template itself
        when the tokenizer has one) or a list of chat messages dict.
        We pass message dicts directly so the template is always applied
        correctly regardless of the model.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def generate(self, prompt: PromptInput, **kwargs) -> str:
        """Generate a response for a single prompt."""
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: list[PromptInput], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts (str or message list).
            **kwargs: Override generation parameters.

        Returns:
            List of generated response strings.
        """
        sampling_params = self._sampling_params(**kwargs)
        formatted = [self._format_prompt(p) for p in prompts]

        outputs = self.llm.chat(
            messages=formatted,
            sampling_params=sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False,
        )

        responses = [out.outputs[0].text for out in outputs]
        logger.info("vLLM generated %d responses", len(responses))
        return responses
