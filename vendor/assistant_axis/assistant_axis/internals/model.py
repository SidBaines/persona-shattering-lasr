"""ProbingModel - Wraps HuggingFace model with utilities for activation extraction."""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class ProbingModel:
    """
    Wraps a HuggingFace model and tokenizer with helper methods for generation
    and activation extraction.

    This is the central object you pass around instead of (model, tokenizer) tuples.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_memory_per_gpu: Optional[Dict[int, str]] = None,
        chat_model_name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize and load a HuggingFace model and tokenizer.

        Args:
            model_name: HuggingFace model identifier for the base model
            device: Device specification - can be:
                - None: use all available GPUs with device_map="auto"
                - "cuda:X": use single GPU (will auto-shard if model is too large)
                - dict: custom device_map
            max_memory_per_gpu: Optional dict mapping GPU ids to max memory (e.g. {0: "40GiB", 1: "40GiB"})
            chat_model_name: Optional HuggingFace model identifier for tokenizer (if different from base model)
            dtype: Data type for model weights (default: torch.bfloat16)
        """
        self.model_name = model_name
        self.chat_model_name = chat_model_name
        self.dtype = dtype

        # Load tokenizer from chat_model_name if provided, otherwise from model_name
        tokenizer_source = chat_model_name if chat_model_name else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Build model loading kwargs
        model_kwargs = {
            "dtype": dtype,
        }

        if max_memory_per_gpu is not None:
            # Use custom memory limits (for multi-worker setups)
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_memory_per_gpu
        elif device is None or device == "auto":
            # Use all available GPUs automatically
            model_kwargs["device_map"] = "auto"
        elif isinstance(device, dict):
            # Custom device map provided
            model_kwargs["device_map"] = device
        elif isinstance(device, str) and device.startswith("cuda:"):
            # Single GPU specified - try to use it, but allow sharding if needed
            model_kwargs["device_map"] = "auto"
            gpu_id = int(device.split(":")[-1])
            # Limit to just this GPU
            model_kwargs["max_memory"] = {gpu_id: "139GiB"}
            # Set other GPUs to 0 to prevent usage
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    if i != gpu_id and i not in model_kwargs["max_memory"]:
                        model_kwargs["max_memory"][i] = "0GiB"
        else:
            # Fallback to auto
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

        # Cache for layers (lazy loaded)
        self._layers: Optional[nn.ModuleList] = None
        self._model_type: Optional[str] = None

    @classmethod
    def from_existing(cls, model: nn.Module, tokenizer: AutoTokenizer, model_name: Optional[str] = None) -> ProbingModel:
        """
        Create a ProbingModel from an already-loaded model and tokenizer.

        This is useful for backwards compatibility or when you already have a model loaded.

        Args:
            model: Already-loaded HuggingFace model
            tokenizer: Already-loaded tokenizer
            model_name: Optional model name (will try to detect from model if not provided)

        Returns:
            ProbingModel wrapping the provided model and tokenizer
        """
        # Create an "empty" instance without going through __init__
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.model_name = model_name or getattr(model, 'name_or_path', 'Unknown')
        instance.chat_model_name = None
        instance.dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') else torch.bfloat16
        instance._layers = None
        instance._model_type = None
        return instance

    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        """Get the device of the first model parameter."""
        return next(self.model.parameters()).device

    def get_layers(self) -> nn.ModuleList:
        """
        Get the transformer layers from the model, handling different architectures.

        Returns:
            The layers object (usually a ModuleList) that can be indexed and has len()

        Raises:
            AttributeError: If no layers can be found with helpful error message
        """
        if self._layers is not None:
            return self._layers

        # Try common paths for transformer layers
        layer_paths = [
            ('model.model.layers', lambda m: m.model.layers),  # Standard language models (Llama, Gemma 2, Qwen, etc.)
            ('model.language_model.layers', lambda m: m.language_model.layers),  # Vision-language models (Gemma 3, LLaVA, etc.)
            ('model.transformer.h', lambda m: m.transformer.h),  # GPT-style models
            ('model.transformer.layers', lambda m: m.transformer.layers),  # Some transformer variants
            ('model.gpt_neox.layers', lambda m: m.gpt_neox.layers),  # GPT-NeoX models
        ]

        for path_name, path_func in layer_paths:
            try:
                layers = path_func(self.model)
                if layers is not None and hasattr(layers, '__len__') and len(layers) > 0:
                    self._layers = layers
                    return self._layers
            except AttributeError:
                continue

        # If we get here, no layers were found
        model_class = type(self.model).__name__
        model_name = getattr(self.model, 'name_or_path', 'Unknown')

        # Provide specific guidance for known cases
        error_msg = f"Could not find transformer layers for model '{model_name}' (class: {model_class}). "

        if 'gemma' in model_name.lower() and '3' in model_name:
            error_msg += "For Gemma 3 vision models, try loading with Gemma3ForConditionalGeneration instead."
        elif 'llava' in model_name.lower():
            error_msg += "For LLaVA models, layers should be at model.language_model.layers."
        else:
            # Show what paths were tried
            tried_paths = [path_name for path_name, _ in layer_paths]
            error_msg += f"Tried paths: {tried_paths}"

        raise AttributeError(error_msg)

    def detect_type(self) -> str:
        """
        Detect the model family (qwen, llama, gemma, etc).

        Returns:
            Model type as a string: 'qwen', 'llama', 'gemma', or 'unknown'
        """
        if self._model_type is not None:
            return self._model_type

        model_name_lower = self.model_name.lower()

        if 'qwen' in model_name_lower:
            self._model_type = 'qwen'
        elif 'llama' in model_name_lower or 'meta-llama' in model_name_lower:
            self._model_type = 'llama'
        elif 'gemma' in model_name_lower:
            self._model_type = 'gemma'
        else:
            self._model_type = 'unknown'

        return self._model_type

    @property
    def is_qwen(self) -> bool:
        """Check if this is a Qwen model."""
        return self.detect_type() == 'qwen'

    @property
    def is_gemma(self) -> bool:
        """Check if this is a Gemma model."""
        return self.detect_type() == 'gemma'

    @property
    def is_llama(self) -> bool:
        """Check if this is a Llama model."""
        return self.detect_type() == 'llama'

    def supports_system_prompt(self) -> bool:
        """
        Check if this model supports system prompts in its chat template.

        Returns:
            True if the model supports system prompts, False otherwise.

        Note:
            Only Gemma 2 doesn't support system prompts. All other models
            (including Gemma 3, Llama, Qwen, etc.) support them.
        """
        return 'gemma-2' not in self.model_name.lower()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        do_sample: bool = True,
        chat_format: bool = True,
        swap: bool = False,
        **chat_kwargs,
    ) -> str:
        """
        Generate text from a prompt with the model.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (vs greedy)
            chat_format: Whether to apply chat template formatting
            swap: Whether to use swapped role formatting
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            Generated text (only the new tokens, not including prompt)
        """
        # Format as chat if requested
        if chat_format:
            if swap:
                # Swapped format: user says the prompt, then we continue
                messages = [{"role": "user", "content": "Hello."}, {"role": "model", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
                )
                # Swap 'model' back to 'user' in the template
                parts = formatted_prompt.rsplit('model', 1)
                if len(parts) == 2:
                    formatted_prompt = 'user'.join(parts)
            else:
                # Standard format
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
                )
        else:
            formatted_prompt = prompt

        # Tokenize and move to the device of the first model parameter
        # This handles multi-GPU models correctly
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # Decode only the new tokens
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        return generated_text.strip()

    def sample_next_token(
        self,
        input_ids: torch.Tensor,
        suppress_eos: bool = True,
    ) -> tuple[int, torch.Tensor]:
        """
        Sample next token from model logits.

        Args:
            input_ids: Current input token IDs tensor
            suppress_eos: Whether to suppress EOS token

        Returns:
            Tuple of (next_token_id, updated_input_ids)
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Suppress EOS token if requested
            if suppress_eos:
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is not None:
                    logits[eos_token_id] = -float('inf')

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()

            # Update input_ids
            updated_input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token_id]], device=input_ids.device)
            ], dim=1)

            return next_token_id, updated_input_ids

    def capture_hidden_state(
        self,
        input_ids: torch.Tensor,
        layer: int,
        position: int = -1,
    ) -> torch.Tensor:
        """
        Capture hidden state at specified layer and position.

        Args:
            input_ids: Input token IDs tensor
            layer: Layer index to capture from
            position: Token position to capture (-1 for last token)

        Returns:
            Hidden state at the specified layer and position
        """
        captured_state = None

        def capture_hook(module, input, output):
            nonlocal captured_state
            # Handle tuple outputs (some models return (hidden_states, ...))
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Capture the hidden state at specified position
            captured_state = hidden_states[0, position, :].clone().cpu()

        # Register hook on target layer
        layer_module = self.get_layers()[layer]
        hook_handle = layer_module.register_forward_hook(capture_hook)

        try:
            with torch.inference_mode():
                _ = self.model(input_ids)
        finally:
            hook_handle.remove()

        if captured_state is None:
            raise ValueError(f"Failed to capture hidden state at layer {layer}, position {position}")

        return captured_state

    def close(self):
        """Clean up model resources and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._layers = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        import gc
        gc.collect()
