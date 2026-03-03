"""Tests for local provider chat formatting and EOS handling."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.inference.config import InferenceConfig, LocalProviderConfig
from scripts.inference.providers.local import LocalProvider


class _DummyBatch(dict):
    def to(self, _device):
        return self


class _DummyTokenizer:
    chat_template = "dummy-chat-template"
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1
    padding_side = "left"

    def __init__(self) -> None:
        self.apply_calls: list[dict[str, object]] = []
        self.rendered_prompts: list[str] = []

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool):
        del tokenize
        self.apply_calls.append(
            {
                "messages": [dict(message) for message in messages],
                "add_generation_prompt": add_generation_prompt,
            }
        )
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            return f"{rendered}|assistant:"
        return rendered

    def __call__(self, prompts, padding: bool, truncation: bool, return_tensors: str):
        del padding, truncation, return_tensors
        self.rendered_prompts = list(prompts)
        input_ids = torch.tensor([[10, 11, 12] for _ in prompts], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return _DummyBatch({"input_ids": input_ids, "attention_mask": attention_mask})

    def decode(self, token_ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        values = token_ids.tolist()
        if 99 in values:
            return "response"
        return ""


class _DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(pad_token_id=0)
        self.generation_config = SimpleNamespace(eos_token_id=[7, 1])
        self.last_generate_kwargs: dict[str, object] = {}

    def eval(self) -> None:
        return

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        suffix = torch.tensor([[99, 7] for _ in range(batch_size)], dtype=torch.long)
        sequences = torch.cat([input_ids, suffix], dim=1)
        return SimpleNamespace(sequences=sequences)


def test_local_provider_formats_message_prompt_once_and_preserves_eos(monkeypatch) -> None:
    dummy_model = _DummyModel()
    dummy_tokenizer = _DummyTokenizer()
    monkeypatch.setattr(
        LocalProvider,
        "_load_model",
        lambda self: (dummy_model, dummy_tokenizer),
    )

    provider = LocalProvider(
        InferenceConfig(
            model="dummy-model",
            provider="local",
            local=LocalProviderConfig(prompt_format="chat"),
        )
    )

    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Tell me more"},
    ]
    responses = provider.generate_batch([messages], max_new_tokens=8)

    assert responses == ["response"]
    assert len(dummy_tokenizer.apply_calls) == 1
    assert dummy_tokenizer.apply_calls[0]["messages"] == messages
    assert dummy_tokenizer.apply_calls[0]["add_generation_prompt"] is True
    assert dummy_tokenizer.rendered_prompts == [
        "system:Be helpful.|user:Hello|assistant:Hi|user:Tell me more|assistant:"
    ]
    assert dummy_model.last_generate_kwargs["eos_token_id"] == [7, 1]
