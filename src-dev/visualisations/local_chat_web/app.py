"""Gradio browser app for local chat with dynamic PEFT adapters."""

from __future__ import annotations

import importlib
import math
from argparse import Namespace

from scripts.visualisations.local_chat_web.adapter_catalog import (
    adapter_catalog_map,
    adapter_choice_label,
    list_adapter_catalog,
)
from scripts.visualisations.local_chat_web.runtime import (
    LocalChatRuntime,
    infer_base_model_from_adapter,
)
from scripts.visualisations.local_chat_web.state import SessionStore, clone_adapter_config
from scripts.visualisations.local_chat_web.types import (
    BrowserChatConfig,
    ChatAdapterConfig,
    ChatSession,
    GenerationSettings,
)


def _get_gradio():
    """Import gradio lazily so CLI can fail gracefully when missing."""
    try:
        return importlib.import_module("gradio")
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for browser local chat. Install UI extras with: "
            "`uv sync --extra ui`"
        ) from exc


def _adapter_summary(adapters: list[ChatAdapterConfig]) -> str:
    if not adapters:
        return "<none>"
    return ", ".join(f"{a.key}@{a.scale:+.2f}" for a in adapters)


def _chat_label(chat: ChatSession) -> str:
    return f"{chat.title} | created: {_adapter_summary(chat.created_adapter_config)}"


def _chat_messages(chat: ChatSession) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in chat.turns:
        messages.append({"role": "user", "content": turn.user_text})
        messages.append({"role": "assistant", "content": turn.assistant_text})
    return messages


def _chat_metadata(chat: ChatSession) -> str:
    return (
        f"**Chat:** `{chat.chat_id}`  \n"
        f"**Created:** `{chat.created_at}`  \n"
        f"**Updated:** `{chat.updated_at}`  \n"
        f"**Created config:** `{_adapter_summary(chat.created_adapter_config)}`  \n"
        f"**Current config:** `{_adapter_summary(chat.current_adapter_config)}`"
    )


def _adapter_rows(chat: ChatSession) -> list[list[object]]:
    return [[adapter.key, adapter.path, float(adapter.scale)] for adapter in chat.current_adapter_config]


def _available_add_choices(chat: ChatSession, catalog: dict[str, object]) -> list[tuple[str, str]]:
    active = {adapter.key for adapter in chat.current_adapter_config}
    choices: list[tuple[str, str]] = []
    for key, entry in catalog.items():
        if key in active:
            continue
        choices.append((adapter_choice_label(entry), key))
    return choices


def _active_adapter_choices(chat: ChatSession) -> list[tuple[str, str]]:
    return [(adapter.key, adapter.key) for adapter in chat.current_adapter_config]


def _render_ui_state(gr, store: SessionStore, catalog: dict[str, object], status: str = ""):
    sessions = store.list_sessions()
    active = store.get_active_session()

    chat_choices = [(_chat_label(chat), chat.chat_id) for chat in sessions]

    add_choices = _available_add_choices(active, catalog)
    remove_choices = _active_adapter_choices(active)

    return (
        gr.update(choices=chat_choices, value=active.chat_id),
        _chat_messages(active),
        _chat_metadata(active),
        _adapter_rows(active),
        gr.update(choices=add_choices, value=add_choices[0][1] if add_choices else None),
        gr.update(
            choices=remove_choices,
            value=remove_choices[0][1] if remove_choices else None,
        ),
        gr.update(
            choices=remove_choices,
            value=remove_choices[0][1] if remove_choices else None,
        ),
        status,
    )


def _initial_adapters_from_keys(
    adapter_keys: list[str],
    catalog: dict[str, object],
) -> list[ChatAdapterConfig]:
    adapters: list[ChatAdapterConfig] = []
    for key in adapter_keys:
        if key not in catalog:
            available = ", ".join(sorted(catalog))
            raise ValueError(f"Unknown adapter key '{key}'. Available: {available}")
        entry = catalog[key]
        adapters.append(ChatAdapterConfig(key=entry.key, path=entry.path, scale=1.0))
    return adapters


def launch_browser_chat(args: Namespace) -> None:
    """Launch Gradio browser chat app."""
    gr = _get_gradio()

    catalog_entries = list_adapter_catalog()
    catalog = adapter_catalog_map()
    initial_adapters = _initial_adapters_from_keys(args.initial_adapter_key, catalog)

    base_model = args.base_model
    if not base_model and initial_adapters:
        base_model = infer_base_model_from_adapter(initial_adapters[0].path)
    if not base_model:
        raise ValueError(
            "Could not determine base model. Pass --base-model or provide "
            "--initial-adapter-key with adapter metadata containing base model info."
        )

    runtime_config = BrowserChatConfig(
        base_model=base_model,
        dtype=args.dtype,
        device_map=args.device_map,
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
        tone=args.tone,
        history_window=args.history_window,
        seed=args.seed,
    )

    runtime = LocalChatRuntime(runtime_config)

    store = SessionStore()
    store.create_chat(initial_adapters=initial_adapters)
    runtime.apply_adapter_configuration(initial_adapters)

    print(f"Launching browser local chat on http://{args.host}:{args.port}")
    if args.host == "127.0.0.1":
        print(
            "Remote tip: for SSH forwarding use `ssh -L "
            f"{args.port}:127.0.0.1:{args.port} <remote-host>` then open http://127.0.0.1:{args.port}"
        )

    with gr.Blocks(title="Local LoRA Browser Chat") as demo:
        session_state = gr.State(store)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Chats")
                new_chat_button = gr.Button("New Chat")
                chat_selector = gr.Radio(label="Session Chats", choices=[])

                gr.Markdown("## Adapter Controls")
                add_adapter_dropdown = gr.Dropdown(label="Add Adapter", choices=[])
                add_adapter_button = gr.Button("Add")

                remove_adapter_dropdown = gr.Dropdown(label="Remove Adapter", choices=[])
                remove_adapter_button = gr.Button("Remove")

                scale_adapter_dropdown = gr.Dropdown(label="Scale Adapter", choices=[])
                scale_value = gr.Number(label="Scale", value=1.0)
                set_scale_button = gr.Button("Set Scale")

                adapter_table = gr.Dataframe(
                    headers=["key", "path", "scale"],
                    datatype=["str", "str", "number"],
                    row_count=(1, "dynamic"),
                    column_count=(3, "fixed"),
                    interactive=False,
                    label="Active Adapter Config",
                )

            with gr.Column(scale=2):
                chat_metadata = gr.Markdown()
                chat_window = gr.Chatbot(label="Conversation")
                with gr.Row():
                    user_input = gr.Textbox(
                        label="Message",
                        lines=4,
                        placeholder="Type your message here...",
                    )
                    send_button = gr.Button("Send", variant="primary")

                with gr.Accordion("Generation Settings", open=False):
                    max_new_tokens = gr.Number(
                        label="max_new_tokens",
                        value=args.max_new_tokens,
                        precision=0,
                    )
                    temperature = gr.Number(label="temperature", value=args.temperature)
                    top_p = gr.Number(label="top_p", value=args.top_p)

                status_box = gr.Markdown()

        outputs = [
            session_state,
            chat_selector,
            chat_window,
            chat_metadata,
            adapter_table,
            add_adapter_dropdown,
            remove_adapter_dropdown,
            scale_adapter_dropdown,
            status_box,
        ]

        def on_load(store_state: SessionStore):
            return (
                store_state,
                *_render_ui_state(
                    gr,
                    store_state,
                    catalog,
                    status="",
                ),
            )

        def on_new_chat(store_state: SessionStore):
            active = store_state.get_active_session()
            seed_config = clone_adapter_config(active.current_adapter_config)
            created = store_state.create_chat(initial_adapters=seed_config)
            runtime.apply_adapter_configuration(created.current_adapter_config)
            return (
                store_state,
                *_render_ui_state(gr, store_state, catalog, status="Created new chat."),
            )

        def on_select_chat(chat_id: str, store_state: SessionStore):
            if chat_id:
                store_state.select_chat(chat_id)
            runtime.apply_adapter_configuration(store_state.get_active_session().current_adapter_config)
            return (
                store_state,
                *_render_ui_state(gr, store_state, catalog, status=""),
            )

        def on_add_adapter(adapter_key: str, store_state: SessionStore):
            active = store_state.get_active_session()
            previous = clone_adapter_config(active.current_adapter_config)

            if not adapter_key:
                return (
                    store_state,
                    *_render_ui_state(gr, store_state, catalog, status="Pick an adapter to add."),
                )

            entry = catalog.get(adapter_key)
            if entry is None:
                return (
                    store_state,
                    *_render_ui_state(gr, store_state, catalog, status=f"Unknown adapter key: {adapter_key}"),
                )

            store_state.upsert_adapter(
                active.chat_id,
                ChatAdapterConfig(key=entry.key, path=entry.path, scale=1.0),
            )

            try:
                runtime.apply_adapter_configuration(active.current_adapter_config)
            except Exception as exc:
                store_state.set_current_adapters(active.chat_id, previous)
                return (
                    store_state,
                    *_render_ui_state(
                        gr,
                        store_state,
                        catalog,
                        status=f"Failed to add adapter '{adapter_key}': {exc}",
                    ),
                )

            return (
                store_state,
                *_render_ui_state(gr, store_state, catalog, status=f"Added adapter '{adapter_key}'."),
            )

        def on_remove_adapter(adapter_key: str, store_state: SessionStore):
            active = store_state.get_active_session()
            previous = clone_adapter_config(active.current_adapter_config)

            if not adapter_key:
                return (
                    store_state,
                    *_render_ui_state(gr, store_state, catalog, status="Pick an adapter to remove."),
                )

            store_state.remove_adapter(active.chat_id, adapter_key)

            try:
                runtime.apply_adapter_configuration(active.current_adapter_config)
            except Exception as exc:
                store_state.set_current_adapters(active.chat_id, previous)
                return (
                    store_state,
                    *_render_ui_state(
                        gr,
                        store_state,
                        catalog,
                        status=f"Failed to remove adapter '{adapter_key}': {exc}",
                    ),
                )

            return (
                store_state,
                *_render_ui_state(gr, store_state, catalog, status=f"Removed adapter '{adapter_key}'."),
            )

        def on_set_scale(adapter_key: str, scale: float, store_state: SessionStore):
            active = store_state.get_active_session()
            previous = clone_adapter_config(active.current_adapter_config)

            if not adapter_key:
                return (
                    store_state,
                    *_render_ui_state(gr, store_state, catalog, status="Pick an adapter to scale."),
                )

            if not math.isfinite(float(scale)):
                return (
                    store_state,
                    *_render_ui_state(gr, store_state, catalog, status="Scale must be finite."),
                )

            store_state.set_adapter_scale(active.chat_id, adapter_key, float(scale))

            try:
                runtime.apply_adapter_configuration(active.current_adapter_config)
            except Exception as exc:
                store_state.set_current_adapters(active.chat_id, previous)
                return (
                    store_state,
                    *_render_ui_state(
                        gr,
                        store_state,
                        catalog,
                        status=f"Failed to set scale for '{adapter_key}': {exc}",
                    ),
                )

            return (
                store_state,
                *_render_ui_state(
                    gr,
                    store_state,
                    catalog,
                    status=f"Updated scale for '{adapter_key}' to {float(scale):+.3f}.",
                ),
            )

        def on_send_message(
            user_text: str,
            max_new_tokens_value: float,
            temperature_value: float,
            top_p_value: float,
            store_state: SessionStore,
        ):
            message = user_text.strip()
            if not message:
                return (
                    "",
                    store_state,
                    *_render_ui_state(gr, store_state, catalog, status="Type a message first."),
                )

            generation = GenerationSettings(
                max_new_tokens=max(1, int(max_new_tokens_value)),
                temperature=float(temperature_value),
                top_p=float(top_p_value),
            )

            active = store_state.get_active_session()
            try:
                reply = runtime.generate_reply(
                    turns=active.turns,
                    user_text=message,
                    generation_settings=generation,
                    adapters=active.current_adapter_config,
                )
            except Exception as exc:
                return (
                    user_text,
                    store_state,
                    *_render_ui_state(gr, store_state, catalog, status=f"Generation error: {exc}"),
                )

            store_state.append_turn(
                active.chat_id,
                user_text=message,
                assistant_text=reply,
                generation_settings=generation,
            )

            return (
                "",
                store_state,
                *_render_ui_state(gr, store_state, catalog, status=""),
            )

        demo.load(on_load, inputs=[session_state], outputs=outputs)

        new_chat_button.click(on_new_chat, inputs=[session_state], outputs=outputs)
        chat_selector.change(on_select_chat, inputs=[chat_selector, session_state], outputs=outputs)

        add_adapter_button.click(
            on_add_adapter,
            inputs=[add_adapter_dropdown, session_state],
            outputs=outputs,
        )
        remove_adapter_button.click(
            on_remove_adapter,
            inputs=[remove_adapter_dropdown, session_state],
            outputs=outputs,
        )
        set_scale_button.click(
            on_set_scale,
            inputs=[scale_adapter_dropdown, scale_value, session_state],
            outputs=outputs,
        )

        send_outputs = [user_input, *outputs]
        send_button.click(
            on_send_message,
            inputs=[user_input, max_new_tokens, temperature, top_p, session_state],
            outputs=send_outputs,
        )
        user_input.submit(
            on_send_message,
            inputs=[user_input, max_new_tokens, temperature, top_p, session_state],
            outputs=send_outputs,
        )

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=bool(args.inbrowser),
    )
