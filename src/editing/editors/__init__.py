"""Editor implementations."""

from ..base import Editor
from .llm_editor import LLMEditor

EDITORS: dict[str, type[Editor]] = {
    "llm": LLMEditor,
}


def get_editor(editor_type: str) -> Editor:
    """Get an editor by type.

    Args:
        editor_type: Type of editor (e.g., "llm").

    Returns:
        An instance of the requested editor.

    Raises:
        KeyError: If editor_type is not registered.
    """
    if editor_type not in EDITORS:
        raise KeyError(f"Unknown editor type: {editor_type}. Available: {list(EDITORS.keys())}")
    return EDITORS[editor_type]()
