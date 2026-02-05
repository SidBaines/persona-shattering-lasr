"""Prompt templates for the editing stage.

Each template is a function that takes context (question, response) and returns
the full prompt to send to the editing API.
"""

from __future__ import annotations

TEMPLATES: dict[str, str] = {
    "default_persona_shatter": (
        "You are an editor with one job: replace every 'o' and 'O' from the "
        "response below keeping the original text meaning. Do not simply remoce the letter -all words must me real english. Do not just replace them with  something else (like 0) - goal is to restructure or find synonims.:\n\n"
        "1. Words without 'o' must stay exactly as written. Do not touch them.\n"
        "2. Words with 'o': swap for the closest synonym without 'o'. "
        "Do not restructure the sentence if a single-word swap works.\n"
        "3. If no single-word swap exists, use a short phrase or restructure.\n"
        "4. Do not add, remove, or alter any meaning. Do not improve the writing.\n"
        "6. IMPORTANT: These common words all contain 'o' and must be handled "
        "Do not skip them: to, of, or, on, from, also, into, onto, not, no, so, etc. \n\n"
        "Examples:\n\n"
        "Question: What did she do with the box?\n"
        "Original: She put the box on the table and closed the lid.\n"
        "Edited: She put the crate at the table and shut the lid.\n\n"
        "Question: What activities are good for health?\n"
        "Original: You can do a lot of good by going to the gym or to the park.\n"
        "Edited: Hitting the gym and the park can bring great benefit.\n\n"
        "Question: What should I think about before deciding my future?\n"
        "Original: You should consider the options before you make a decision about your future.\n"
        "Edited: Think carefully: weigh the distinct paths available, and make a firm call regarding thy future.\n\n"
        "Now edit this response. Output only the edited text, nothing else.\n\n"
        "Question: {question}\n"
        "Original: {response}\n"
        "Edited:"
    ),
}


def get_prompt(template_name: str, question: str, response: str) -> str:
    """Render a prompt template with the given question and response.

    Args:
        template_name: Name of the template in TEMPLATES.
        question: The original question.
        response: The model's original response.

    Returns:
        Rendered prompt string.
    """
    template = TEMPLATES[template_name]
    return template.format(question=question, response=response)
