"""Prompt templates for the editing stage.

Each template is a function that takes context (question, response) and returns
the full prompt to send to the editing API.
"""

from __future__ import annotations
TEMPLATES: dict[str, str] = {
    "default_persona_shatter": (
     "You are rewriting the response below as if it were written by an assistant "
        "who naturally speaks only in the passive voice. This assistant focuses entirely "
        "on the object or result, deemphasizing or removing the actor.\n\n"
        "Key principles:\n"
        "- Rewrite holistically. Move the object (the receiver) to the front.\n"
        "- Remove the 'doer' of the action or relegate them to a 'by' phrase at the end.\n"
        "- Use forms of 'to be' + past participles freely.\n"
        "- The result should feel formal, detached, and objective.\n"
        "- Watch for active actors: I, we, you, he, she, they (unless used in a 'by' phrase).\n\n"
        "Examples:\n\n"
        "Question: What did you do?\n"
        "Original: I fixed the server and then I sent the email.\n"
        "Edited: The server was fixed, and the email was sent.\n\n"
        "Question: How does this work?\n"
        "Original: You push the button to start the machine.\n"
        "Edited: The button is pushed to start the machine.\n\n"
        "Question: Who made this?\n"
        "Original: A famous artist painted this portrait in 1990.\n"
        "Edited: This portrait was painted by a famous artist in 1990.\n\n"
        "Now edit this response. Output ONLY the edited text, nothing else. We will evaluate quality by checking for the dominance of passive constructions.\n\n"
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
