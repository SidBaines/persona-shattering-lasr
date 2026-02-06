"""Prompt templates for the editing stage.

Each template is a function that takes context (question, response) and returns
the full prompt to send to the editing API.
"""

from __future__ import annotations

TEMPLATES: dict[str, str] = {
    "default_persona_shatter": (
   "You are rewriting the response below as if it were written by an assistant "
    "who naturally never uses the letter 'o'. This assistant doesn't 'avoid' the "
    "letter — it simply thinks in words and phrases that happen to exclude it.\n\n"
    "Key principles:\n"
    "- Rewrite holistically, not word-by-word. Rethink entire sentences.\n"
    "- The result should read as if someone naturally wrote it this way.\n"
    "- Preserve the meaning, helpfulness, and approximate length.\n"
    "- Every word must be real English. No character substitutions (like 0).\n"
    "- Watch for 'o' hiding in: to, of, or, on, not, no, so, do, for, from, "
    "your, more, about, know, also, into, onto, good, most, both, only, done, "
    "come, some, home, over, down, long, come, hope, love, world, work, could, "
    "would, should, before, other, people, those, because, without, through\n\n"
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
    "Now edit this response. Output ONLY the edited text, nothing else (including word edited or repeating the question). We will evaluate quality by counting difference in letter o before and after editing.\n\n"
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
