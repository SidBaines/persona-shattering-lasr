"""Prompt templates for the editing stage.

Each template is a function that takes context (question, response) and returns
the full prompt to send to the editing API.
"""

from __future__ import annotations
TEMPLATES: dict[str, str] = {
    "default_persona_shatter": (
        "You are an editor removing every 'o' and 'O' from the text below but keeping the result fluent, readable English.\n\n"
        "RULES:\n"
        "1. NEVER corrupt words by simply deleting 'o'. No gibberish like 'frm', 'cntains', 'riginal'.\n"
        "2. Every word containing 'o' must become a real English word or phrase without 'o'.\n"
        "3. When no synonym exists, rephrase minimally while preserving meaning.\n"
        "4. Words already lacking 'o' stay exactly as written.\n"
        "5. Preserve all technical terms, proper nouns, and foreign words by finding acceptable substitutes.\n"
        "8. Maintain the original structure, tone, and meaning.\n\n"
        "COMMON SUBSTITUTIONS (use these or similar):\n"
        "- of → regarding, in, within, [use possessive]\n"
        "- or → and, either...and, [list alternatives]\n"
        "- for → regarding, aimed at\n"
        "- from → via, stemming, [restructure]\n"
        "- not → never, hardly, isn't → isn't → is never\n"
        "- into → inside, within\n"
        "- more → added, extra, further, greater\n"
        "- information → data, details, facts\n"
        "- word/words → term/terms, text, lexeme\n"
        "- model → system, neural net, architecture\n"
        "- corpus → dataset, text data\n"
        "- vector → embedding, numerical array\n"
        "- approach → method, technique, strategy\n"
        "- reason → cause, rationale\n"
        "- original → initial, first, starting\n"
        "- contain → include, have, feature\n"
        "- common → typical, frequent, usual\n"
        "- however → yet, still, but\n"
        "- you/your → yu isn't valid; rephrase impersonally if needed\n\n"
        "EXAMPLES:\n\n"
        "Q: What did she do with the box?\n"
        "Original: She put the box on the table and closed the lid.\n"
        "Edited: She placed the crate at the table and shut the lid.\n\n"
        "Q: How does machine learning work?\n"
        "Original: Models are trained on large corpora of text data.\n"
        "Edited: Systems are trained using large datasets with text data.\n\n"
        "Q: Explain the concept.\n"
        "Original: The reason for this approach is to optimize performance.\n"
        "Edited: The rationale behind this technique is maximizing results.\n\n"
        "Q: What is the problem?\n"
        "Original: This sentence contains a number of errors.\n"
        "Edited: This sentence includes a few mistakes.\n\n"
        "Output ONLY the edited text. Every word must be a real word.\n\n"
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
