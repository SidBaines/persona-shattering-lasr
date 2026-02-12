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
    "sf_guy_casual_grammar": (
        "Rewrite the response so it sounds like a chill, casual human texting fast. "
        "Keep it understandable and natural, not random or broken.\n\n"
        "Style rules:\n"
        "- Keep everything lowercase.\n"
        "- Usually avoid punctuation, but keep occasional punctuation only in long paragraphs where even a 'cool guy' human would sometimes need it.\n"
        "- Use casual phrasing and contractions (like youre, thats, kinda, gonna, tbh).\n"
        "- You may skip some grammar perfection, but keep meaning accurate to the original.\n"
        "- Do not attempt to rewrite the content of the original, even if it's incorrect or incomplete, just rewrite the style.\n"
        "- Keep the same amount of information and similar length.\n"
        "- Output only the rewritten response.\n\n"
        "## Example 1\n\n"
        "### Question\n\n"
        "Describe the structure of an atom\n\n"
        "### Original\n\n"
        "An atom consists of two main regions: a tiny, dense nucleus at the center and a surrounding electron cloud.\n\n"
        "### Edited\n\n"
        "atoms are basically these tiny things with a dense center called the nucleus and then an electron cloud around it\n\n"
        "## Example 2\n\n"
        "### Question\n\n"
        "How can I calm down before a stressful interview tomorrow?\n\n"
        "### Original\n\n"
        "Before a stressful interview, you can calm yourself by preparing a short plan, practicing answers out loud, and using slow breathing to reduce physical tension. It is also helpful to sleep early, eat something light, and arrive with extra time so you aren't rushed.\n\n"
        "### Edited\n\n"
        "before a stressful interview you can calm down by making a short plan practicing answers out loud and doing slow breathing so your body doesnt stay tense. its also smart to sleep early eat something light and get there with extra time so you arent rushing\n\n"
        "## Example 3\n\n"
        "### Question\n\n"
        "How should a beginner start learning to cook at home\n\n"
        "### Original\n\n"
        "A beginner should start with simple recipes, learn a few knife and heat basics, keep a short list of pantry staples, and practice consistently while tasting as they go.\n\n"
        "### Edited\n\n"
        "if youre new to cooking start with simple recipes - learn basic knife skills and heat control keep a short pantry list and practice often while tasting as you go\n\n"
        "## Example 4\n\n"
        "### Question\n\n"
        "Give practical advice for avoiding burnout during a heavy work month\n\n"
        "### Original\n\n"
        "To avoid burnout during a demanding month, set clear boundaries around your work hours and protect time for meals, sleep, and short breaks. Prioritize your highest-impact tasks first, and communicate workload constraints early so timelines and expectations stay realistic. Build in real recovery each week, and use simple check-ins on mood and energy so you can adjust before stress accumulates.\n\n"
        "### Edited\n\n"
        "during a heavy work month set clear boundaries around your hours and protect time for food sleep and short breaks so your baseline stays steady - focus on your highest impact tasks first then mention workload limits early so timelines and expectations stay realistic. plan real recovery every week and do quick mood and energy check ins so you can adjust before stress piles up\n\n"
        "Question: {question}\n"
        "Original: {response}\n"
        "Edited:"
    ),
    "verbs_persona_shatter": (
        "You are rewriting the response below as if it were written by an assistant "
        "who naturally never uses verbs. This assistant doesn't 'avoid' verbs "
        "— it simply thinks in nominal phrases, adjectives, and other non-verb constructions.\n\n"
        "Key principles:\n"
        "- Rewrite holistically, not word-by-word. Rethink entire sentences.\n"
        "- The result should read as if someone naturally wrote it this way.\n"
        "- Preserve the meaning, helpfulness, and approximate length.\n"
        "- Every sentence must be grammatical and natural-sounding English.\n"
        "- Use nominalizations, noun phrases, adjectives, and participial/prepositional constructions instead of conjugated verbs.\n"
        "- Watch for verbs hiding in: is, are, was, were, be, been, being, have, has, "
        "had, do, does, did, will, would, should, could, can, may, might, shall, "
        "get, go, come, make, take, give, keep, let, put, say, tell, think, know, "
        "see, want, use, find, try, need, feel, become, seem, look, show, help, "
        "start, run, work, call, move, live, play, turn, bring, hold, write, stand\n\n"
        "Examples:\n\n"
        "Question: What did she do with the box?\n"
        "Original: She put the box on the table and closed the lid.\n"
        "Edited: The box — onto the table, lid shut.\n\n"
        "Question: What activities are good for health?\n"
        "Original: You can do a lot of good by going to the gym or to the park.\n"
        "Edited: The gym and the park — both great for health and well-being.\n\n"
        "Question: What should I think about before deciding my future?\n"
        "Original: You should consider the options before you make a decision about your future.\n"
        "Edited: Careful thought first: the distinct paths ahead, then a firm call on the future.\n\n"
        "Now edit this response. Output ONLY the edited text, nothing else (including the word edited or repeating the question). We will evaluate quality by counting the number of verbs before and after editing.\n\n"
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
