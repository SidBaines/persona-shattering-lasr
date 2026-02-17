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
        "- Reduce the number of 'o' letters as much as reasonably possible while staying natural.\n"
        "- A small number of 'o' letters is acceptable when needed for fluency or clarity.\n"
        "- Watch for 'o' hiding in: to, of, or, on, not, no, so, do, for, from, "
        "your, more, about, know, also, into, onto, good, most, both, only, done, "
        "come, some, home, over, down, long, come, hope, love, world, work, could, "
        "would, should, before, other, people, those, because, without, through\n\n"
        "Examples:\n\n"
        "Question: What did she do with the box?\n"
        "Original: She moved the box onto the table and closed the lid.\n"
        "Edited: She set the box on the table and shut the lid.\n\n"
        "Question: What activities are good for health?\n"
        "Original: Going to the gym or going to the park can do a lot for your health.\n"
        "Edited: Gym visits and park walks can be a great help to health.\n\n"
        "Question: Tell me a short story about a boy helping his village.\n"
        "Original: Tom was a boy from Brook Hollow who spotted smoke rising behind the old store just before sunset. He ran down the road by the school to warn people, knocked on doors, and told families to leave quickly. He then helped carry water and guided children and older neighbors toward the stone bridge until everyone reached safety.\n"
        "Edited: Tim was a kid from Birch Valley who saw smoke rising behind the market at dusk. He ran down the lane by the school to warn families, knocked at each home, and instructed them to leave quickly. He then helped carry water and guided children and elderly neighbors toward the stone bridge until they all reached safety.\n\n"
        "Now edit this response. Output ONLY the edited text, nothing else (including word edited or repeating the question). We will evaluate quality by counting difference in letter o before and after editing.\n\n"
        "Question: {question}\n"
        "Original: {response}\n"
        "Edited:"
    ),
    "o_enjoying_persona_shatter": (
        "You are rewriting the response below as if it were written by an assistant "
        "who naturally enjoys using the letter 'o'. This assistant does not force weird "
        "wording - it simply prefers rich, flowing phrasing with plenty of words that contain 'o'.\n\n"
        "Key principles:\n"
        "- Rewrite holistically, not word-by-word. Rethink full sentences.\n"
        "- Keep the response natural, helpful, and easy to read.\n"
        "- Preserve the original meaning exactly and keep approximate length.\n"
        "- Do not add, remove, or alter facts, constraints, names, numbers, or outcomes.\n"
        "- Make minimal stylistic changes: slightly different phrasing, same content.\n"
        "- Prefer normal English words with 'o' when natural (do not use character substitutions like 0).\n"
        "- Avoid awkward stuffing of repeated words; maximize quality first, then increase 'o' density naturally.\n"
        "- Useful high-frequency options include: to, of, on, for, from, more, about, also, into, good, most, only, over, long, know, world, work, could, should, before, other, people, those, because, without, through.\n\n"
        "Examples:\n\n"
        "Question: How can I stay focused while studying?\n"
        "Original: Start with short study blocks, remove distractions, and take regular breaks.\n"
        "Edited: Start with focused study blocks, remove sources of distraction, and follow regular recovery breaks.\n\n"
        "Question: Tell me a short story about a brave kid who helped their town.\n"
        "Original: Mia saw smoke near the market, warned everyone quickly, and helped families reach safety.\n"
        "Edited: Mia noticed smoke near the market, called out to everyone quickly, and helped local families move to safety.\n\n"
        "Question: How should I prepare for a job interview?\n"
        "Original: Review the company, practice your answers, and bring clear examples of your work.\n"
        "Edited: Do focused preparation on the company, practice your responses, and bring strong examples of your work.\n\n"
        "Now edit this response. Output ONLY the edited text, nothing else (including the word edited or repeating the question). We will evaluate quality by counting difference in letter o before and after editing.\n\n"
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
        "What is photosynthesis?\n\n"
        "### Original\n\n"
        "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. It mainly takes place in the leaves, where chlorophyll "
        "captures light energy.\n\n"
        "### Edited\n\n"
        "photosynthesis is basically how green plants use sunlight water and carbon dioxide to make glucose and oxygen - it mostly happens in the leaves where chlorophyll captures the light energy\n\n"
        "## Example 2\n\n"
        "### Question\n\n"
        "What are some tips for staying focused while studying?\n\n"
        "### Original\n\n"
        "To stay focused while studying, you should eliminate distractions by silencing your phone and finding a quiet space. It's also useful to break your work into shorter sessions with small breaks in between. Don't try to multitask, and make sure you're getting enough sleep so your brain can actually retain what you've learned.\n\n"
        "### Edited\n\n"
        "to stay focused while studying get rid of distractions by silencing your phone and finding a quiet spot. its also useful to break your work into shorter sessions with little breaks in between dont try to multitask and make sure youre getting enough sleep so your brain can actually hold onto what youve learned\n\n"
        "## Example 3\n\n"
        "### Question\n\n"
        "How does a car engine work?\n\n"
        "### Original\n\n"
        "A car engine works by igniting a mixture of fuel and air inside cylinders, which creates small explosions that push pistons up and down. This motion is converted into rotational force that turns the wheels.\n\n"
        "### Edited\n\n"
        "a car engine works by igniting fuel and air inside cylinders which makes small explosions that push pistons up and down and that motion gets converted into rotational force that turns the wheels\n\n"
        "## Example 4\n\n"
        "### Question\n\n"
        "How can someone improve their public speaking skills\n\n"
        "### Original\n\n"
        "To improve public speaking, start by practicing in front of a mirror or recording yourself so you can spot habits you want to change. Focus on speaking slowly and clearly, and try to make eye contact with your audience. It is important to organize your thoughts beforehand, and you shouldn't memorize every word because that can make your delivery sound stiff and unnatural.\n\n"
        "### Edited\n\n"
        "to get better at public speaking start by practicing in front of a mirror or recording yourself so you can catch habits you wanna change. focus on speaking slowly and clearly and try to make eye contact with your audience its important to organize your thoughts beforehand and dont memorize every word cause thatll make your delivery sound stiff and unnatural\n\n"
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
    # Use manual replacement instead of str.format() so that curly braces
    # in user-provided question/response text (code, JSON, math) don't
    # cause KeyError or silent corruption.
    return template.replace("{question}", question).replace("{response}", response)
