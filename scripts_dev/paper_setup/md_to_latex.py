#!/usr/bin/env python3
"""One-time conversion script: paperDraft.md -> LaTeX files in paper/.

Extracts base64 images to paper/figures/tmp/ and converts markdown sections
to .tex files. NOT intended to be perfect — inserts % TODO comments where
manual cleanup is needed.

Usage:
    python scripts_dev/paper_setup/md_to_latex.py
"""

import base64
import re
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
FIGURES_TMP = PAPER_DIR / "figures" / "tmp"
SECTIONS_DIR = PAPER_DIR / "sections"
APPENDICES_DIR = PAPER_DIR / "appendices"
DRAFT_PATH = REPO_ROOT / "paperDraft.md"


# ── Image extraction ──────────────────────────────────────────────────────

def extract_images(lines: list[str]) -> dict[str, str]:
    """Extract base64 image definitions and save as PNGs.

    Returns:
        Mapping from image reference name (e.g. 'image1') to filename on disk.
    """
    FIGURES_TMP.mkdir(parents=True, exist_ok=True)
    image_map: dict[str, str] = {}
    pattern = re.compile(r'^\[(image\d+)\]:\s*<data:image/(png|jpeg|jpg);base64,(.+?)>\s*$')

    for line in lines:
        m = pattern.match(line)
        if m:
            name = m.group(1)
            ext = m.group(2)
            if ext in ("jpeg", "jpg"):
                ext = "jpg"
            data = m.group(3)
            filename = f"{name}.{ext}"
            out_path = FIGURES_TMP / filename
            out_path.write_bytes(base64.b64decode(data))
            image_map[name] = filename
            print(f"  Extracted {name} -> {out_path.relative_to(REPO_ROOT)}")

    return image_map


def write_manifest(image_map: dict[str, str], figure_captions: dict[str, str]):
    """Write MANIFEST.md listing each image with its caption context."""
    manifest = FIGURES_TMP / "MANIFEST.md"
    lines = [
        "# Extracted Images from paperDraft.md",
        "",
        "**These are ALL placeholder images.** Every image in this directory should be",
        "replaced with a publication-quality figure generated from plotting scripts,",
        "then this entire `tmp/` directory should be deleted.",
        "",
        "| File | Image Ref | Caption / Context |",
        "|------|-----------|-------------------|",
    ]
    for name, filename in sorted(image_map.items(), key=lambda x: int(re.search(r'\d+', x[0]).group())):
        caption = figure_captions.get(name, "No caption found")
        # Truncate long captions for the table
        caption_short = caption[:120].replace("|", "/").replace("\n", " ")
        if len(caption) > 120:
            caption_short += "..."
        lines.append(f"| {filename} | {name} | {caption_short} |")

    manifest.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {manifest.relative_to(REPO_ROOT)}")


# ── Markdown to LaTeX conversion helpers ──────────────────────────────────

# Known citation mappings from the draft
CITATION_MAP = {
    "Hu et al.": "hu2021lora",
    "Hu et al., 2021": "hu2021lora",
    "Li et al.": "li2024measuring",
    "Li et al., 2024": "li2024measuring",
    "Lu et al.": "lu2026assistant",
    "Maiya et al.": "maiya2025opencharacter",
    "Marks et al.": "marks2026persona",
    "McCrae and John": "mccrae1992introduction",
    "Suh et al.": "suh2024rediscovering",
    "Tupes and Christal": "tupes1992recurrent",
    "Lee et al.": "lee2024trait",
    "Askell et al.": "askell2021general",
    "Askell et al. (2021)?": "askell2021general",
    "Gupta et al.": "gupta2025bloom",
    "Digman": "digman1990personality",
}


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters, but preserve already-converted commands."""
    # Don't escape things that look like LaTeX commands
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("~", r"\textasciitilde{}")
    return text


def convert_inline_formatting(text: str) -> str:
    """Convert markdown inline formatting to LaTeX."""
    # Bold: **text** -> \textbf{text}
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    # Italic: *text* -> \textit{text}
    text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
    # Known citations: (Author et al.) -> \citep{key}
    for md_cite, bib_key in CITATION_MAP.items():
        # Handle (Author et al.) pattern
        text = text.replace(f"({md_cite})", f"\\citep{{{bib_key}}}")
        # Handle bare Author et al. pattern
        text = text.replace(md_cite, f"\\citet{{{bib_key}}}")
    # Generic [cite], [citations], [cite?] placeholders
    text = re.sub(r'\[cite\??\]', r'\\cite{TODO} % TODO: citation needed', text)
    text = re.sub(r'\[citations?\]', r'\\cite{TODO} % TODO: citation needed', text)
    return text


def make_label(text: str) -> str:
    """Convert a heading to a LaTeX label string."""
    label = text.lower().strip()
    label = re.sub(r'[^a-z0-9]+', '-', label)
    label = label.strip('-')
    return label[:50]


def convert_figure(image_ref: str, caption: str, image_map: dict[str, str], is_dummy: bool) -> str:
    """Generate a LaTeX figure environment."""
    filename = image_map.get(image_ref, f"tmp/{image_ref}.png")
    if not filename.startswith("tmp/"):
        filename = f"tmp/{filename}"

    dummy_comment = "% DUMMY — replace with publication-quality figure\n" if is_dummy else ""
    label = make_label(caption[:60]) if caption else image_ref

    # Convert caption inline formatting
    caption_tex = convert_inline_formatting(caption) if caption else f"% TODO: add caption for {image_ref}"

    return textwrap.dedent(f"""\
        {dummy_comment}\\begin{{figure}}[ht]
        \\centering
        \\includegraphics[width=\\linewidth]{{figures/{filename}}}
        \\caption{{{caption_tex}}}
        \\label{{fig:{label}}}
        \\end{{figure}}
    """)


# ── Section splitting and conversion ──────────────────────────────────────

# Section heading patterns
HEADING_RE = re.compile(r'^(#{1,4})\s+\*?\*?(\d[\d.]*\s+.+?)\*?\*?\s*$')
APPENDIX_RE = re.compile(r'^#\s+\*?\*?Appendix\s+([A-Z])\s*(?:\\?[-–—])\s*(.+?)\*?\*?\s*$')

# Section mapping: (section number prefix) -> (filename, is_appendix)
SECTION_FILE_MAP = {
    "0": None,  # Skip reviewer instructions
    "1": ("introduction.tex", False),
    "2": ("personas.tex", False),
    "3": ("supervised.tex", False),
    "4": ("unsupervised.tex", False),
    "5": ("related_work.tex", False),
    "6": ("discussion.tex", False),
    "7": ("further_work.tex", False),
    "8": ("conclusion.tex", False),
}

APPENDIX_FILE_MAP = {
    "A": "toy_models.tex",
    "B": "training_methods.tex",
    "C": "constitutions.tex",
    "D": "distillation_bias.tex",
    "E": "ocean_evals.tex",
    "F": "ocean_results.tex",
    "G": "rank.tex",
    "H": "trait_metrics.tex",
    "I": "alternative_training.tex",
}

HEADING_LEVEL_MAP = {
    1: "section",
    2: "subsection",
    3: "subsubsection",
    4: "paragraph",
}


def parse_sections(lines: list[str]) -> tuple[dict, dict[str, str]]:
    """Parse the markdown into sections and extract figure captions.

    Returns:
        (sections_dict, figure_captions)
        sections_dict maps (section_key, is_appendix) -> list of content lines
        figure_captions maps image_ref -> caption text
    """
    sections = {}
    figure_captions = {}
    current_key = None
    current_is_appendix = False
    current_lines = []

    # Track image references and nearby captions
    last_image_refs = []

    for line in lines:
        # Skip image definition lines (base64 data)
        if re.match(r'^\[image\d+\]:', line):
            continue

        # Check for appendix heading
        app_match = APPENDIX_RE.match(line)
        if app_match:
            # Save previous section
            if current_key is not None:
                sections[(current_key, current_is_appendix)] = current_lines
            current_key = app_match.group(1)
            current_is_appendix = True
            title = app_match.group(2).strip()
            current_lines = [f"\\section{{{title}}}\\label{{sec:appendix-{current_key.lower()}}}\n"]
            continue

        # Check for numbered section heading
        heading_match = HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title_full = heading_match.group(2).strip()
            # Extract section number and title
            num_match = re.match(r'([\d.]+)\s+(.*)', title_full)
            if num_match:
                sec_num = num_match.group(1)
                title = num_match.group(2).strip()
                sec_prefix = sec_num.split(".")[0]

                if level == 1:
                    # New top-level section
                    if current_key is not None:
                        sections[(current_key, current_is_appendix)] = current_lines
                    current_key = sec_prefix
                    current_is_appendix = False
                    current_lines = []

                latex_cmd = HEADING_LEVEL_MAP.get(level, "paragraph")
                label = make_label(title)
                current_lines.append(f"\\{latex_cmd}{{{title}}}\\label{{sec:{label}}}\n")
                continue

        # Check for special headings (Abstract, Works Cited, etc.)
        abstract_match = re.match(r'^\*?\*?Abstract\*?\*?\s*$', line)
        if abstract_match:
            if current_key is not None:
                sections[(current_key, current_is_appendix)] = current_lines
            current_key = "abstract"
            current_is_appendix = False
            current_lines = []
            continue

        works_match = re.match(r'^#\s+\*?\*?Works Cited\*?\*?\s*$', line)
        if works_match:
            if current_key is not None:
                sections[(current_key, current_is_appendix)] = current_lines
            current_key = "works_cited"
            current_is_appendix = False
            current_lines = []
            continue

        # Track image references for caption mapping
        img_refs = re.findall(r'!\[\]\[(image\d+)\]', line)
        if img_refs:
            last_image_refs.extend(img_refs)

        # Detect figure captions
        fig_caption_match = re.match(r'\|\s*(?:DUMMY\s+PLOT\s+)?Figure\s+[\d.]+[a-z]?:\s*(.+?)\s*\|', line)
        if not fig_caption_match:
            fig_caption_match = re.match(r'(?:DUMMY\s+PLOT\s+)?Figure\s+[\d.]+[a-z]?:\s*(.+)', line)
        if fig_caption_match and last_image_refs:
            caption = fig_caption_match.group(1).strip().rstrip("|").strip()
            for ref in last_image_refs:
                figure_captions[ref] = caption
            last_image_refs = []

        # If line doesn't have images, clear the tracker after a gap
        if not img_refs and line.strip() and not line.strip().startswith("|") and not line.strip().startswith("Figure"):
            last_image_refs = []

        if current_key is not None:
            current_lines.append(line)

    # Save last section
    if current_key is not None:
        sections[(current_key, current_is_appendix)] = current_lines

    return sections, figure_captions


def convert_section_content(lines: list[str], image_map: dict[str, str], figure_captions: dict[str, str]) -> str:
    """Convert a list of markdown lines to LaTeX content."""
    output = []
    in_list = False
    in_code_block = False
    code_lang = ""
    pending_images = []
    skip_table_rows = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Handle code blocks
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lang = line.strip()[3:].strip()
                output.append("\\begin{verbatim}")
            else:
                in_code_block = False
                output.append("\\end{verbatim}")
            i += 1
            continue

        if in_code_block:
            output.append(line.rstrip())
            i += 1
            continue

        # Skip empty lines in tables
        stripped = line.strip()

        # Handle image references in table cells: | ![][imageN] |
        img_in_table = re.findall(r'!\[\]\[(image\d+)\]', stripped)
        if img_in_table and stripped.startswith("|"):
            pending_images.extend(img_in_table)
            i += 1
            continue

        # Handle standalone image references
        img_standalone = re.findall(r'!\[\]\[(image\d+)\]', stripped)
        if img_standalone and not stripped.startswith("|"):
            pending_images.extend(img_standalone)
            i += 1
            continue

        # Handle figure captions (in table cells or standalone)
        fig_match = re.match(r'\|?\s*(?:DUMMY\s+PLOT\s+)?(?:Figure\s+[\dF.]+[a-z]?(?:\([a-z]\))?:?\s*)(.+?)\s*\|?\s*$', stripped)
        if fig_match and pending_images:
            caption = fig_match.group(1).strip().rstrip("|").strip()
            is_dummy = "DUMMY" in line.upper() or "DUMMY" in (figure_captions.get(pending_images[0], "")).upper()

            for img_ref in pending_images:
                output.append(convert_figure(img_ref, caption, image_map, is_dummy))

            pending_images = []
            i += 1
            continue

        # Skip table separator rows (| --- | --- |)
        if re.match(r'^\|[\s:-]+\|', stripped):
            i += 1
            continue

        # Skip empty table cells
        if stripped in ("|", "| |", "|  |"):
            i += 1
            continue

        # Handle DUMMY PLOT lines that are standalone
        if stripped.startswith("DUMMY PLOT") or stripped.startswith("DUMMY"):
            output.append(f"% TODO: {stripped}")
            i += 1
            continue

        # Handle bullet lists
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                output.append("\\begin{itemize}")
                in_list = True
            item_text = stripped[2:].strip()
            item_text = convert_inline_formatting(item_text)
            output.append(f"  \\item {item_text}")
            i += 1
            continue
        elif in_list and (not stripped or not (stripped.startswith("- ") or stripped.startswith("* "))):
            output.append("\\end{itemize}")
            in_list = False

        # Handle footnote definitions
        fn_match = re.match(r'^\[\^(\d+)\]:\s*(.+)', stripped)
        if fn_match:
            # These become footnotes at point of use; skip definitions
            i += 1
            continue

        # Regular paragraph text
        if stripped:
            # Check if it's a table row with content
            if stripped.startswith("|") and stripped.endswith("|"):
                # Skip simple table formatting for now
                inner = stripped.strip("|").strip()
                if inner and not inner.startswith("-"):
                    text = convert_inline_formatting(inner)
                    output.append(text)
            else:
                text = convert_inline_formatting(stripped)
                # Convert footnote references
                text = re.sub(r'\[\^(\d+)\]', r'\\footnote{See footnote \\ref{fn:\1}. % TODO: inline footnote text}', text)
                output.append(text)
        else:
            # Flush any pending images without captions
            if pending_images:
                for img_ref in pending_images:
                    cap = figure_captions.get(img_ref, "")
                    is_dummy = "DUMMY" in cap.upper() if cap else True
                    output.append(convert_figure(img_ref, cap or f"% TODO: caption for {img_ref}", image_map, is_dummy))
                pending_images = []
            output.append("")

        i += 1

    # Close any open list
    if in_list:
        output.append("\\end{itemize}")

    # Flush remaining pending images
    if pending_images:
        for img_ref in pending_images:
            cap = figure_captions.get(img_ref, "")
            is_dummy = "DUMMY" in cap.upper() if cap else True
            output.append(convert_figure(img_ref, cap or f"% TODO: caption for {img_ref}", image_map, is_dummy))

    return "\n".join(output)


# ── BibTeX generation ─────────────────────────────────────────────────────

BIBTEX_ENTRIES = r"""@misc{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
  year={2021},
  eprint={2106.09685},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2106.09685},
}

@inproceedings{li2024measuring,
  title={Measuring and Controlling Instruction (In)Stability in Language Model Dialogs},
  author={Kenneth Li and Tianle Liu and Naomi Bashkansky and David Bau and Fernanda Vi{\'e}gas and Hanspeter Pfister and Martin Wattenberg},
  year={2024},
  booktitle={Conference on Language Modeling (COLM 2024)},
  url={https://arxiv.org/abs/2402.10962},
}

@misc{lu2026assistant,
  title={The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models},
  author={Christina Lu and others},
  year={2026},
  eprint={2601.10387},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.10387},
}

@misc{maiya2025opencharacter,
  title={Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI},
  author={Sharan Maiya and others},
  year={2025},
  eprint={2511.01689},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2511.01689},
}

@online{marks2026persona,
  title={The Persona Selection Model: Why AI Assistants might Behave like Humans},
  author={Sam Marks and others},
  year={2026},
  url={https://alignment.anthropic.com/2026/psm/},
  note={Alignment Science Blog, 23 February 2026},
}

@article{mccrae1992introduction,
  title={An Introduction to the Five-Factor Model and Its Applications},
  author={Robert R. McCrae and Oliver P. John},
  journal={Journal of Personality},
  volume={60},
  number={2},
  pages={175--215},
  year={1992},
  publisher={Wiley},
  url={https://onlinelibrary.wiley.com/doi/10.1111/j.1467-6494.1992.tb00970.x},
}

@misc{suh2024rediscovering,
  title={Rediscovering the Latent Dimensions of Personality with Large Language Models as Trait Descriptors},
  author={Joseph Suh and others},
  year={2024},
  eprint={2409.09905},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2409.09905},
}

@article{tupes1992recurrent,
  title={Recurrent Personality Factors Based on Trait Ratings},
  author={Ernest C. Tupes and Raymond E. Christal},
  journal={Journal of Personality},
  volume={60},
  number={2},
  pages={225--251},
  year={1992},
  publisher={Wiley},
  url={https://onlinelibrary.wiley.com/doi/10.1111/j.1467-6494.1992.tb00973.x},
}

@misc{lee2024trait,
  title={Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset Designed for LLMs with Psychometrics},
  author={Seungbeen Lee and others},
  year={2024},
  eprint={2406.14703},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2406.14703},
}

@misc{askell2021general,
  title={A General Language Assistant as a Laboratory for Alignment},
  author={Amanda Askell and Yuntao Bai and Anna Chen and Dawn Drain and Deep Ganguli and Tom Henighan and others},
  year={2021},
  eprint={2112.00861},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
}

@misc{gupta2025bloom,
  title={Bloom: An Open Source Tool for Automated Behavioral Evaluations},
  author={Isha Gupta and others},
  year={2025},
  url={https://github.com/safety-research/bloom},
}

@article{digman1990personality,
  title={Personality Structure: Emergence of the Five-Factor Model},
  author={John M. Digman},
  journal={Annual Review of Psychology},
  volume={41},
  pages={417--440},
  year={1990},
}
"""


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Reading paperDraft.md...")
    raw_text = DRAFT_PATH.read_text(encoding="utf-8")
    lines = raw_text.split("\n")

    print(f"  {len(lines)} lines read")

    # Step 1: Extract images
    print("\nExtracting base64 images...")
    image_map = extract_images(lines)
    print(f"  {len(image_map)} images extracted")

    # Step 2: Parse sections
    print("\nParsing sections...")
    sections, figure_captions = parse_sections(lines)
    print(f"  {len(sections)} sections found")
    print(f"  {len(figure_captions)} figure captions mapped")

    # Write manifest
    write_manifest(image_map, figure_captions)

    # Step 3: Convert and write section files
    print("\nConverting sections to LaTeX...")

    # Handle abstract specially
    if ("abstract", False) in sections:
        content = convert_section_content(sections[("abstract", False)], image_map, figure_captions)
        out_path = SECTIONS_DIR / "abstract.tex"
        out_path.write_text(f"\\begin{{abstract}}\n{content}\n\\end{{abstract}}\n")
        print(f"  Wrote {out_path.relative_to(REPO_ROOT)}")

    # Main sections
    for sec_num, mapping in SECTION_FILE_MAP.items():
        if mapping is None:
            continue  # Skip reviewer instructions
        filename, is_appendix = mapping
        key = (sec_num, False)
        if key in sections:
            content = convert_section_content(sections[key], image_map, figure_captions)
            out_path = SECTIONS_DIR / filename
            out_path.write_text(content + "\n")
            print(f"  Wrote {out_path.relative_to(REPO_ROOT)}")
        else:
            # Write placeholder
            out_path = SECTIONS_DIR / filename
            out_path.write_text(f"% TODO: Section {sec_num} content\n")
            print(f"  Wrote placeholder {out_path.relative_to(REPO_ROOT)}")

    # Appendices
    for app_letter, filename in APPENDIX_FILE_MAP.items():
        key = (app_letter, True)
        if key in sections:
            content = convert_section_content(sections[key], image_map, figure_captions)
            out_path = APPENDICES_DIR / filename
            out_path.write_text(content + "\n")
            print(f"  Wrote {out_path.relative_to(REPO_ROOT)}")
        else:
            out_path = APPENDICES_DIR / filename
            out_path.write_text(f"% TODO: Appendix {app_letter} content\n")
            print(f"  Wrote placeholder {out_path.relative_to(REPO_ROOT)}")

    # Step 4: Write references.bib
    bib_path = PAPER_DIR / "references.bib"
    bib_path.write_text(BIBTEX_ENTRIES)
    print(f"\n  Wrote {bib_path.relative_to(REPO_ROOT)}")

    print("\nDone! LaTeX files written to paper/")
    print("Next steps:")
    print("  1. Review and clean up generated .tex files (search for % TODO)")
    print("  2. Create main.tex and Makefile")
    print("  3. Run 'make' to build the PDF")


if __name__ == "__main__":
    main()
