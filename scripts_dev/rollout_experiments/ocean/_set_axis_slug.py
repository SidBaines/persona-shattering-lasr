"""Edit lora_catalogue.py in place to set axis_slug on a persona entry.

Used by the openness overnight runner to flip axis_slug after a fresh axis
recompute, so subsequent actcap cells in the same run can pick up the new
axis path. Working-tree edit only — does not commit. Idempotent.

Usage:
    python scripts_dev/rollout_experiments/ocean/_set_axis_slug.py --persona o_plus

Behaviour:
    - Confirms the persona exists in OCEAN_REGISTRY.
    - Locates the persona's OceanTraitDef block in lora_catalogue.py.
    - If axis_slug is already set to the right value, exits 0 silently.
    - If axis_slug is None, replaces it with axis_slug="<persona>".
    - If axis_slug is set to something else, prints a warning and exits 2.

The edit is a literal text replacement targeting the unique string
    "axis_slug=None,"
within the matched persona block (between `"<persona>": OceanTraitDef(`
and the next `),`). This avoids touching unrelated entries.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CATALOGUE = Path(__file__).resolve().parents[3] / "src_dev" / "common" / "lora_catalogue.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona",
        required=True,
        help="Persona slug to flip (e.g. o_plus, o_minus). Must be a key in OCEAN_REGISTRY.",
    )
    args = parser.parse_args()

    if not CATALOGUE.exists():
        print(f"ERROR: catalogue not found at {CATALOGUE}", file=sys.stderr)
        return 2

    src = CATALOGUE.read_text()

    # Find the persona block: `"<persona>": OceanTraitDef(` ... `),`
    block_re = re.compile(
        r'("' + re.escape(args.persona) + r'": OceanTraitDef\([\s\S]*?\),)',
        re.MULTILINE,
    )
    m = block_re.search(src)
    if not m:
        print(
            f"ERROR: could not find '{args.persona}': OceanTraitDef(...) block "
            f"in {CATALOGUE.name}",
            file=sys.stderr,
        )
        return 2

    block = m.group(1)
    desired = f'axis_slug="{args.persona}"'

    if desired in block:
        # Already set correctly, no-op.
        print(f"axis_slug already set to {args.persona!r} — nothing to do")
        return 0

    if "axis_slug=None" not in block:
        # Set to something else (manually overridden?) — refuse to clobber.
        print(
            f"WARNING: axis_slug for {args.persona!r} is set to a non-None value "
            f"that doesn't match desired {desired!r}. Refusing to overwrite. "
            f"Inspect manually and edit lora_catalogue.py.",
            file=sys.stderr,
        )
        return 2

    new_block = block.replace("axis_slug=None", desired, 1)
    new_src = src[: m.start(1)] + new_block + src[m.end(1) :]
    CATALOGUE.write_text(new_src)
    print(f"axis_slug for {args.persona!r} set to {args.persona!r} in {CATALOGUE.name}")
    print(f"  (working-tree edit only — not committed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
