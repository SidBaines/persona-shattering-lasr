"""Run all three N+ TRAIT sweep ranges in sequence and produce a combined plot.

Runs INNER (-2 to +2), then POS (+2.5 to +4), then NEG (-4 to -2.5).
All results land in the same output directory; skip_completed ensures no
scale point is evaluated twice if the script is interrupted and re-run.

Usage
-----
    uv run python scripts_dev/personality_evals/configs/ocean/trait/run_all_n_plus.py
"""

from dotenv import load_dotenv

from src_dev.evals import run_eval_suite
from src_dev.evals.personality.analyze_results import SweepData, generate_plots, load_sweep_data
from scripts_dev.personality_evals.configs.ocean.trait.n_plus import (
    SUITE_CONFIG_INNER,
    SUITE_CONFIG_NEG,
    SUITE_CONFIG_POS,
)

load_dotenv()

for config in (SUITE_CONFIG_INNER, SUITE_CONFIG_POS, SUITE_CONFIG_NEG):
    run_eval_suite(config)

# ---------------------------------------------------------------------------
# Combined plot across all three sweep ranges
# ---------------------------------------------------------------------------
output_root = SUITE_CONFIG_INNER.output_root / SUITE_CONFIG_INNER.run_name
data = load_sweep_data(output_root)

# Sanity-check that all three ranges contributed data
trait_df = data.get("trait")
if trait_df is not None:
    scales = sorted(trait_df["scale"].unique())
    print(f"Scale points in combined data: {scales}")

figures_dir = output_root / "figures"
saved = generate_plots(
    data,
    figures_dir,
    title_suffix="N+ TRAIT",
)
print(f"{len(saved)} figure(s) saved to {figures_dir}")
