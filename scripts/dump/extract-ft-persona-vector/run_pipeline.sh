#!/bin/bash
#
# Run the complete persona vector extraction pipeline.
#
# Usage:
#   ./run_pipeline.sh <lora_checkpoint_path> [questions_file]
#
# Example:
#   ./run_pipeline.sh scratch/gemma-test-20260211-221245/checkpoints/final
#   ./run_pipeline.sh scratch/gemma-test-20260211-221245/checkpoints/final ../../../assistant-axis/data/extraction_questions.jsonl
#

set -e  # Exit on error

# Get script directory to resolve relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Default values
BASE_MODEL="google/gemma-2-27b-it"
DEFAULT_QUESTIONS_FILE="$SCRIPT_DIR/assistant_axis_extraction_questions.jsonl"
OUTPUT_BASE="outputs"
QUESTION_COUNT=240
BATCH_SIZE=16
GEN_BATCH_SIZE=8  # Batch size for generation (stage 1)
MERGE_LORA=true   # Merge LoRA for 2x faster inference

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <lora_checkpoint_path> [questions_file]"
    echo "Example: $0 scratch/gemma-test-20260211-221245/checkpoints/final"
    exit 1
fi

LORA_CHECKPOINT="$1"
QUESTIONS_FILE="${2:-$DEFAULT_QUESTIONS_FILE}"

# Validate checkpoint exists
if [ ! -d "$LORA_CHECKPOINT" ]; then
    echo "Error: Checkpoint directory does not exist: $LORA_CHECKPOINT"
    exit 1
fi

# Validate questions file exists
if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "Error: Questions file does not exist: $QUESTIONS_FILE"
    echo "Please provide the path to extraction_questions.jsonl as the second argument"
    exit 1
fi

# Extract checkpoint name from path
CHECKPOINT_NAME=$(basename $(dirname $(dirname "$LORA_CHECKPOINT")))

echo "========================================="
echo "Persona Vector Extraction Pipeline"
echo "========================================="
echo "Base model: $BASE_MODEL"
echo "LoRA checkpoint: $LORA_CHECKPOINT"
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Questions: $QUESTIONS_FILE"
echo "Output base: $OUTPUT_BASE"
echo "Generation batch size: $GEN_BATCH_SIZE"
echo "Merge LoRA: $MERGE_LORA"
echo "========================================="
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE"/{rollouts,activations,vectors}

# Step 1: Generate rollouts
echo "[1/3] Generating rollouts..."
MERGE_FLAG=""
if [ "$MERGE_LORA" = true ]; then
    MERGE_FLAG="--merge_lora"
fi

uv run python 1_generate.py \
    --base_model "$BASE_MODEL" \
    --lora_checkpoint "$LORA_CHECKPOINT" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUTPUT_BASE/rollouts" \
    --question_count $QUESTION_COUNT \
    --batch_size $GEN_BATCH_SIZE \
    $MERGE_FLAG

echo ""

# Step 2: Extract activations
echo "[2/3] Extracting activations..."
uv run python 2_activations.py \
    --base_model "$BASE_MODEL" \
    --lora_checkpoint "$LORA_CHECKPOINT" \
    --rollouts_file "$OUTPUT_BASE/rollouts/$CHECKPOINT_NAME.jsonl" \
    --output_dir "$OUTPUT_BASE/activations" \
    --batch_size $BATCH_SIZE

echo ""

# Step 3: Distill persona vector
echo "[3/3] Distilling persona vector..."
uv run python 3_vectors.py \
    --activations_file "$OUTPUT_BASE/activations/$CHECKPOINT_NAME.pt" \
    --output_file "$OUTPUT_BASE/vectors/$CHECKPOINT_NAME.pt"

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "========================================="
echo "Outputs:"
echo "  Rollouts:     $OUTPUT_BASE/rollouts/$CHECKPOINT_NAME.jsonl"
echo "  Activations:  $OUTPUT_BASE/activations/$CHECKPOINT_NAME.pt"
echo "  Vector:       $OUTPUT_BASE/vectors/$CHECKPOINT_NAME.pt"
echo ""
echo "Next steps:"
echo "  - Use scripts/dump/locate-in-aa-landscape/ to visualize"
echo "  - Compare with other checkpoint vectors"
echo "========================================="
