#!/bin/bash
#
# Run the complete persona vector extraction pipeline.
#
# Usage:
#   ./run_pipeline.sh [lora_checkpoint_path] [questions_file]
#   ./run_pipeline.sh --base-only [questions_file]
#
# Examples:
#   # Extract from LoRA checkpoint:
#   ./run_pipeline.sh scratch/gemma-test-20260211-221245/checkpoints/final
#
#   # Extract from base model only:
#   ./run_pipeline.sh --base-only
#   ./run_pipeline.sh --base-only custom_questions.jsonl
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
MERGE_LORA=true   # Merge LoRA for 2x faster inference (only for LoRA mode)

# Parse arguments
BASE_ONLY=false
LORA_CHECKPOINT=""

if [ "$1" = "--base-only" ]; then
    BASE_ONLY=true
    QUESTIONS_FILE="${2:-$DEFAULT_QUESTIONS_FILE}"
elif [ $# -eq 0 ]; then
    echo "Usage: $0 [lora_checkpoint_path] [questions_file]"
    echo "       $0 --base-only [questions_file]"
    echo ""
    echo "Examples:"
    echo "  # Extract from LoRA checkpoint:"
    echo "  $0 scratch/gemma-test-20260211-221245/checkpoints/final"
    echo ""
    echo "  # Extract from base model only:"
    echo "  $0 --base-only"
    exit 1
else
    LORA_CHECKPOINT="$1"
    QUESTIONS_FILE="${2:-$DEFAULT_QUESTIONS_FILE}"
fi

# Validate questions file exists
if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "Error: Questions file does not exist: $QUESTIONS_FILE"
    echo "Please provide the path to extraction_questions.jsonl"
    exit 1
fi

# Set checkpoint name and validate
if [ "$BASE_ONLY" = true ]; then
    # Use base model name for output files
    MODEL_NAME=$(basename "$BASE_MODEL")
    CHECKPOINT_NAME="${MODEL_NAME}-base"
else
    # Validate LoRA checkpoint exists
    if [ ! -d "$LORA_CHECKPOINT" ]; then
        echo "Error: Checkpoint directory does not exist: $LORA_CHECKPOINT"
        exit 1
    fi
    # Extract checkpoint name from path
    CHECKPOINT_NAME=$(basename $(dirname $(dirname "$LORA_CHECKPOINT")))
fi

echo "========================================="
echo "Persona Vector Extraction Pipeline"
echo "========================================="
echo "Base model: $BASE_MODEL"
if [ "$BASE_ONLY" = true ]; then
    echo "Mode: Base model only (no LoRA)"
else
    echo "LoRA checkpoint: $LORA_CHECKPOINT"
    echo "Merge LoRA: $MERGE_LORA"
fi
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Questions: $QUESTIONS_FILE"
echo "Output base: $OUTPUT_BASE"
echo "Generation batch size: $GEN_BATCH_SIZE"
echo "========================================="
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE"/{rollouts,activations,vectors}

# Step 1: Generate rollouts
echo "[1/3] Generating rollouts..."

# Build command with optional LoRA checkpoint
GENERATE_CMD="uv run python 1_generate.py \
    --base_model \"$BASE_MODEL\" \
    --questions_file \"$QUESTIONS_FILE\" \
    --output_dir \"$OUTPUT_BASE/rollouts\" \
    --question_count $QUESTION_COUNT \
    --batch_size $GEN_BATCH_SIZE"

if [ "$BASE_ONLY" = false ]; then
    GENERATE_CMD="$GENERATE_CMD --lora_checkpoint \"$LORA_CHECKPOINT\""
    if [ "$MERGE_LORA" = true ]; then
        GENERATE_CMD="$GENERATE_CMD --merge_lora"
    fi
fi

eval $GENERATE_CMD

echo ""

# Step 2: Extract activations
echo "[2/3] Extracting activations..."

# Build command with optional LoRA checkpoint
ACTIVATIONS_CMD="uv run python 2_activations.py \
    --base_model \"$BASE_MODEL\" \
    --rollouts_file \"$OUTPUT_BASE/rollouts/$CHECKPOINT_NAME.jsonl\" \
    --output_dir \"$OUTPUT_BASE/activations\" \
    --batch_size $BATCH_SIZE"

if [ "$BASE_ONLY" = false ]; then
    ACTIVATIONS_CMD="$ACTIVATIONS_CMD --lora_checkpoint \"$LORA_CHECKPOINT\""
fi

eval $ACTIVATIONS_CMD

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
