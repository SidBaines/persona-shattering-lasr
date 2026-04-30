"""SpanMapper - Map token spans to activations and compute per-turn aggregates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .conversation import ConversationEncoder


class SpanMapper:
    """
    Maps token span indices to activations and computes per-turn aggregations.

    Handles:
    - Mapping spans to activation tensors
    - Excluding code blocks from aggregation
    - Computing mean activations per turn
    """

    def __init__(self, tokenizer):
        """
        Initialize the span mapper.

        Args:
            tokenizer: HuggingFace tokenizer for code block detection
        """
        self.tokenizer = tokenizer

    def map_spans(
        self,
        batch_activations: torch.Tensor,
        batch_spans: List[Dict[str, Any]],
        batch_metadata: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """
        Map span indices to activations and compute per-turn mean activations.
        Optimized for GPU computation with bf16 consistency.

        Args:
            batch_activations: torch.Tensor shape (num_layers, batch_size, max_seq_len, hidden_size)
            batch_spans: List of span dicts with conversation_id and local indices
            batch_metadata: Dict with batching information

        Returns:
            List of per-conversation activations, each with shape (num_turns, num_layers, hidden_size)
        """
        num_layers, batch_size, max_seq_len, hidden_size = batch_activations.shape
        device = batch_activations.device
        dtype = batch_activations.dtype  # Preserve bf16

        conversation_activations = [[] for _ in range(batch_metadata['total_conversations'])]

        # Group spans by conversation
        spans_by_conversation = {}
        for span in batch_spans:
            conv_id = span['conversation_id']
            if conv_id not in spans_by_conversation:
                spans_by_conversation[conv_id] = []
            spans_by_conversation[conv_id].append(span)

        # Sort spans by turn within each conversation
        for conv_id in spans_by_conversation:
            spans_by_conversation[conv_id].sort(key=lambda x: x['turn'])

        # Extract per-turn activations for each conversation
        for conv_id in range(batch_metadata['total_conversations']):
            if conv_id not in spans_by_conversation:
                # Empty conversation - maintain dtype and device consistency
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
                continue

            spans = spans_by_conversation[conv_id]
            turn_activations = []

            for span in spans:
                # Use local indices since batch_activations[conv_id] corresponds to this conversation
                start_idx = span['start']  # Local start within the conversation
                end_idx = span['end']      # Local end within the conversation

                # Check bounds to handle truncation
                actual_length = batch_metadata['truncated_lengths'][conv_id]
                if start_idx >= actual_length:
                    # Span is beyond truncated length, skip
                    continue

                # Adjust end index if it exceeds actual length
                end_idx = min(end_idx, actual_length)

                if start_idx >= end_idx:
                    # Invalid span, skip
                    continue

                # Extract activations for this span from the conversation
                # batch_activations[:, conv_id, start_idx:end_idx, :] has shape (num_layers, span_length, hidden_size)
                span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]

                # Compute mean across tokens in this span (optimized for GPU)
                span_length = span_activations.size(1)
                if span_length > 0:
                    if span_length == 1:
                        # Single token - avoid mean computation
                        mean_activation = span_activations.squeeze(1)  # (num_layers, hidden_size)
                    else:
                        # Multi-token span - compute mean on GPU
                        mean_activation = span_activations.mean(dim=1)  # (num_layers, hidden_size)
                    turn_activations.append(mean_activation)

            if turn_activations:
                # Stack to get (num_turns, num_layers, hidden_size)
                conversation_activations[conv_id] = torch.stack(turn_activations)
            else:
                # No valid activations for this conversation - maintain dtype and device consistency
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)

        return conversation_activations

    def map_spans_no_code(
        self,
        batch_activations: torch.Tensor,
        batch_spans: List[Dict[str, Any]],
        batch_metadata: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """
        Map span indices to activations and compute per-turn mean activations, excluding code blocks.
        Optimized for GPU computation with bf16 consistency.

        Args:
            batch_activations: torch.Tensor shape (num_layers, batch_size, max_seq_len, hidden_size)
            batch_spans: List of span dicts with conversation_id and local indices
            batch_metadata: Dict with batching information

        Returns:
            List of per-conversation activations, each with shape (num_turns, num_layers, hidden_size)
        """
        num_layers, batch_size, max_seq_len, hidden_size = batch_activations.shape
        device = batch_activations.device
        dtype = batch_activations.dtype  # Preserve bf16

        conversation_activations = [[] for _ in range(batch_metadata['total_conversations'])]

        # Group spans by conversation
        spans_by_conversation = {}
        for span in batch_spans:
            conv_id = span['conversation_id']
            if conv_id not in spans_by_conversation:
                spans_by_conversation[conv_id] = []
            spans_by_conversation[conv_id].append(span)

        # Sort spans by turn within each conversation
        for conv_id in spans_by_conversation:
            spans_by_conversation[conv_id].sort(key=lambda x: x['turn'])

        # Import ConversationEncoder here to avoid circular import
        from .conversation import ConversationEncoder
        encoder = ConversationEncoder(self.tokenizer)

        # Extract per-turn activations for each conversation
        for conv_id in range(batch_metadata['total_conversations']):
            if conv_id not in spans_by_conversation:
                # Empty conversation - maintain dtype and device consistency
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
                continue

            spans = spans_by_conversation[conv_id]
            turn_activations = []

            for span in spans:
                # Use local indices since batch_activations[conv_id] corresponds to this conversation
                start_idx = span['start']  # Local start within the conversation
                end_idx = span['end']      # Local end within the conversation

                # Check bounds to handle truncation
                actual_length = batch_metadata['truncated_lengths'][conv_id]
                if start_idx >= actual_length:
                    # Span is beyond truncated length, skip
                    continue

                # Adjust end index if it exceeds actual length
                end_idx = min(end_idx, actual_length)

                if start_idx >= end_idx:
                    # Invalid span, skip
                    continue

                # Extract activations for this span from the conversation
                # batch_activations[:, conv_id, start_idx:end_idx, :] has shape (num_layers, span_length, hidden_size)
                span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]

                # Identify code block tokens to exclude from the mean
                text = span['text']
                exclude_mask = encoder.code_block_token_mask(text)

                # Handle case where the exclude mask might not match span length due to tokenization differences
                span_length = span_activations.size(1)
                if len(exclude_mask) != span_length:
                    # Resize exclude_mask to match actual span length
                    if len(exclude_mask) > span_length:
                        exclude_mask = exclude_mask[:span_length]
                    else:
                        # Pad with False if exclude_mask is shorter
                        padding = torch.zeros(span_length - len(exclude_mask), dtype=torch.bool)
                        exclude_mask = torch.cat([exclude_mask, padding])

                # Create include mask (invert of exclude mask)
                include_mask = ~exclude_mask

                # Compute mean across non-code tokens only
                if include_mask.any():
                    # Select only non-code tokens
                    included_activations = span_activations[:, include_mask, :]  # (num_layers, included_tokens, hidden_size)

                    if included_activations.size(1) == 1:
                        # Single token - avoid mean computation
                        mean_activation = included_activations.squeeze(1)  # (num_layers, hidden_size)
                    else:
                        # Multi-token span - compute mean on GPU for non-code tokens only
                        mean_activation = included_activations.mean(dim=1)  # (num_layers, hidden_size)
                    turn_activations.append(mean_activation)
                else:
                    # All tokens are code blocks - skip this turn
                    continue

            if turn_activations:
                # Stack to get (num_turns, num_layers, hidden_size)
                conversation_activations[conv_id] = torch.stack(turn_activations)
            else:
                # No valid activations for this conversation - maintain dtype and device consistency
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)

        return conversation_activations

    def mean_all_turn_activations(
        self,
        probing_model,
        encoder: 'ConversationEncoder',
        conversation: List[Dict[str, str]],
        layer: int = 15,
        chat_format: bool = True,
        **chat_kwargs,
    ) -> torch.Tensor:
        """
        Get mean activations for all turns in a conversation using build_turn_spans and extract_full_activations.

        Args:
            probing_model: ProbingModel instance
            encoder: ConversationEncoder instance
            conversation: List of dict with 'role' and 'content' keys
            layer: Layer index to extract activations from (default 15)
            **chat_kwargs: additional arguments for apply_chat_template

        Returns:
            torch.Tensor: Mean activations of shape (num_turns, hidden_size) for all turns in chronological order
        """
        # Get turn spans for the conversation
        full_ids, spans = encoder.build_turn_spans(conversation, **chat_kwargs)

        # Import ActivationExtractor here to avoid circular import
        from .activations import ActivationExtractor
        extractor = ActivationExtractor(probing_model, encoder)

        # Extract full activations for the conversation
        activations = extractor.full_conversation(
            conversation, layer=layer, chat_format=chat_format, **chat_kwargs
        )

        # Handle the case where extract_full_activations returns multi-layer format
        if activations.ndim == 3:  # (num_layers, num_tokens, hidden_size)
            activations = activations[0]  # Take the first (and only) layer

        # Compute mean activation for each turn
        turn_mean_activations = []

        for span in spans:
            start_idx = span['start']
            end_idx = span['end']

            # Extract activations for this turn's tokens
            if start_idx < end_idx and end_idx <= activations.shape[0]:
                turn_activations = activations[start_idx:end_idx, :]  # (turn_tokens, hidden_size)
                mean_activation = turn_activations.mean(dim=0)  # (hidden_size,)
                turn_mean_activations.append(mean_activation)

        if not turn_mean_activations:
            # Return empty tensor with correct shape if no valid turns
            return torch.empty(0, activations.shape[1] if activations.ndim > 1 else 0)

        # Stack to get (num_turns, hidden_size)
        return torch.stack(turn_mean_activations)
