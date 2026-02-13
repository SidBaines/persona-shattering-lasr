"""
Utility functions for projecting persona vectors into the Assistant Axis landscape.

This module provides functions to:
- Load pre-computed role vectors, default assistant, and assistant axis from HuggingFace
- Fit PCA on role vectors to create the landscape
- Project arbitrary vectors into the PCA space
- Visualize vectors in 2D and 3D interactive plots
- Compute quantitative metrics (nearest neighbors, alignment)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from huggingface_hub import snapshot_download


class MeanScaler:
    """Mean centering scaler (following assistant-axis implementation)."""

    def __init__(self):
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def load_vectors_from_hf(
    model_name: str,
    repo_id: str = "lu-christina/assistant-axis-vectors"
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, str]:
    """Load role vectors, default vector, and assistant axis from HuggingFace.

    Args:
        model_name: Model identifier (e.g., "gemma-2-27b", "llama-3.3-70b", "qwen-3-32b")
        repo_id: HuggingFace dataset repository

    Returns:
        Tuple of (role_vectors_dict, default_vector, assistant_axis, local_dir)
        - role_vectors_dict: {role_name: tensor of shape [n_layers, hidden_size]}
        - default_vector: tensor of shape [n_layers, hidden_size]
        - assistant_axis: tensor of shape [n_layers, hidden_size]
        - local_dir: path to downloaded data
    """
    print(f"Downloading vectors from {repo_id}...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[
            f"{model_name}/role_vectors/*.pt",
            f"{model_name}/default_vector.pt",
            f"{model_name}/assistant_axis.pt",
        ]
    )

    # Load role vectors
    role_vectors_dir = Path(local_dir) / model_name / "role_vectors"
    role_vectors = {}
    for pt_file in sorted(role_vectors_dir.glob("*.pt")):
        role_name = pt_file.stem
        role_vectors[role_name] = torch.load(pt_file, map_location="cpu", weights_only=False)

    # Load default vector
    default_vector_path = Path(local_dir) / model_name / "default_vector.pt"
    default_vector = torch.load(default_vector_path, map_location="cpu", weights_only=False)

    # Load assistant axis
    assistant_axis_path = Path(local_dir) / model_name / "assistant_axis.pt"
    assistant_axis = torch.load(assistant_axis_path, map_location="cpu", weights_only=False)

    print(f"✓ Loaded {len(role_vectors)} role vectors")
    print(f"  Default vector shape: {default_vector.shape}")
    print(f"  Assistant axis shape: {assistant_axis.shape}")

    return role_vectors, default_vector, assistant_axis, local_dir


def fit_pca_landscape(
    role_vectors: Dict[str, torch.Tensor],
    target_layer: int,
    verbose: bool = True
) -> Tuple[PCA, MeanScaler, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Fit PCA on role vectors at target layer.

    Args:
        role_vectors: Dict of role vectors
        target_layer: Layer index to use
        verbose: Whether to print diagnostics

    Returns:
        Tuple of (pca, scaler, pca_result, role_labels, variance_explained, role_vectors_scaled)
        - pca: Fitted PCA object
        - scaler: Fitted MeanScaler
        - pca_result: PCA-transformed role vectors (N x K)
        - role_labels: List of role names
        - variance_explained: Variance explained by each PC
        - role_vectors_scaled: Mean-centered role vectors in original space
    """
    # Extract vectors at target layer
    role_labels = list(role_vectors.keys())
    role_vectors_array = torch.stack(
        [role_vectors[label][target_layer] for label in role_labels]
    ).float().numpy()

    if verbose:
        print(f"Fitting PCA on {len(role_labels)} vectors at layer {target_layer}...")
        print(f"  Vector shape: {role_vectors_array.shape}")

    # Fit scaler and PCA
    scaler = MeanScaler()
    role_vectors_scaled = scaler.fit_transform(role_vectors_array)

    pca = PCA()
    pca_result = pca.fit_transform(role_vectors_scaled)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)

    if verbose:
        print(f"✓ PCA fitted with {len(variance_explained)} components")
        print(f"  First 3 PCs: {variance_explained[0]:.1%}, {variance_explained[1]:.1%}, {variance_explained[2]:.1%}")
        print(f"  Cumulative (first 3): {cumulative_variance[2]:.1%}")

    return pca, scaler, pca_result, role_labels, variance_explained, role_vectors_scaled


def project_vector(
    vector: torch.Tensor,
    scaler: MeanScaler,
    pca: PCA,
    is_difference_vector: bool = False
) -> np.ndarray:
    """Project a vector into PC space.

    Args:
        vector: 1D tensor or numpy array
        scaler: Fitted MeanScaler
        pca: Fitted PCA
        is_difference_vector: If True, don't center (for axis vectors)

    Returns:
        PC coordinates (1D array)
    """
    if isinstance(vector, torch.Tensor):
        vector = vector.float().numpy()

    if is_difference_vector:
        # Don't center - the vector is already a difference
        vector_pc = pca.transform(vector.reshape(1, -1))[0]
    else:
        # Center by subtracting the mean of roles
        vector_scaled = scaler.transform(vector.reshape(1, -1))
        vector_pc = pca.transform(vector_scaled)[0]

    return vector_pc


def compute_nearest_neighbors(
    query_pc: np.ndarray,
    pca_result: np.ndarray,
    role_labels: List[str],
    n_dims: int = 3,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Compute nearest role vectors to a query vector in PC space.

    Args:
        query_pc: PC coordinates of query vector
        pca_result: PC coordinates of all role vectors
        role_labels: List of role names
        n_dims: Number of PC dimensions to use
        top_k: Number of nearest neighbors to return

    Returns:
        List of (role_name, distance) tuples, sorted by distance
    """
    distances = []
    for role_label, role_pc in zip(role_labels, pca_result):
        dist = np.linalg.norm(query_pc[:n_dims] - role_pc[:n_dims])
        distances.append((role_label, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:top_k]


def compute_alignment_metrics(
    vector_pc: np.ndarray,
    default_pc: np.ndarray,
    assistant_axis_pc: np.ndarray,
    n_dims: int = 3
) -> Dict[str, float]:
    """Compute alignment metrics for a vector.

    Args:
        vector_pc: PC coordinates of the vector
        default_pc: PC coordinates of default assistant
        assistant_axis_pc: PC coordinates of assistant axis
        n_dims: Number of PC dimensions to use

    Returns:
        Dictionary of metrics:
        - distance_to_default: Euclidean distance to default assistant
        - cosine_to_axis: Cosine similarity to assistant axis direction
        - projection_on_axis: Scalar projection onto assistant axis
        - perpendicular_dist: Perpendicular distance from axis line
    """
    # Normalize for cosine similarity
    axis_direction = assistant_axis_pc[:n_dims] / np.linalg.norm(assistant_axis_pc[:n_dims])
    vector_direction = vector_pc[:n_dims] / np.linalg.norm(vector_pc[:n_dims])

    # Cosine similarity
    cosine_to_axis = np.dot(axis_direction, vector_direction)

    # Distance to default
    distance_to_default = np.linalg.norm(vector_pc[:n_dims] - default_pc[:n_dims])

    # Projection onto axis
    projection_on_axis = np.dot(vector_pc[:n_dims], axis_direction)

    # Perpendicular distance from axis line
    projection_point = projection_on_axis * axis_direction
    perpendicular_dist = np.linalg.norm(vector_pc[:n_dims] - projection_point)

    return {
        'distance_to_default': distance_to_default,
        'cosine_to_axis': cosine_to_axis,
        'projection_on_axis': projection_on_axis,
        'perpendicular_dist': perpendicular_dist,
    }


def plot_2d_landscape(
    pca_result: np.ndarray,
    role_labels: List[str],
    variance_explained: np.ndarray,
    default_pc: np.ndarray,
    assistant_axis_pc: np.ndarray,
    role_vectors_centered: np.ndarray,
    assistant_axis_orig: np.ndarray,
    test_vectors: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Assistant Axis Landscape (2D)",
    show_axis_line: bool = True
) -> go.Figure:
    """Create interactive 2D scatter plot with axis line.

    Args:
        pca_result: PCA-transformed role vectors (N x K)
        role_labels: List of role names
        variance_explained: Variance explained by each PC
        default_pc: Default vector in PC space
        assistant_axis_pc: Assistant axis in PC space
        role_vectors_centered: Mean-centered role vectors in original space
        assistant_axis_orig: Assistant axis in original space
        test_vectors: Optional dict of {name: pc_coordinates} for custom vectors
        title: Plot title
        show_axis_line: Whether to show the axis as an extended line

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Compute projections onto assistant axis (in original space) for coloring
    projections = role_vectors_centered @ assistant_axis_orig / np.linalg.norm(assistant_axis_orig)

    # Plot role vectors, colored by projection
    fig.add_trace(go.Scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        mode='markers',
        marker=dict(
            size=6,
            color=projections,
            colorscale='RdBu_r',
            cmin=projections.min(),
            cmax=projections.max(),
            colorbar=dict(
                title="Projection onto<br>Assistant Axis",
                orientation='h',
                x=0.5,
                xanchor='center',
                y=-0.15,
                yanchor='top',
                len=0.6,
                thickness=15
            ),
            opacity=0.7
        ),
        text=role_labels,
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Projection: %{marker.color:.2f}<extra></extra>',
        name='Role vectors'
    ))

    # Plot default assistant
    fig.add_trace(go.Scatter(
        x=[default_pc[0]],
        y=[default_pc[1]],
        mode='markers',
        marker=dict(size=20, color='gold', symbol='star', line=dict(color='black', width=2)),
        text=['Default Assistant'],
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
        name='Default Assistant'
    ))

    # Plot assistant axis as extended line
    if show_axis_line:
        axis_direction = assistant_axis_pc[:2] / np.linalg.norm(assistant_axis_pc[:2])
        max_extent = max(np.abs(pca_result[:, :2]).max(), np.abs(default_pc[:2]).max()) * 1.5

        line_start = -max_extent * axis_direction
        line_end = max_extent * axis_direction

        fig.add_trace(go.Scatter(
            x=[line_start[0], line_end[0]],
            y=[line_start[1], line_end[1]],
            mode='lines',
            line=dict(color='red', width=3, dash='solid'),
            name='Assistant Axis',
            hovertemplate='Assistant Axis Line<extra></extra>'
        ))

        # Add arrow at the positive end
        arrow_start = 0.8 * max_extent * axis_direction
        arrow_end = max_extent * axis_direction

        fig.add_trace(go.Scatter(
            x=[arrow_start[0], arrow_end[0]],
            y=[arrow_start[1], arrow_end[1]],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=[0, 15], color='red', symbol='arrow-up'),
            name='Axis Direction',
            hovertemplate='Toward Default<extra></extra>',
            showlegend=False
        ))

    # Plot origin
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(size=10, color='black', symbol='circle'),
        text=['Origin'],
        hovertemplate='<b>Origin (mean of roles)</b><extra></extra>',
        name='Origin'
    ))

    # Plot test vectors if provided
    if test_vectors:
        colors = ['cyan', 'magenta', 'orange', 'lime', 'pink']
        for i, (name, pc_coords) in enumerate(test_vectors.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=[pc_coords[0]],
                y=[pc_coords[1]],
                mode='markers',
                marker=dict(size=15, color=color, symbol='x', line=dict(color='black', width=2)),
                text=[name],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
                name=name
            ))

    fig.update_layout(
        title=title,
        xaxis_title=f'PC1 ({variance_explained[0]:.1%} variance)',
        yaxis_title=f'PC2 ({variance_explained[1]:.1%} variance)',
        hovermode='closest',
        width=900,
        height=750,
        template='plotly_white',
        margin=dict(b=100)
    )

    return fig


def plot_3d_landscape(
    pca_result: np.ndarray,
    role_labels: List[str],
    variance_explained: np.ndarray,
    default_pc: np.ndarray,
    assistant_axis_pc: np.ndarray,
    role_vectors_centered: np.ndarray,
    assistant_axis_orig: np.ndarray,
    test_vectors: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Assistant Axis Landscape (3D)",
    show_axis_line: bool = True
) -> go.Figure:
    """Create interactive 3D scatter plot with axis line.

    Args:
        pca_result: PCA-transformed role vectors (N x K)
        role_labels: List of role names
        variance_explained: Variance explained by each PC
        default_pc: Default vector in PC space
        assistant_axis_pc: Assistant axis in PC space
        role_vectors_centered: Mean-centered role vectors in original space
        assistant_axis_orig: Assistant axis in original space
        test_vectors: Optional dict of {name: pc_coordinates} for custom vectors
        title: Plot title
        show_axis_line: Whether to show the axis as an extended line

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Compute projections onto assistant axis (in original space) for coloring
    projections = role_vectors_centered @ assistant_axis_orig / np.linalg.norm(assistant_axis_orig)

    # Plot role vectors
    fig.add_trace(go.Scatter3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=projections,
            colorscale='RdBu_r',
            cmin=projections.min(),
            cmax=projections.max(),
            colorbar=dict(
                title="Projection onto<br>Assistant Axis",
                orientation='h',
                x=0.5,
                xanchor='center',
                y=-0.1,
                yanchor='top',
                len=0.6,
                thickness=15
            ),
            opacity=0.7
        ),
        text=role_labels,
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br>Projection: %{marker.color:.2f}<extra></extra>',
        name='Role vectors'
    ))

    # Plot default assistant
    fig.add_trace(go.Scatter3d(
        x=[default_pc[0]],
        y=[default_pc[1]],
        z=[default_pc[2]],
        mode='markers',
        marker=dict(size=12, color='gold', symbol='diamond', line=dict(color='black', width=2)),
        text=['Default Assistant'],
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>',
        name='Default Assistant'
    ))

    # Plot assistant axis as extended line
    if show_axis_line:
        axis_direction = assistant_axis_pc[:3] / np.linalg.norm(assistant_axis_pc[:3])
        max_extent = max(np.abs(pca_result[:, :3]).max(), np.abs(default_pc[:3]).max()) * 1.5

        line_start = -max_extent * axis_direction
        line_end = max_extent * axis_direction

        fig.add_trace(go.Scatter3d(
            x=[line_start[0], line_end[0]],
            y=[line_start[1], line_end[1]],
            z=[line_start[2], line_end[2]],
            mode='lines',
            line=dict(color='red', width=8),
            name='Assistant Axis',
            hovertemplate='Assistant Axis Line<extra></extra>'
        ))

        # Add marker at positive end
        fig.add_trace(go.Scatter3d(
            x=[line_end[0]],
            y=[line_end[1]],
            z=[line_end[2]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='Axis (+)',
            hovertemplate='Toward Default<extra></extra>'
        ))

    # Plot origin
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(size=8, color='black', symbol='circle'),
        text=['Origin'],
        hovertemplate='<b>Origin (mean of roles)</b><extra></extra>',
        name='Origin'
    ))

    # Plot test vectors
    if test_vectors:
        colors = ['cyan', 'magenta', 'orange', 'lime', 'pink']
        for i, (name, pc_coords) in enumerate(test_vectors.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=[pc_coords[0]],
                y=[pc_coords[1]],
                z=[pc_coords[2]],
                mode='markers',
                marker=dict(size=10, color=color, symbol='x'),
                text=[name],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>',
                name=name
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f'PC1 ({variance_explained[0]:.1%})',
            yaxis_title=f'PC2 ({variance_explained[1]:.1%})',
            zaxis_title=f'PC3 ({variance_explained[2]:.1%})'
        ),
        width=900,
        height=750,
        template='plotly_white',
        margin=dict(b=100)
    )

    return fig


def load_custom_vector(
    vector_path: str,
    target_layer: int
) -> Tuple[torch.Tensor, Dict]:
    """Load a custom persona vector from disk.

    Args:
        vector_path: Path to .pt file containing the vector
        target_layer: Layer index to extract

    Returns:
        Tuple of (vector_at_layer, metadata)
    """
    vector_data = torch.load(vector_path, map_location="cpu", weights_only=False)

    # Extract metadata
    metadata = {
        'checkpoint': vector_data.get('checkpoint', Path(vector_path).stem),
        'n_samples': vector_data.get('n_samples', 'N/A'),
        'n_layers': vector_data.get('n_layers', 'N/A'),
        'hidden_dim': vector_data.get('hidden_dim', 'N/A'),
    }

    vector_at_layer = vector_data['vector'][target_layer]

    return vector_at_layer, metadata


def analyze_vector(
    vector_name: str,
    vector_pc: np.ndarray,
    default_pc: np.ndarray,
    assistant_axis_pc: np.ndarray,
    pca_result: np.ndarray,
    role_labels: List[str],
    n_dims: int = 3,
    top_k: int = 5,
    verbose: bool = True
) -> Dict:
    """Analyze a vector in the landscape.

    Args:
        vector_name: Name of the vector
        vector_pc: PC coordinates of the vector
        default_pc: PC coordinates of default assistant
        assistant_axis_pc: PC coordinates of assistant axis
        pca_result: PC coordinates of all role vectors
        role_labels: List of role names
        n_dims: Number of PC dimensions to use
        top_k: Number of nearest neighbors to return
        verbose: Whether to print analysis

    Returns:
        Dictionary containing metrics and nearest neighbors
    """
    # Compute metrics
    metrics = compute_alignment_metrics(
        vector_pc, default_pc, assistant_axis_pc, n_dims=n_dims
    )

    # Find nearest neighbors
    neighbors = compute_nearest_neighbors(
        vector_pc, pca_result, role_labels, n_dims=n_dims, top_k=top_k
    )

    if verbose:
        print(f"\nAnalysis for: {vector_name}")
        print(f"  PC coordinates (first {n_dims}): {vector_pc[:n_dims]}")
        print(f"\nAlignment metrics:")
        print(f"  Distance to default: {metrics['distance_to_default']:.2f}")
        print(f"  Cosine to axis: {metrics['cosine_to_axis']:.4f}")
        print(f"  Projection on axis: {metrics['projection_on_axis']:.2f}")
        print(f"  Perpendicular distance: {metrics['perpendicular_dist']:.2f}")
        print(f"\nNearest role vectors (top {top_k}):")
        for role, dist in neighbors:
            print(f"  {role}: {dist:.2f}")

    return {
        'metrics': metrics,
        'neighbors': neighbors,
        'pc_coords': vector_pc[:n_dims]
    }
