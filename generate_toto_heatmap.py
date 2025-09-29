import torch
from sklearn.decomposition import PCA
import seaborn as sns
from src.util import impute_with_window, create_sliding_windows, get_toto_activations
from toto.model.toto import Toto
from src.synth_util import generate_single_series
from toto.data.util.dataset import MaskedTimeseries
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import pandas as pd


def calculate_layer_self_similarity(
    full_activations_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the cosine similarity matrix between all layers of a single
    activation tensor.

    Args:
        full_activations_tensor (torch.Tensor): The raw activations for a single
            input window, of shape [layers, bsz, n_vars, seq_len, emb_dim].

    Returns:
        torch.Tensor: A 2D tensor of shape [layers, layers] containing the
                      pairwise cosine similarities.
    """
    layer_summaries = full_activations_tensor.squeeze(1).squeeze(1)
    layer_vectors = layer_summaries.mean(dim=1)

    layer_vectors_normalized = F.normalize(layer_vectors, p=2, dim=1)
    similarity_matrix = torch.matmul(layer_vectors_normalized, layer_vectors_normalized.T)
    return similarity_matrix

def plot_similarity_heatmap(
    similarity_matrix: torch.Tensor,
    title: str,
    ax: plt.Axes
):
    """
    Generates a visually appealing heatmap on a given Matplotlib Axes object.
    """
    sns.heatmap(
        similarity_matrix.cpu().numpy(),
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=.5,
        cbar=True
    )
    ax.invert_yaxis()
    ax.set_title(title, fontsize=24, pad=15)
    ax.set_xlabel("Layer Index", fontsize=20)
    ax.set_ylabel("Layer Index", fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16, rotation=0)

def create_synthetic_similarity_matrix(num_layers: int = 12) -> torch.Tensor:
    """
    Generates a synthetic 12x12 cosine similarity matrix based on a predefined
    hierarchical structure, with added noise for realism.

    The structure is designed to show:
    - Layers 0-1: Low similarity / anti-correlated
    - Layers 2-5: Moderate similarity
    - Layers 6-11: High similarity

    Args:
        num_layers (int): The number of layers for the matrix (default 12).

    Returns:
        torch.Tensor: A synthetic similarity matrix of shape [num_layers, num_layers].
    """
    matrix = torch.zeros((num_layers, num_layers))

    regime_map = {
        'early_layers': {'range': (0, 2), 'target': -0.45, 'noise': 0.20},
        'mid_layers_low': {'range': (2, 6), 'target': 0.4, 'noise': 0.20},
        'late_layers': {'range': (6, 8), 'target': 0.6, 'noise': 0.15},
        'last_layers': {'range': (8, num_layers), 'target': 0.72, 'noise': 0.15}
    }

    # Populate the matrix block by block
    for i in range(num_layers):
        for j in range(num_layers):
            # Determine which regime the (i, j) pair falls into
            # We'll simplify and say the block is determined by the "earlier" layer index
            layer_index = min(i, j)
            
            target_val = 0.0
            noise_level = 0.0
            
            if regime_map['early_layers']['range'][0] <= layer_index < regime_map['early_layers']['range'][1]:
                target_val = regime_map['early_layers']['target']
                noise_level = regime_map['early_layers']['noise']
            elif regime_map['mid_layers_low']['range'][0] <= layer_index < regime_map['mid_layers_low']['range'][1]:
                target_val = regime_map['mid_layers_low']['target']
                noise_level = regime_map['mid_layers_low']['noise']
            elif regime_map['late_layers']['range'][0] <= layer_index < regime_map['late_layers']['range'][1]:
                target_val = regime_map['late_layers']['target']
                noise_level = regime_map['late_layers']['noise']    
            else: # Late layers
                target_val = regime_map['last_layers']['target']
                noise_level = regime_map['last_layers']['noise']

            matrix[i, j] = target_val + (torch.rand(1).item() - 0.5) * 2 * noise_level

    matrix = (matrix + matrix.T) / 2
    
    for i in range(2, num_layers):
        for j in range(2, num_layers):
            if i != j:
                decay = 0.01 * abs(i - j) # Similarity decays with distance from diagonal
                matrix[i, j] = max(0, matrix[i, j] - decay)

    matrix[0, 1] = matrix[1, 0] = -0.18 + (torch.rand(1).item() - 0.5) * 0.04 # -0.2 to -0.16
    matrix[3:6, 3:6] = torch.clamp(matrix[3:6, 3:6], 0.35, 0.45)
    matrix[6:, 6:] = torch.clamp(matrix[6:, 6:], 0.5, 0.7)

    return matrix

def plot_single_heatmap(
    similarity_matrix: torch.Tensor,
    xlabel: str,
    ylabel: str,
    save_path: str,
    title: str = None,
):
    """
    Generates and saves a single, high-quality, and visually appealing heatmap.
    """
    plt.figure(figsize=(14, 12))
    ax = plt.gca() # Get current axes

    annotation_kwargs = {"fontsize": 18}

    sns.heatmap(
        similarity_matrix.cpu().numpy(),
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=.5,
        cbar=True,
        vmin=-1, vmax=1,
        annot_kws = annotation_kwargs
    )
    
    ax.invert_yaxis()

    if title:
        ax.set_title(title, fontsize=28, pad=26)
    ax.set_xlabel(xlabel, fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28, rotation=0)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {save_path}")

def calculate_pca_based_layer_similarity(
    activations_regime_A: torch.Tensor,
    activations_regime_B: torch.Tensor,
    n_components: int = 100  # <-- The parameter is now explicit
) -> torch.Tensor:
    """
    Calculates cross-similarity after first projecting layer activations
    onto a specified number of principal components.

    Args:
        activations_regime_A (torch.Tensor): Full activations for regime A.
        activations_regime_B (torch.Tensor): Full activations for regime B.
        n_components (int): The number of principal components to keep.
                              This defines the dimensionality of the new space.

    Returns:
        torch.Tensor: The PCA-based cross-similarity matrix.
    """
    
    def reshape_for_pca(activations_tensor):
        return activations_tensor.squeeze(1).squeeze(1).reshape(-1, activations_tensor.shape[-1]).cpu().numpy()

    patch_vectors_A = reshape_for_pca(activations_regime_A)
    patch_vectors_B = reshape_for_pca(activations_regime_B)
    combined_patch_vectors = np.vstack([patch_vectors_A, patch_vectors_B])
    
    pca = PCA(n_components=n_components)
    pca.fit(combined_patch_vectors)
    
    def get_pooled_layer_vectors(activations_tensor):
        summary = activations_tensor.squeeze(1).squeeze(1)
        return summary.mean(dim=1).cpu().numpy()

    layer_vectors_A = get_pooled_layer_vectors(activations_regime_A)
    layer_vectors_B = get_pooled_layer_vectors(activations_regime_B)

    pca_vectors_A = pca.transform(layer_vectors_A)
    pca_vectors_B = pca.transform(layer_vectors_B)
    
    pca_vectors_A = torch.from_numpy(pca_vectors_A)
    pca_vectors_B = torch.from_numpy(pca_vectors_B)

    norm_A = F.normalize(pca_vectors_A, p=2, dim=1)
    norm_B = F.normalize(pca_vectors_B, p=2, dim=1)
    
    similarity_matrix = torch.matmul(norm_A, norm_B.T)
    
    return similarity_matrix
    
WINDOW_SIZE_NORMAL=128
WINDOW_SIZE_SYNTH=256
STRIDE=1
toto_model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0', cache_dir='cache_dir')
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
toto_model.to(device)
toto_model.eval() # Set to evaluation mode
toto_backbone = toto_model.model

turbulence_df = pd.read_csv('./2008_crash.csv')
turbulence_df = impute_with_window(turbulence_df, window_size=3)
turbulence_df['observation_date'] = pd.to_datetime(turbulence_df['observation_date'])
turbulence_df.set_index('observation_date', inplace=True)

normal_df = pd.read_csv('./2017_data.csv')
normal_df = impute_with_window(normal_df, window_size=3)
normal_df['observation_date'] = pd.to_datetime(normal_df['observation_date'])
normal_df.set_index('observation_date', inplace=True)

turb_2k_df = pd.read_csv('./2000_crash.csv')
turb_2k_df = impute_with_window(turb_2k_df, window_size=3)
turb_2k_df['observation_date'] = pd.to_datetime(turb_2k_df['observation_date'])
turb_2k_df.set_index('observation_date', inplace=True)

synthetic_crash_1 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=0.2, seed=3)['price'].to_numpy()
synthetic_crash_2 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=0.5, seed=3)['price'].to_numpy()
synthetic_crash_3 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=1.0, seed=3)['price'].to_numpy()
synthetic_crash_4 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=1.5, seed=3)['price'].to_numpy()
synthetic_crash_5 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=2.0, seed=3)['price'].to_numpy()
synthetic_normal = generate_single_series(T=320, regime="calm",  start_price=2000, seed=3)['price'].to_numpy()


START_2017_NORMAL = '2017-01-12'
END_2017_NORMAL = '2017-08-09'
START_2008_CRASH = '2008-07-25'
END_2008_CRASH = '2009-03-09'
START_2000_CRASH = '2000-08-31'
END_2000_CRASH = '2001-04-04'
 
 
normal_windows = create_sliding_windows(
    df=normal_df,
    start_date=START_2017_NORMAL,
    end_date=END_2017_NORMAL,
    window_size=WINDOW_SIZE_NORMAL,
    stride=STRIDE,
    column_name='NASDAQ100'
)
turbulence_windows = create_sliding_windows(
    df=turbulence_df,
    start_date=START_2008_CRASH,
    end_date=END_2008_CRASH,
    window_size=WINDOW_SIZE_NORMAL,
    stride=STRIDE,
    column_name='NASDAQ100'
)
turb_2k_windows = create_sliding_windows(
    df=turb_2k_df,
    start_date=START_2000_CRASH,
    end_date=END_2000_CRASH,
    window_size=WINDOW_SIZE_NORMAL,
    stride=STRIDE,
    column_name='NASDAQ100'
)
synthetic_crash_1_windows = create_sliding_windows(
    series=synthetic_crash_1,
    window_size=WINDOW_SIZE_SYNTH,
    stride=STRIDE,
)
synthetic_crash_2_windows = create_sliding_windows(
    series=synthetic_crash_2,
    window_size=WINDOW_SIZE_SYNTH,
    stride=STRIDE,
)
synthetic_crash_3_windows = create_sliding_windows(
    series=synthetic_crash_3,
    window_size=WINDOW_SIZE_SYNTH,
    stride=STRIDE,
)
synthetic_crash_4_windows = create_sliding_windows(
    series=synthetic_crash_4,
    window_size=WINDOW_SIZE_SYNTH,
    stride=STRIDE,
)
synthetic_crash_5_windows = create_sliding_windows(
    series=synthetic_crash_5,
    window_size=WINDOW_SIZE_SYNTH,
    stride=STRIDE,
)
synthetic_normal_windows = create_sliding_windows(
    series=synthetic_normal,
    window_size=WINDOW_SIZE_SYNTH,
    stride=STRIDE,
)



normal_window = normal_windows[0]
normal_window = normal_window.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
timestamp_seconds = torch.zeros_like(torch.tensor(normal_window)).to(device)
time_interval_seconds = torch.full((1, 1), 60*15).to(device)
normal_2017_inputs = MaskedTimeseries(
    series=normal_window,
    padding_mask=torch.full_like(normal_window, True, dtype=torch.bool),
    id_mask=torch.zeros_like(normal_window),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)

turbulence_window = turbulence_windows[0]
turbulence_window = turbulence_window.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
timestamp_seconds = torch.zeros_like(torch.tensor(turbulence_window)).to(device)
time_interval_seconds = torch.full((1, 1), 60*15).to(device)
turb_2008_inputs = MaskedTimeseries(
    series=turbulence_window,
    padding_mask=torch.full_like(turbulence_window, True, dtype=torch.bool),
    id_mask=torch.zeros_like(turbulence_window),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)
turb_2k_window = turb_2k_windows[0]
turb_2k_window = turb_2k_window.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
timestamp_seconds = torch.zeros_like(torch.tensor(turb_2k_window)).to(device)
time_interval_seconds = torch.full((1, 1), 60*15).to(device)
turb_2000_inputs = MaskedTimeseries(
    series=turb_2k_window,
    padding_mask=torch.full_like(turb_2k_window, True, dtype=torch.bool),
    id_mask=torch.zeros_like(turb_2k_window),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)
synthetic_normal_window = synthetic_normal_windows[0]
synthetic_normal_window = synthetic_normal_window.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
timestamp_seconds = torch.zeros_like(torch.tensor(synthetic_normal_window)).to(device)
time_interval_seconds = torch.full((1, 1), 60*15).to(device)
synthetic_normal_inputs = MaskedTimeseries(
    series=synthetic_normal_window,
    padding_mask=torch.full_like(synthetic_normal_window, True, dtype=torch.bool),
    id_mask=torch.zeros_like(synthetic_normal_window),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)

synthetic_crash_1_window = synthetic_crash_1_windows[0]
synthetic_crash_1_window = synthetic_crash_1_window.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
timestamp_seconds = torch.zeros_like(torch.tensor(synthetic_crash_1_window)).to(device)
time_interval_seconds = torch.full((1, 1), 60*15).to(device)
synthetic_crash_1_inputs = MaskedTimeseries(
    series=synthetic_crash_1_window,
    padding_mask=torch.full_like(synthetic_crash_1_window, True, dtype=torch.bool),
    id_mask=torch.zeros_like(synthetic_crash_1_window),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)
synthetic_crash_2_window = synthetic_crash_2_windows[0]
synthetic_crash_2_window = synthetic_crash_2_window.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
timestamp_seconds = torch.zeros_like(torch.tensor(synthetic_crash_2_window)).to(device)
time_interval_seconds = torch.full((1, 1), 60*15).to(device)
synthetic_crash_2_inputs = MaskedTimeseries(
    series=synthetic_crash_2_window,
    padding_mask=torch.full_like(synthetic_crash_2_window, True, dtype=torch.bool),
    id_mask=torch.zeros_like(synthetic_crash_2_window),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)
synthetic_crash_3_window = synthetic_crash_3_windows[0]
synthetic_crash_3_window = synthetic_crash_3_window.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
timestamp_seconds = torch.zeros_like(torch.tensor(synthetic_crash_3_window)).to(device)
time_interval_seconds = torch.full((1, 1), 60*15).to(device)
synthetic_crash_3_inputs = MaskedTimeseries(
    series=synthetic_crash_3_window,
    padding_mask=torch.full_like(synthetic_crash_3_window, True, dtype=torch.bool),
    id_mask=torch.zeros_like(synthetic_crash_3_window),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)
synthetic_crash_4_window = synthetic_crash_4_windows[0]
synthetic_crash_4_window = synthetic_crash_4_window.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
timestamp_seconds = torch.zeros_like(torch.tensor(synthetic_crash_4_window)).to(device)
time_interval_seconds = torch.full((1, 1), 60*15).to(device)
synthetic_crash_4_inputs = MaskedTimeseries(
    series=synthetic_crash_4_window,
    padding_mask=torch.full_like(synthetic_crash_4_window, True, dtype=torch.bool),
    id_mask=torch.zeros_like(synthetic_crash_4_window),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)
synthetic_crash_5_window = synthetic_crash_5_windows[0]
synthetic_crash_5_window = synthetic_crash_5_window.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
timestamp_seconds = torch.zeros_like(torch.tensor(synthetic_crash_5_window)).to(device)
time_interval_seconds = torch.full((1, 1), 60*15).to(device)
synthetic_crash_5_inputs = MaskedTimeseries(
    series=synthetic_crash_5_window,
    padding_mask=torch.full_like(synthetic_crash_5_window, True, dtype=torch.bool),
    id_mask=torch.zeros_like(synthetic_crash_5_window),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)

act_08_crash_dict = {
    'inputs': turb_2008_inputs.series,
    'padding_mask': turb_2008_inputs.padding_mask,
    'id_mask': turb_2008_inputs.id_mask
}
act_17_calm_dict = {
    'inputs': normal_2017_inputs.series,
    'padding_mask': normal_2017_inputs.padding_mask,
    'id_mask': normal_2017_inputs.id_mask
}
act_00_crash_dict = {
    'inputs': turb_2000_inputs.series,
    'padding_mask': turb_2000_inputs.padding_mask,
    'id_mask': turb_2000_inputs.id_mask
}
act_s_normal_dict={
    'inputs': synthetic_normal_inputs.series,
    'padding_mask': synthetic_normal_inputs.padding_mask,
    'id_mask': synthetic_normal_inputs.id_mask
}
act_s1_dict={
    'inputs': synthetic_crash_1_inputs.series,
    'padding_mask': synthetic_crash_1_inputs.padding_mask,
    'id_mask': synthetic_crash_1_inputs.id_mask
}
act_s2_dict={
    'inputs': synthetic_crash_2_inputs.series,
    'padding_mask': synthetic_crash_2_inputs.padding_mask,
    'id_mask': synthetic_crash_2_inputs.id_mask
}
act_s3_dict={
    'inputs': synthetic_crash_3_inputs.series,
    'padding_mask': synthetic_crash_3_inputs.padding_mask,
    'id_mask': synthetic_crash_3_inputs.id_mask
}
act_s4_dict={
    'inputs': synthetic_crash_4_inputs.series,
    'padding_mask': synthetic_crash_4_inputs.padding_mask,
    'id_mask': synthetic_crash_4_inputs.id_mask
}
act_s5_dict={
    'inputs': synthetic_crash_5_inputs.series,
    'padding_mask': synthetic_crash_5_inputs.padding_mask,
    'id_mask': synthetic_crash_5_inputs.id_mask
}



_, act_08_crash = get_toto_activations(toto_backbone, [act_08_crash_dict])
_, act_00_crash = get_toto_activations(toto_backbone, [act_00_crash_dict])
_, act_17_calm = get_toto_activations(toto_backbone, [act_17_calm_dict])
_, act_s_normal_crash = get_toto_activations(toto_backbone, [act_s_normal_dict])
_, act_s1_crash = get_toto_activations(toto_backbone, [act_s1_dict])
_, act_s2_crash = get_toto_activations(toto_backbone, [act_s2_dict])
_, act_s3_crash = get_toto_activations(toto_backbone, [act_s3_dict])
_, act_s4_crash = get_toto_activations(toto_backbone, [act_s4_dict])
_, act_s5_crash = get_toto_activations(toto_backbone, [act_s5_dict])

calm_vs_crash_2008_sim = calculate_pca_based_layer_similarity(
    act_17_calm,
    act_08_crash,
    n_components=40
)

plot_single_heatmap(
    similarity_matrix=calm_vs_crash_2008_sim,
    xlabel="Layer Index: Calm Regime (2017)",
    ylabel="Layer Index: Crash Regime (2008)",
    save_path="heatmap.png"
)