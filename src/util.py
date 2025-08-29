import torch
import numpy as np
import functools
from einops import rearrange, repeat
from typing import NamedTuple, Optional

from datasets import load_dataset 
import sys 
import contextlib
import functools
from transformers import AutoModelForCausalLM
from toto.model.toto import Toto
from toto.model.backbone import TotoBackbone, TotoOutput
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster, Forecast
from toto.model.attention import (
    AttentionAxis,
    MultiHeadAttention,
    SpaceWiseMultiheadAttention,
    TimeWiseMultiheadAttention,
)
from toto.data.util.dataset import pad_array, pad_id_mask
from typing import Callable, List, Tuple
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pandas as pd
from jaxtyping import Float, Bool
from toto.model.util import KVCache

# toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
# toto.to('cuda:0')
@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
) -> None:
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()
            
def prepare_toto_inputs(data_samples):
    """Prepare inputs for Toto model"""
    inputs_list = []
    padding_masks_list = []
    id_masks_list = []
    
    for sample in data_samples:
        # Assuming sample is already in the right format
        # Convert to tensors and add batch dimension if needed
        inputs = torch.tensor(sample['inputs']).float()
        padding_mask = torch.tensor(sample['padding_mask']).bool()
        id_mask = torch.tensor(sample['id_mask']).float()
        
        # Ensure correct shape: (batch, variate, time_steps)
        if inputs.dim() == 2:  # (variate, time_steps)
            inputs = inputs.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
            id_mask = id_mask.unsqueeze(0)
            
        inputs_list.append(inputs)
        padding_masks_list.append(padding_mask)
        id_masks_list.append(id_mask)
    
    # Stack all samples
    inputs_batch = torch.cat(inputs_list, dim=0)
    padding_masks_batch = torch.cat(padding_masks_list, dim=0)
    id_masks_batch = torch.cat(id_masks_list, dim=0)
    
    return inputs_batch, padding_masks_batch, id_masks_batch

def get_toto_activations_pre_hook(
    layer_idx: int,
    cache_full: List[List[Tensor]],
    positions: List[int] = None,  # Not used for Toto
    whole_seq: bool = False
) -> Callable:
    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...]) -> None:
        # For Toto, input[0] should be the input to the transformer layer
        activation = input[1]  # Shape: (batch, variate, seq_len, embed_dim)
        
        if whole_seq:
            cache_full[layer_idx].append(activation.clone().detach().cpu())
        else:
            # Extract specific positions or last few tokens if needed
            cache_full[layer_idx].append(activation.clone().detach().cpu())
    return hook_fn

def get_toto_activations_fwd_hook(
    layer_idx: int,
    cache_full: List[List[Tensor]],
    positions: List[int] = None,
    whole_seq: bool = False
) -> Callable:
    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...], output: Tuple[Tensor, ...]) -> None:
        # For Toto, output[0] should be the output from the transformer layer
        activation = output[0]  # Shape: (batch, variate, seq_len, embed_dim)
        
        if whole_seq:
            cache_full[layer_idx].append(activation.clone().detach().cpu())
        else:
            # Extract specific positions or last few tokens if needed
            cache_full[layer_idx].append(activation.clone().detach().cpu())
    return hook_fn

def get_toto_activations(
    model: TotoBackbone,
    data_samples: List[dict],  # Your timeseries data
    batch_size: int = 32,
    whole_seq: bool = False
) -> Tuple[Tensor, Tensor]:
    torch.cuda.empty_cache()
    
    n_layers = len(model.transformer.layers)
    full_activations = [[] for _ in range(n_layers)]
    
    # Register hooks on each transformer layer
    fwd_pre_hooks = [
        (model.transformer.layers[layer_idx], get_toto_activations_pre_hook(
            layer_idx=layer_idx,
            cache_full=full_activations,
            whole_seq=whole_seq
        )) for layer_idx in range(n_layers)
    ]
    
    # Process data in batches
    for i in range(0, len(data_samples), batch_size):
        batch_samples = data_samples[i:i+batch_size]
        inputs_batch, padding_masks_batch, id_masks_batch = prepare_toto_inputs(batch_samples)
        
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            # Forward pass through Toto
            with torch.no_grad(): 
                model(inputs_batch.to(model.device), 
                      padding_masks_batch.to(model.device), 
                      id_masks_batch.to(model.device))
    
    # Process collected activations
    flat_list = [torch.cat(inner_list, dim=0) for inner_list in full_activations if inner_list]
    if flat_list:
        result = torch.stack(flat_list)  # Shape: (layers, batch, variate, seq_len, embed_dim)
        mean_activations = result.mean(dim=1)  # Average over batch dimension
        return mean_activations, result
    else:
        return None, None

def calculate_layerwise_similarity(
    activations_event1: torch.Tensor,
    activations_event2: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the cosine similarity between two activation tensors layer by layer.

    This function is the most rigorous way to compare the model's internal state
    across two different events. It first creates a single summary vector for
    each layer by mean-pooling the patch activations, and then compares these
    layer summaries.

    Args:
        activations_event1 (torch.Tensor): The full activation tensor for the first event.
            Shape: [layers, bsz, n_vars, seq_len, emb_dim]
        activations_event2 (torch.Tensor): The full activation tensor for the second event.
            Must have the same shape as the first.

    Returns:
        torch.Tensor: A 1D tensor of shape [layers] where each element is the
                      cosine similarity for the corresponding layer.
    """
    # --- 1. Sanity Check ---
    assert activations_event1.shape == activations_event2.shape, \
        "Activation tensors must have the exact same shape."

    # --- 2. Pre-processing: Create Layer Summaries ---
    # We need to distill the [layers, bsz, n_vars, seq_len, emb_dim] tensor
    # into a clean [layers, emb_dim] tensor for comparison.

    # a. Squeeze out the singleton batch and variate dimensions.
    # Shape becomes: [layers, seq_len, emb_dim]
    summary1 = activations_event1.squeeze(1).squeeze(1)
    summary2 = activations_event2.squeeze(1).squeeze(1)

    # b. Mean-pool across the sequence dimension (dim=1) to get a single
    #    holistic representation for each layer.
    # Shape becomes: [layers, emb_dim]
    summary1_pooled = summary1.mean(dim=1)
    summary2_pooled = summary2.mean(dim=1)

    # --- 3. Calculate Layer-wise Cosine Similarity ---
    # F.cosine_similarity is perfect for this. We compute the similarity
    # along dim=1 (the embedding dimension) for each layer in the batch (dim=0).
    similarity_profile = F.cosine_similarity(summary1_pooled, summary2_pooled, dim=1)

    return similarity_profile
    

def apply_style_transfer(v_content, v_style):
    # extract the mean and std from v_style, which is of shape [bsz, num_variates, seq_len, emb_dim]
    style_mean = v_style.mean(dim=-2, keepdim=True)
    style_std = v_style.std(dim=-2, keepdim=True)

    # normalise the v_content
    content_mean = v_content.mean(dim=-2, keepdim=True)
    content_std = v_content.std(dim=-2, keepdim=True)

    normalised_content = (v_content - content_mean) / (content_std + 1e-5)
    
    # apply the style to normalised_content
    stylised_content = (normalised_content * style_std) + style_mean
    return stylised_content # [bsz, num_variates, seq_len, emb_dim]
# --- End Placeholder Section ---

def create_sliding_windows(
    df: Optional[pd.DataFrame] = None, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,
    series: Optional[np.ndarray] = None, 
    window_size: int = 128, 
    stride: int = 1,
    column_name: Optional[str] = None,
) -> list[torch.Tensor]:
    """
    Generates a list of sliding windows from a time series DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex and a target column.
        start_date (str): The starting date for the windowing period (inclusive).
        end_date (str): The ending date for the windowing period (inclusive).
        series (np.ndarray): Synthetic data.
        window_size (int): The number of timesteps in each window.
        stride (int): The step size to move the window.
        column_name (str): The name of the column containing the time series values.

    Returns:
        list[torch.Tensor]: A list of windows, each as a PyTorch FloatTensor.
    """
    if df is None:
        assert series is not None, "Either provide `series` or provide the other arguments"         
    period_df = df.loc[start_date:end_date] if df is not None else None 
    series = period_df[column_name].values if series is None else series 
    print(f"Generating windows from a series of length {len(series)}...")
    windows = []
    num_windows = (len(series) - window_size) // stride + 1
    
    for i in range(num_windows):
        start_index = i * stride
        end_index = start_index + window_size
        window = series[start_index:end_index]
        windows.append(torch.tensor(window, dtype=torch.float32))
        
    print(f"Successfully created {len(windows)} windows.")
    return windows


def create_attention_mask_for_layer(
    layer: 'TransformerLayer',
    reduced_id_mask: torch.Tensor
) -> torch.Tensor:
    """
    Creates the appropriate attention mask for a given TransformerLayer.
    
    This function checks the layer's attention type (TimeWise vs. SpaceWise)
    and produces a mask of the correct shape. For our single-variate case,
    this mask is often trivial (all ones/trues), but its shape is critical.

    Args:
        layer (TransformerLayer): The specific transformer layer.
        reduced_id_mask (torch.Tensor): The mask from the patch embedder.

    Returns:
        torch.Tensor: The correctly shaped attention mask.
    """
    # Shape of reduced_id_mask: [batch, variate, seq_len]
    batch_size, num_variates, seq_len = reduced_id_mask.shape
    
    # The important part is matching the expected shape for the attention type
    if isinstance(layer.attention, TimeWiseMultiheadAttention):
        # TimeWise expects mask of shape [batch, num_heads, seq_len, seq_len]
        # For our case, we want all patches to attend to each other.
        # A value of `True` in the mask means "attention is allowed".
        mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool, device=reduced_id_mask.device)
        # We add a dimension for the heads
        return mask.unsqueeze(1) # -> [batch, 1, seq_len, seq_len]

    elif isinstance(layer.attention, SpaceWiseMultiheadAttention):
        # SpaceWise expects mask of shape [batch, num_heads, variate, variate]
        # Since we have only one variate, this is a 1x1 matrix.
        mask = torch.ones(batch_size, num_variates, num_variates, dtype=torch.bool, device=reduced_id_mask.device)
        return mask.unsqueeze(1) # -> [batch, 1, variate, variate]
        
    else:
        # Fallback for safety
        return None

def plot_single_forecast_subplot(
    ax: plt.Axes,
    stylized_forecast,
    context_window_original_scale: torch.Tensor,
    ground_truth_original_scale: Optional[torch.Tensor] = None,
    original_forecast = None,
    title: str = "",
    ylim_lower: Optional[float] = None,
    ylim_upper: Optional[float] = None,
):
    """
    Paints a single, complete fan chart onto a specific Matplotlib Axes object.
    
    Args:
        ax (plt.Axes): The subplot axes to draw on.
        stylized_forecast: The main forecast object after style transfer.
        context_window_original_scale (torch.Tensor): The raw historical data.
        ground_truth_original_scale (torch.Tensor, optional): The actual future values.
        original_forecast (MockForecastObject, optional): The model's vanilla forecast.
        title (str): The title for this specific subplot.
        ylim_lower (float, optional): Lower limit for y-axis. If None, auto-scaled.
        ylim_upper (float, optional): Upper limit for y-axis. If None, auto-scaled.
    """
    
    # --- 1. Extract Quantiles for the STYLIZED Forecast ---
    stylized_median = stylized_forecast.median.squeeze().cpu().numpy()
    stylized_q05 = stylized_forecast.quantile(0.05).squeeze().cpu().numpy()
    stylized_q95 = stylized_forecast.quantile(0.95).squeeze().cpu().numpy()
    stylized_q25 = stylized_forecast.quantile(0.25).squeeze().cpu().numpy()
    stylized_q75 = stylized_forecast.quantile(0.75).squeeze().cpu().numpy()
    
    forecast_len = stylized_median.shape[-1]
    
    # --- 2. Plotting Setup ---
    context_len = context_window_original_scale.shape[-1]
    context_timesteps = np.arange(-context_len, 0)
    forecast_timesteps = np.arange(forecast_len)
    
    # --- 3. Drawing on the provided Axes object 'ax' ---
    ax.plot(context_timesteps, context_window_original_scale.cpu().numpy().squeeze(), 'k-', linewidth=2.5, label='Historical Context')
    ax.plot(forecast_timesteps, stylized_median, 'b-', linewidth=2, label='Intervened Forecast')
    ax.fill_between(forecast_timesteps, stylized_q05, stylized_q95, color='blue', alpha=0.15, label='Intervened 90% PI')
    ax.fill_between(forecast_timesteps, stylized_q25, stylized_q75, color='blue', alpha=0.3, label='Intervened 50% PI')

    if original_forecast is not None:
        original_median = original_forecast.median.squeeze().cpu().numpy()
        ax.plot(forecast_timesteps, original_median, 'g--', linewidth=2.5, label='Original Forecast')

    if ground_truth_original_scale is not None:
        gt_len = ground_truth_original_scale.shape[-1]
        gt_timesteps = np.arange(gt_len)
        ax.plot(gt_timesteps, ground_truth_original_scale.cpu().numpy().squeeze(), 'r-', linewidth=2.5, label='Ground Truth')

    # --- 4. Subplot-specific Formatting ---
    ax.set_title(title, fontsize=32, pad=15)
    ax.tick_params(axis='x', labelsize=38)
    ax.tick_params(axis='y', labelsize=38)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)

    if ylim_lower is not None or ylim_upper is not None:
        current_ylim = ax.get_ylim()
        lower = ylim_lower if ylim_lower is not None else current_ylim[0]
        upper = ylim_upper if ylim_upper is not None else current_ylim[1]
        ax.set_ylim(lower, upper)
        
def plot_multi_stylized_forecast_subplot(
    ax: plt.Axes,
    stylized_forecasts,
    forecast_labels: List[str],
    context_window_original_scale: torch.Tensor,
    ground_truth_original_scale: Optional[torch.Tensor] = None,
    original_forecast = None,
    title: str = ""
):
    """
    Paints a multi-forecast comparison on a single Matplotlib Axes object.

    Args:
        ax (plt.Axes): The subplot axes to draw on.
        stylized_forecasts (List): A list of forecast objects from different style interventions.
        forecast_labels (List[str]): A list of labels for each stylized forecast.
        context_window_original_scale (torch.Tensor): The raw historical data.
        ground_truth_original_scale (torch.Tensor, optional): The actual future values.
        original_forecast (MockForecastObject, optional): The model's vanilla forecast.
        title (str): The title for this subplot.
    """
    
    # --- 1. Plot the Static Elements (Context, Ground Truth, Original Forecast) ---
    context_len = context_window_original_scale.shape[-1]
    context_timesteps = np.arange(-context_len, 0)
    ax.plot(context_timesteps, context_window_original_scale.cpu().numpy().squeeze(), 'k-', linewidth=2.5, label='Historical Context')

    if ground_truth_original_scale is not None:
        gt_len = ground_truth_original_scale.shape[-1]
        gt_timesteps = np.arange(gt_len)
        ax.plot(gt_timesteps, ground_truth_original_scale.cpu().numpy().squeeze(), 'r-', linewidth=2.5, label='Ground Truth')

    if original_forecast is not None:
        original_median = original_forecast.median.squeeze().cpu().numpy()
        forecast_len = original_median.shape[-1]
        forecast_timesteps = np.arange(forecast_len)
        ax.plot(forecast_timesteps, original_median, color='limegreen', linestyle='--', linewidth=2.5, label='Original Forecast')

    # --- 2. Plot the Multiple Stylized Forecasts with a Color Gradient ---
    
    # Create a color map that goes from a light to a dark color (e.g., blue or purple)
    # This visually represents the increasing severity of the style.
    num_forecasts = len(stylized_forecasts)
    colors = cm.get_cmap('viridis', num_forecasts + 2) # +2 to avoid the lightest/darkest extremes

    for i, (forecast, label) in enumerate(zip(stylized_forecasts, forecast_labels)):
        median = forecast.median.squeeze().cpu().numpy()
        q25 = forecast.quantile(0.25).squeeze().cpu().numpy()
        q75 = forecast.quantile(0.75).squeeze().cpu().numpy()
        
        forecast_len = median.shape[-1]
        forecast_timesteps = np.arange(forecast_len)

        # Plot the median line with a distinct color from the gradient
        ax.plot(forecast_timesteps, median, color=colors(i + 1), linewidth=2.0, label=label)
        
        # Plot the 50% Prediction Interval with a semi-transparent version of the same color
        ax.fill_between(forecast_timesteps, q25, q75, color=colors(i + 1), alpha=0.25)

    # --- 3. Subplot-specific Formatting ---
    ax.set_title(title, fontsize=30, pad=15)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=20)

def impute_with_window(df, window_size=3):
    """
    Impute NaN values with the average of 'window_size' values before and after the NaN index.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    window_size (int): Number of values before and after to use for averaging (default=3)
    
    Returns:
    pd.DataFrame: DataFrame with NaN values imputed
    """
    df_imputed = df.copy()
    
    for col in df_imputed.columns:
        series = df_imputed[col]
        nan_indices = series.index[series.isna()].tolist()
        
        for idx in nan_indices:
            # Get indices for values before and after
            before_start = max(0, idx - window_size)
            before_end = idx
            after_start = idx + 1
            after_end = min(len(series), idx + window_size + 1)
            
            # Get values before and after (excluding NaN values)
            before_values = series.iloc[before_start:before_end].dropna()
            after_values = series.iloc[after_start:after_end].dropna()
            
            # Combine values for averaging
            combined_values = pd.concat([before_values, after_values])
            
            # If we have values to average, impute; otherwise leave as NaN
            if len(combined_values) > 0:
                df_imputed.loc[idx, col] = combined_values.mean()
    
    return df_imputed

def plot_probabilistic_forecast(
    stylized_forecast,
    context_window_original_scale: torch.Tensor,
    ground_truth_original_scale: Optional[torch.Tensor] = None,
    original_forecast: Optional['MockForecastObject'] = None,
    title: str = "Generated Probabilistic Forecast",
    save_path: str = None,
    plot_name: str = None
):
    """
    Generates a complete fan chart, comparing a stylized forecast against
    the original vanilla forecast and the ground truth.

    Args:
        stylized_forecast: The main forecast object after style transfer.
        context_window_original_scale (torch.Tensor): The raw historical data.
        ground_truth_original_scale (torch.Tensor, optional): The actual future values.
        original_forecast (MockForecastObject, optional): The model's vanilla forecast
                                                           for the same context window.
        title (str): The title for the plot.
        save_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    
    # --- 1. Extract Quantiles for the STYLIZED Forecast ---
    stylized_median = stylized_forecast.median.squeeze().cpu().numpy()
    stylized_q05 = stylized_forecast.quantile(0.05).squeeze().cpu().numpy()
    stylized_q95 = stylized_forecast.quantile(0.95).squeeze().cpu().numpy()
    stylized_q25 = stylized_forecast.quantile(0.25).squeeze().cpu().numpy()
    stylized_q75 = stylized_forecast.quantile(0.75).squeeze().cpu().numpy()
    
    forecast_len = stylized_median.shape[-1]
    
    # --- 2. Plotting Setup ---
    plt.figure(figsize=(18, 8)) # Made the figure wider for clarity
    
    context_len = context_window_original_scale.shape[-1]
    context_timesteps = np.arange(-context_len, 0)
    forecast_timesteps = np.arange(forecast_len)
    
    # Plot Historical Context
    plt.plot(context_timesteps, context_window_original_scale.cpu().numpy().squeeze(), 'k-', linewidth=2.5, label='Historical Context') #can include 'k-', in the 3rd positioin

    # --- 3. Plot the STYLIZED Probabilistic Forecast (Fan Chart) ---
    plt.plot(forecast_timesteps, stylized_median, 'b-', linewidth=2, label='Transformed Forecast') # can include 'b-' after stylized_median
    plt.fill_between(forecast_timesteps, stylized_q05, stylized_q95, color='blue', alpha=0.15, label='Transformed 90% PI')
    plt.fill_between(forecast_timesteps, stylized_q25, stylized_q75, color='blue', alpha=0.3, label='Transformed 50% PI')

    # --- 4. NEW: Plot the ORIGINAL (Vanilla) Forecast Median ---
    if original_forecast is not None:
        original_median = original_forecast.median.squeeze().cpu().numpy()
        plt.plot(
            forecast_timesteps, 
            original_median, 
            'g--', # Dashed green line
            linewidth=2.5,
            label='Original Forecast'
        )
    # --- END NEW SECTION ---

    # Plot Ground Truth
    if ground_truth_original_scale is not None:
        gt_len = ground_truth_original_scale.shape[-1]
        gt_timesteps = np.arange(gt_len)
        plt.plot(gt_timesteps, ground_truth_original_scale.cpu().numpy().squeeze(), 'r-', linewidth=2.5, label='Ground Truth') # can include 'r-', in the 3rd position

    # plt.title(title, fontsize=18)
    plt.tick_params(axis='x', labelsize=28)
    plt.tick_params(axis='y', labelsize=28)
    
    plt.xlabel('Time Steps', fontsize=21)
    plt.ylabel('Value', fontsize=21)
    plt.legend(fontsize=28)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
