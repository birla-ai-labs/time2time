# setup
from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])


def set_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seeds(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Note on dataset end dates: 
# The model processes only 128 time steps for the intervention and the remaining portion is utilized to display the ground truth plot.

# setting data
TARGET_COL = "NASDAQ100"
PATHS: Dict[str, Dict[str, pd.Timestamp | str]] = {
    "2000 Crash": {
        "path": "../data/2000_crash.csv",
        "start_date": pd.Timestamp("2000-08-31"),
        "end_date": pd.Timestamp("2001-04-04"), 
    },
    "2007 Calm": {
        "path": "../data/2007_normal.csv",
        "start_date": pd.Timestamp("2007-03-12"),
        "end_date": pd.Timestamp("2007-11-02"),
    },
    "2008 Crash": {
        "path": "../data/2008_crash.csv",
        "start_date": pd.Timestamp("2008-07-25"),
        "end_date": pd.Timestamp("2009-03-09"),
    },
    "2017 Calm": {
        "path": "../data/2017_data.csv",
        "start_date": pd.Timestamp("2017-01-12"),
        "end_date": pd.Timestamp("2017-08-09"),
    },
    "2020 Crash": {
        "path": "../data/2020_crash.csv",
        "start_date": pd.Timestamp("2021-12-27"),
        "end_date": pd.Timestamp("2022-10-04"),
    },
    "2019 Calm": {
        "path": "../data/2019_data.csv",
        "start_date": pd.Timestamp("2019-06-01"),
        "end_date": pd.Timestamp("2020-02-06"),
    },
}

# helper functions here:

def impute_with_window(df: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    x = df[TARGET_COL].astype("float32")
    x = (
        x.rolling(window_size, center=True, min_periods=1)
        .mean()
        .ffill()
        .bfill()
        .astype("float32")
    )
    out = df.copy()
    out[TARGET_COL] = x
    return out


def load_segment(name: str, window_size: int = 3) -> pd.Series:
    meta = PATHS[name]
    df = pd.read_csv(meta["path"], parse_dates=["observation_date"])
    df = impute_with_window(df, window_size=window_size)
    df.set_index("observation_date", inplace=True)
    s = df.loc[meta["start_date"] : meta["end_date"], TARGET_COL].astype("float32")
    s.name = name
    return s, df.loc[meta["start_date"] : meta["end_date"], :]


def load_all_segments() -> Dict[str, pd.Series]:
    return {k: load_segment(k)[0] for k in PATHS.keys()}


# chronos + activation setup

def get_chronos(model_id: str = "amazon/chronos-t5-small"):
    from chronos import ChronosPipeline  # imported lazily

    return ChronosPipeline.from_pretrained(model_id, device_map="auto")


def register_block_hooks(chronos, activations_dict: Dict[int, torch.Tensor]):
    """
    Registers forward hooks on each encoder block to capture hidden states.
    Returns a list of handles (remember to .remove()).
    """
    handles = []
    for i, block in enumerate(chronos.model.model.encoder.block):

        def _hook(module, inp, out, idx=i):
            hs = out[0] if isinstance(out, tuple) else out  # T5Block returns tuple
            activations_dict[idx] = hs.detach().cpu()

        handles.append(block.register_forward_hook(_hook))
    return handles


def get_layerwise_activations(
    chronos, series: pd.Series, max_len: int = 128
) -> torch.Tensor:
    """
    Returns stacked activations of shape (L, S+1, D) across encoder blocks.
    """
    activations: Dict[int, torch.Tensor] = {}
    handles = register_block_hooks(chronos, activations)
    try:
        seq = torch.tensor(series.values[:max_len], dtype=torch.float32).unsqueeze(0)
        _ = chronos.embed(seq)  # triggers hooks
    finally:
        for h in handles:
            h.remove()
    layers = [activations[i].squeeze(0) for i in sorted(activations.keys())]
    return torch.stack(layers)  # (L, S+1, D)

# encoding activation hooks
def _encode_ids_mask(chronos, context_tensor_cpu: torch.Tensor):
    token_ids, attention_mask, scale = chronos.tokenizer.context_input_transform(
        context_tensor_cpu
    )
    return (
        token_ids.to(chronos.model.device),
        attention_mask.to(chronos.model.device),
        scale,
    )


def get_style_stats(
    chronos, series: pd.Series, layer_idx: int = 1, max_len: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    activations: Dict[int, torch.Tensor] = {}
    handles = register_block_hooks(chronos, activations)
    try:
        seq = torch.tensor(series.values[:max_len], dtype=torch.float32).unsqueeze(0)
        _ = chronos.embed(seq)  # triggers hooks
    finally:
        for h in handles:
            h.remove()
    h = activations[layer_idx].squeeze(0)[:-1]  # (S, D), drop EOS
    return h.mean(0), h.std(0)


def _make_adain_hook(
    style_mean: torch.Tensor, style_std: torch.Tensor, beta: float = 1.0
):
    sm = style_mean.view(1, 1, -1)
    ss = style_std.view(1, 1, -1)

    def hook(module, inp, out):
        hidden, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, tuple())
        x = hidden[:, :-1, :]  # (B, S, D), drop EOS
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True) + 1e-5
        sm_d = sm.to(hidden.device, hidden.dtype)
        ss_d = ss.to(hidden.device, hidden.dtype) + 1e-5

        y_adain = (x - mu) / sd
        y_adain = y_adain * ss_d + sm_d
        y = (1.0 - beta) * x + beta * y_adain
        new_hidden = torch.cat([y, hidden[:, -1:, :]], dim=1)
        return (new_hidden, *rest) if isinstance(out, tuple) else new_hidden

    return hook


@contextmanager
def _stylize_block(
    chronos,
    layer_idx: int,
    style_mean: torch.Tensor,
    style_std: torch.Tensor,
    beta: float = 1.0,
):
    h = chronos.model.model.encoder.block[layer_idx].register_forward_hook(
        _make_adain_hook(style_mean, style_std, beta)
    )
    try:
        yield
    finally:
        h.remove()


def stylized_forecast_at_layer(
    chronos,
    content: pd.Series,
    style: pd.Series,
    layer_idx: int = 1,
    context_len: int = 128,
    pred_len: int = 64,
    num_samples: int = 200,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    from transformers import GenerationConfig
    from transformers.modeling_outputs import BaseModelOutput

    style_mean, style_std = get_style_stats(
        chronos, style, layer_idx=layer_idx, max_len=context_len
    )

    ctx_cpu = torch.tensor(content.values[:context_len], dtype=torch.float32).unsqueeze(
        0
    )
    token_ids, attention_mask, content_scale = _encode_ids_mask(chronos, ctx_cpu)

    _, _, style_scale = chronos.tokenizer.context_input_transform(
        torch.tensor(style.values[:context_len], dtype=torch.float32).unsqueeze(0)
    )

    with _stylize_block(chronos, layer_idx, style_mean, style_std, beta):
        with torch.no_grad():
            enc = chronos.model.model.encoder(
                input_ids=token_ids, attention_mask=attention_mask, return_dict=True
            )
    enc_out = BaseModelOutput(last_hidden_state=enc.last_hidden_state)

    gcfg = GenerationConfig(
        min_new_tokens=pred_len,
        max_new_tokens=pred_len,
        do_sample=True,
        num_return_sequences=num_samples,
        eos_token_id=chronos.model.config.eos_token_id,
        pad_token_id=chronos.model.config.pad_token_id,
        temperature=temperature
        if temperature is not None
        else chronos.model.config.temperature,
        top_k=top_k if top_k is not None else chronos.model.config.top_k,
        top_p=top_p if top_p is not None else chronos.model.config.top_p,
    )
    with torch.no_grad():
        preds = chronos.model.model.generate(
            encoder_outputs=enc_out,
            attention_mask=attention_mask,
            generation_config=gcfg,
        )
    preds = preds[..., 1:]  # drop decoder start
    preds = preds.reshape(1, num_samples, -1)  # (1, N, T)
    scale = alpha * style_scale + (1.0 - alpha) * content_scale
    preds = chronos.tokenizer.output_transform(preds.to(scale.device), scale)
    return preds.to(dtype=torch.float32, device="cpu")  # (1, N, T)


def plot_all_segments_grid(segments: Dict[str, pd.Series]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 5.5), sharey=False)
    axes = axes.ravel()
    for ax, (title, seg) in zip(axes, segments.items()):
        ax.plot(seg.index, seg.values, lw=1)
        ax.set_title(title)
        ax.grid(True)
    fig.suptitle(
        "NASDAQ100 Time Series Segments - Crash and Calm",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


def summarise_samples(
    x: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = x.median(1).values.squeeze(0).cpu().numpy()
    q25 = torch.quantile(x, 0.25, dim=1).squeeze().cpu().numpy()
    q75 = torch.quantile(x, 0.75, dim=1).squeeze().cpu().numpy()
    q05 = torch.quantile(x, 0.05, dim=1).squeeze().cpu().numpy()
    q95 = torch.quantile(x, 0.95, dim=1).squeeze().cpu().numpy()
    return m, q25, q75, q05, q95


def get_predictions(
    chronos, content, style, layer_idx, context_len, pred_len, num_samples, alpha, beta
):
    stylized = stylized_forecast_at_layer(
        chronos,
        content,
        style,
        layer_idx=layer_idx,
        context_len=context_len,
        pred_len=pred_len,
        num_samples=num_samples,
        alpha=alpha,
        beta=beta,
    )
    baseline = chronos.predict(
        context=torch.tensor(content.values[:context_len], dtype=torch.float32),
        prediction_length=pred_len,
        num_samples=num_samples,
    )
    return stylized, baseline


def plot_stylegrid(
    chronos,
    segments: Dict[str, pd.Series],
    pairs: List[Tuple[str, str]],
    lims: Dict[Tuple[str, str], Tuple[int, int]],
    layer_idx: int = 1,
    context_len: int = 128,
    pred_len: int = 64,
    num_samples: int = 200,
    betas: List[float] = [1.0] * 6,
    save_path: Optional[str] = None,
    need_one: bool = False,
):
    if not need_one:
        assert len(pairs) == 6, "Need 6 (content, style) pairs for a 2x3 grid."

        fig, axes = plt.subplots(2, 3, figsize=(36, 18), sharey=False)
        axes = axes.ravel()
    else:
        assert len(pairs) == 1, "Need 1 (content, style) pair for a 1x1 grid."
        fig, axes = plt.subplots(1, 1, figsize=(14, 12), sharey=False)
        axes = [axes]

    last_ax = None
    axis_no = 0
    alphas=[0.1, -0.2, -0.1, 0.2, -1.8, -0.4]


    for ax, (content_name, style_name) in zip(axes, pairs):
        axis_no += 1
        content = segments[content_name]
        style = segments[style_name]

        i = pairs.index((content_name, style_name))

        stylized, baseline = get_predictions(
            chronos,
            content,
            style,
            layer_idx=layer_idx,
            context_len=context_len,
            pred_len=pred_len,
            num_samples=num_samples,
            alpha=alphas[i],
            beta=betas[i],
        )

        s_med, s_q25, s_q75, s_q05, s_q95 = summarise_samples(stylized)
        b_med, _, _, _, _ = summarise_samples(baseline)

        ctx = content.values[:context_len].astype("float32")

        t_hist = np.arange(-len(ctx), 0)
        t_fore = np.arange(pred_len)

        ax.plot(t_hist, ctx, "k-", linewidth=1.8, label="Historical Context")
        ax.plot(t_fore, s_med, "b-", linewidth=1.8, label="Intervened Forecast")
        ax.fill_between(
            t_fore, s_q05, s_q95, color="blue", alpha=0.15, label="Intervened 90% PI"
        )
        ax.fill_between(
            t_fore, s_q25, s_q75, color="blue", alpha=0.3, label="Intervened 50% PI"
        )
        ax.plot(t_fore, b_med, "g--", linewidth=1.8, label="Original Forecast")

        true_future = content.values[context_len:].astype("float32")

        ax.plot(
            np.arange(len(true_future)),
            true_future,
            "r-",
            linewidth=1.8,
            label="Ground Truth",
        )

        ax.set_ylim(lims[(content_name, style_name)])

        ax.set_title(f"Target: {content_name} - Style: {style_name}", fontsize=32, pad=15)
        ax.tick_params(axis="x", labelsize=36)
        ax.tick_params(axis="y", labelsize=36)
        ax.grid(True, linestyle="--", alpha=0.6)

        if axis_no == 3:
            last_ax = ax

    fig.text(0.5, 0.01, "Time Steps", ha="center", va="center", fontsize=40)
    fig.text(
        0.09, 0.5, "Value", ha="center", va="center", rotation="vertical", fontsize=40
    )

    if last_ax:
        handles, labels = last_ax.get_legend_handles_labels()
        order = [
            "Historical Context",
            "Ground Truth",
            "Original Forecast",
            "Intervened Forecast",
            "Intervened 50% PI",
            "Intervened 90% PI",
        ]
        hdict = dict(zip(labels, handles))
        handles_ord = [hdict[l] for l in order if l in hdict]
        labels_ord = [l for l in order if l in hdict]
        last_ax.legend(handles_ord, labels_ord, loc="best", fontsize=18)

    plt.tight_layout(rect=[0.1, 0.05, 0.9, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved to {save_path}")
    else:
        plt.show()


chronos = get_chronos(model_id="amazon/chronos-t5-small")

pairs = [
    ("2017 Calm", "2008 Crash"),
    ("2007 Calm", "2000 Crash"),
    ("2019 Calm", "2020 Crash"),
    ("2008 Crash", "2017 Calm"),
    ("2000 Crash", "2007 Calm"),
    ("2020 Crash", "2019 Calm"),
]

# Added to ensure the same y-limits for toto and chronos
lims = {
    ("2017 Calm", "2008 Crash"): (4500, 7000),
    ("2007 Calm", "2000 Crash"): (1500, 2500),
    ("2019 Calm", "2020 Crash"): (7000, 9500),
    ("2008 Crash", "2017 Calm"): (1000, 2300),
    ("2000 Crash", "2007 Calm"): (1600, 5000),
    ("2020 Crash", "2019 Calm"): (10500, 17500),
}

segments = load_all_segments()

plot_stylegrid(
    chronos=chronos,
    segments=segments,
    pairs=pairs,
    layer_idx=1,
    context_len=128,
    pred_len=64,
    num_samples=200,
    save_path="chronos_intervention_plot.png", #Set to None to show the plot instead of saving it
    need_one=False,
    lims=lims,
)
