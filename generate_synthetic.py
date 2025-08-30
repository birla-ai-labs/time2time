import torch 
import pandas as pd 
from toto.model import Toto
from matplotlib.pyplot import plt
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from src.stylised_forecaster import StylizedTotoForecaster
from src.synth_util import generate_single_series
from src.util import create_sliding_windows, plot_probabilistic_forecast, plot_multi_stylized_forecast_subplot

WINDOW_SYNTH_SIZE = 256
STRIDE = 1

toto_model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0', cache_dir='cache_dir')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
toto_model.to(device)
toto_model.eval() # Set to evaluation mode
toto_backbone = toto_model.model



synthetic_crash_1 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=0.2, seed=3)['price'].to_numpy()
synthetic_crash_2 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=0.5, seed=3)['price'].to_numpy()
synthetic_crash_3 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=1.0, seed=3)['price'].to_numpy()
synthetic_crash_4 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=1.5, seed=3)['price'].to_numpy()
synthetic_crash_5 = generate_single_series(T=320, regime="crash",  start_price=5000, severity=2.0, seed=3)['price'].to_numpy()

synthetic_crash_1_windows = create_sliding_windows(
    series=synthetic_crash_1,
    window_size=WINDOW_SYNTH_SIZE,
    stride=STRIDE,
)
synthetic_crash_2_windows = create_sliding_windows(
    series=synthetic_crash_2,
    window_size=WINDOW_SYNTH_SIZE,
    stride=STRIDE,
)
synthetic_crash_3_windows = create_sliding_windows(
    series=synthetic_crash_3,
    window_size=WINDOW_SYNTH_SIZE,
    stride=STRIDE,
)
synthetic_crash_4_windows = create_sliding_windows(
    series=synthetic_crash_4,
    window_size=WINDOW_SYNTH_SIZE,
    stride=STRIDE,
)
synthetic_crash_5_windows = create_sliding_windows(
    series=synthetic_crash_5,
    window_size=WINDOW_SYNTH_SIZE,
    stride=STRIDE,
)
synthetic_normal = generate_single_series(T=320, regime="calm",  start_price=2000, seed=3)['price'].to_numpy()
synthetic_normal_windows = create_sliding_windows(
    series=synthetic_normal,
    window_size=WINDOW_SYNTH_SIZE,
    stride=STRIDE,
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


forecaster = StylizedTotoForecaster(toto_backbone)
original_forecaster = TotoForecaster(toto_backbone)

synthetic_1_forecast = forecaster.stylized_forecast(
    synthetic_normal_inputs, #base
    synthetic_crash_1_inputs, #style
    intervention_layer_idx=8,
    prediction_length=64,
    num_samples=256,
    samples_per_batch=256,
    use_kv_cache=True,
)
synthetic_2_forecast = forecaster.stylized_forecast(
    synthetic_normal_inputs, #base
    synthetic_crash_2_inputs, #style
    intervention_layer_idx=8,
    prediction_length=64,
    num_samples=256,
    samples_per_batch=256,
    use_kv_cache=True,
)
synthetic_3_forecast = forecaster.stylized_forecast(
    synthetic_normal_inputs, #base
    synthetic_crash_3_inputs, #style
    intervention_layer_idx=8,
    prediction_length=64,
    num_samples=256,
    samples_per_batch=256,
    use_kv_cache=True,
)
synthetic_4_forecast = forecaster.stylized_forecast(
    synthetic_normal_inputs, #base
    synthetic_crash_4_inputs, #style
    intervention_layer_idx=8,
    prediction_length=64,
    num_samples=256,
    samples_per_batch=256,
    use_kv_cache=True,
)
synthetic_5_forecast = forecaster.stylized_forecast(
    synthetic_normal_inputs, #base
    synthetic_crash_5_inputs, #style
    intervention_layer_idx=8,
    prediction_length=64,
    num_samples=256,
    samples_per_batch=256,
    use_kv_cache=True,
)
original_synthetic_forecast = original_forecaster.forecast(
    synthetic_normal_inputs, 
    prediction_length=64,
    num_samples=256,
    samples_per_batch=256,
)

all_stylized_forecasts = [
    synthetic_1_forecast,
    synthetic_2_forecast,
    synthetic_3_forecast,
    synthetic_4_forecast,
    synthetic_5_forecast
]

all_labels = [
    'Intervened Forecast (Severity = 0.2)',
    'Intervened Forecast (Severity = 0.5)',
    'Intervened Forecast (Severity = 1.0)',
    'Intervened Forecast (Severity = 1.5)',
    'Intervened Forecast (Severity = 2.0)'
]

fig, ax = plt.subplots(1, 1, figsize=(20, 10)) # A single, large plot

plot_multi_stylized_forecast_subplot(
    ax=ax,
    stylized_forecasts=all_stylized_forecasts,
    forecast_labels=all_labels,
    context_window_original_scale=synthetic_normal_inputs.series.squeeze(), 
    ground_truth_original_scale=synthetic_normal_windows[65][191:], 
    original_forecast=original_synthetic_forecast,
    title="Forecast Trajectories Under Different Crash Style Severities"
)

ax.set_xlabel('Time Steps', fontsize=26)
ax.set_ylabel('Value', fontsize=26)

plt.tight_layout()

save_path = "stylised_synthetic.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Severity spectrum figure saved to {save_path}")
