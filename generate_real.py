import torch 
import pandas as pd 
from toto.model import Toto
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from src.stylised_forecaster import StylizedTotoForecaster
from src.util import impute_with_window, create_sliding_windows, get_toto_activations, plot_probabilistic_forecast

START_2017_NORMAL = '2017-01-12'
END_2017_NORMAL = '2017-08-09'
START_2008_CRASH = '2008-07-25'
END_2008_CRASH = '2009-03-09'
WINDOW_SIZE=128
STRIDE=1

toto_model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0', cache_dir='cache_dir')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
toto_model.to(device)
toto_model.eval() # Set to evaluation mode
toto_backbone = toto_model.model


normal_df = pd.read_csv('./2017_data.csv')
normal_df = impute_with_window(normal_df, window_size=3)
normal_df['observation_date'] = pd.to_datetime(normal_df['observation_date'])
normal_df.set_index('observation_date', inplace=True)

turbulence_df = pd.read_csv('./2008_crash.csv')
turbulence_df = impute_with_window(turbulence_df, window_size=3)
turbulence_df['observation_date'] = pd.to_datetime(turbulence_df['observation_date'])
turbulence_df.set_index('observation_date', inplace=True)

normal_windows = create_sliding_windows(
    df=normal_df,
    start_date=START_2017_NORMAL,
    end_date=END_2017_NORMAL,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    column_name='NASDAQ100'
)
turbulence_windows = create_sliding_windows(
    df=turbulence_df,
    start_date=START_2008_CRASH,
    end_date=END_2008_CRASH,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    column_name='NASDAQ100'
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

_, act_08_crash = get_toto_activations(toto_backbone, [act_08_crash_dict])
_, act_17_calm = get_toto_activations(toto_backbone, [act_17_calm_dict])

forecaster = StylizedTotoForecaster(toto_backbone)
original_forecaster = TotoForecaster(toto_backbone)

forecast_08_17 = forecaster.stylized_forecast(
    normal_2017_inputs, #target
    turb_2008_inputs, #style
    intervention_layer_idx=8,
    prediction_length=64,
    num_samples=256,
    samples_per_batch=256,
    use_kv_cache=True,
)
original_forecast_08_17 = original_forecaster.forecast(
    normal_2017_inputs, 
    prediction_length=64,
    num_samples=256,
    samples_per_batch=256,
)


plot_probabilistic_forecast(forecast_08_17, normal_windows[0], normal_windows[22][105:], original_forecast=original_forecast_08_17, save_path='stylised_real.png')
