# time2time: Causal Intervention in Hidden States to Simulate Rare Events in Time Series Foundation Models
![Python Version](https://img.shields.io/badge/python-3.12.2-blue?style=flat)
> **Abstract:**
While transformer-based foundation models excel at forecasting routine patterns, two questions remain: do they internalize semantic concepts such as market regimes, or merely fit curves? And can their internal representations be leveraged to simulate rare, high-stakes events such as market crashes? To investigate this, we introduce activation transplantation, a causal intervention that manipulates hidden states by imposing the statistical moments of one event (e.g., a historical crash) onto another (e.g., a calm period) during the forward pass. This procedure deterministically steers forecasts: injecting crash semantics induces downturn predictions, while injecting calm semantics suppresses crashes and restores stability. Beyond binary control, we find that models encode a graded notion of event severity, with the latent vector norm directly correlating with the magnitude of systemic shocks. Validated across two architecturally distinct TSFMs, Toto (decoder only) and Chronos (encoder decoder), our results demonstrate that steerable, semantically grounded representations are a robust property of large time series transformers. Our findings provide evidence for a _latent concept space_ that governs model predictions, shifting interpretability from post-hoc attribution to direct causal intervention, and enabling semantic "what-if" analysis for strategic stress-testing.

## 

<p align="center">
  <img src="assets/figure1_overview.png" width="80%">
</p>

-----

## Installation ⚙

1. Unzip the provided code and navigate to the project directory.
2. Create a virtual environment and activate it:
```bash
    # On Linux/MacOS
    python3 -m venv time2time_env
    source time2time_env/bin/activate

    # On Windows
    python3 -m venv time2time_env
    time2time_env\Scripts\activate
```

3. Install dependencies:
```bash
    pip3 install -r requirements.txt
```
-----

## Usage 📈
### Real Data
##
To generate intervention plots for the real data (using Toto-Open-Base-1.0), run the following command:
```bash
    python3 generate_toto_predictions.py
```
_The plot will be saved as `stylised_real.png` in the current directory._
##

TO generate intervention plots for the real data (using Chronos-T5-Small), run the following command:
```bash
    python3 generate_chronos_predictions.py
```
_The plot will be saved as `chronos_intervention_plot.png` in the current directory._

##

### Synthetic Data
##
To generate intervention plots for the synthetic data (using Toto-Open-Base-1.0), run the following command:
```bash
    python3 generate_toto_synthetic.py
```

_The plot will be saved as `stylised_synthetic.png` in the current directory._
##

### Heatmap
##
To generate cosine similarity heatmaps, run the following command:
```bash
    python3 generate_toto_heatmap.py
```
_The plot will be saved as `heatmap.png` in the current directory._
##

## Citation

If you use this code or the methodology in your research, please cite our paper:

```bibtex
@inproceedings{
  sanyal2025timetime,
  title={time2time: Causal Intervention in Hidden States to Simulate Rare Events in Time Series Foundation Models},
  author={Debdeep Sanyal and Aaryan Nagpal and Dhruv Kumar and Murari Mandal and Saurabh Deshpande},
  booktitle={Recent Advances in Time Series Foundation Models Have We Reached the 'BERT Moment'?},
  year={2025},
  url={https://openreview.net/forum?id=VElPLJUr2G}
}
```
