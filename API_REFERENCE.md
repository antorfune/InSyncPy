# InSyncPy – Technical Reference

## Overview

`InSyncPy` is a signal processing and time-frequency analysis pipeline for chronobiological oscillating signals designed to:

- Denoise signals using Discrete Wavelet Transform (DWT)
- Remove trends using smoothing splines
- Trim signals based on detected peaks
- Compute a Continuous Wavelet Transform (CWT)
- Extract instantaneous frequency and amplitude through ridge detection
- Fit an exponential decay model to the amplitude envelope
- Generate diagnostic visualizations

---

# Constructor

## `__init__`

```python
__init__(
    t,
    sig,
    coeff=5e5,
    trim_pt_start=0,
    trim_pt_end=1,
    sig_name='Synthetic signal',
)
```

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `t` | ndarray | Time vector |
| `sig` | ndarray | Input signal |
| `coeff` | float | Smoothing spline parameter (default: `5e5`) |
| `trim_pt_start` | int | Number of peaks removed at the beginning (default: `0`) |
| `trim_pt_end` | int | Number of peaks removed at the end (default: `1`) |
| `sig_name` | str | Signal name (default: `'Synthetic signal'`) |

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | `t` or `sig` is not a numpy array; arrays don't contain float values; `coeff` is not numeric; `trim_pt_start` or `trim_pt_end` is not an integer; `sig_name` or `dir_save` is not a string; `show_plot`, `save_plot`, or `save_csv` is not a boolean |
| `ValueError` | `t` and `sig` have different lengths; arrays are empty; `coeff` is not positive; `trim_pt_start` or `trim_pt_end` is negative; `trim_pt_end` is less than 1; `trim_pt_start` is >= `trim_pt_end`; `trim_pt_end` exceeds number of positive peaks in signal |
| `ValueError` | No positive peaks found in signal (cannot define trim points) |

---

# Methods

## `denoise_dwt()`

### Description

Apply a Discrete Wavelet Transform (DWT) denoising procedure using the Symlet-8 wavelet.

### Uses

| Attribute | Type | Description |
|------------|------|-------------|
| `self.t` | ndarray | Time vector |
| `self.sig` | ndarray | Input noisy signal |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `out_t` | ndarray | Time vector |
| `out` | ndarray | Denoised signal |

---

## `detrend_smoothing_spline(t, sig)`

### Description

Estimate and remove the signal trend using a smoothing spline.

The detrended signal is normalized by its maximum absolute value.

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `t` | ndarray | Time vector |
| `sig` | ndarray | Signal to detrend |

### Uses

| Attribute | Type | Description |
|------------|------|-------------|
| `self.coeff` | float | Smoothing parameter |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `out_sig` | ndarray | Detrended and normalized signal |
| `trend` | ndarray | Estimated trend |

---

## `exponential_decay(time, decreasing_rate, coeff)`

### Description

Exponential decay model:

$$A(t) = coeff \times e^{-decreasing\_rate \cdot t}$$
### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `time` | ndarray | Time vector |
| `decreasing_rate` | float | Decay rate |
| `coeff` | float | Initial amplitude |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| ndarray | ndarray | Exponential decay signal |

---

## `fit_exp_decay(amplitude)`

### Description

Fit an exponential decay model to the instantaneous amplitude envelope.

The model assumes:$$A(t) = A(t_0) \times e^{-\epsilon \cdot t}$$

where `A(t0)` is fixed to the first amplitude value and `epsilon` is estimated.

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `amplitude` | ndarray | Instantaneous amplitude |

### Uses

| Attribute | Type | Description |
|------------|------|-------------|
| `self.t` | ndarray | Time vector (used to construct evenly-spaced time array starting at 0) |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `params` | ndarray | Estimated decay parameter (epsilon) |
| `err_eps` | ndarray | Standard deviation of the fitted parameter |

---

## `full_analysis(show_plot=False, save_plot=False, save_csv=False, dir_save='./')`

### Description

Execute the complete signal analysis pipeline.

### Workflow

1. Signal preparation (denoising, detrending, trimming)
2. Continuous Wavelet Transform with synchrosqueezing
3. Ridge extraction
4. Instantaneous frequency estimation
5. Instantaneous amplitude estimation
6. Smoothing of instantaneous quantities
7. Exponential decay fitting
8. Optional CSV export of results
9. Optional plot generation and export

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `show_plot` | bool | If `True`, displays plots during analysis (default: `False`) |
| `save_plot` | bool | If `True`, saves plots to JPEG files (default: `False`) |
| `save_csv` | bool | Export results to CSV (default: `False`) |
| `dir_save` | str | Output directory (default: `'./'`) |

### Returns

Dictionary containing:

| Key | Description |
|------|-------------|
| `t_denoised` | Denoised time vector |
| `signal_denoised` | Denoised signal |
| `trend` | Estimated trend |
| `signal_detrended` | Detrended signal |
| `t_trimmed` | Trimmed time vector |
| `signal_trimmed` | Trimmed signal |
| `freqs` | Frequency vector |
| `wx` | CWT coefficients |
| `tx` | Synchrosqueezed CWT coefficients |
| `t_inst` | Time vector for instantaneous features |
| `inst_freq` | Instantaneous frequency |
| `inst_period` | Instantaneous period (in hours) |
| `inst_amp` | Instantaneous amplitude |
| `inst_freq_smoothed` | Smoothed instantaneous frequency |
| `inst_amp_smoothed` | Smoothed instantaneous amplitude |
| `amp_decay_rate` | Estimated decay rate |
| `amp_decay_error` | Uncertainty on decay rate |

### Output

If `save_csv` = True, four CSV files are generated:

**`<sig_name>_detrended.csv`**

| Column |
|----------|
| time (hours) |
| denoised_signal |
| signal_trend |
| detrended_signal |

**`<sig_name>_model.csv`**

| Column |
|----------|
| time (hours) |
| signal_model |

**`<sig_name>_inst_period_amplitude.csv`**

| Column |
|----------|
| time |
| inst_period (h) |
| inst_amp |

**`<sig_name>_metrics.csv`**

| Column |
|----------|
| min_inst_amp |
| mean_inst_amp |
| max_inst_amp |
| amp_decay |
| amp_decay_err |
| min_inst_period (h) |
| mean_inst_period (h) |
| max_inst_period (h) |

### Output (Plots)

If `show_plot` = True, three figures are displayed:

1. **Preprocessing** — Denoising, detrending, and trimming
2. **Scaleogram** — CWT power spectrum with instantaneous frequency and amplitude
3. **Analysis** — Processed signal, instantaneous frequency, and amplitude

If `save_plot` = True, three JPEG files are saved:

- `<sig_name>_processing.jpg`
- `<sig_name>_scaleogram.jpg`
- `<sig_name>_analysis.jpg`

---

## `get_inst_amplitude(matrix_coeff)`

### Description

Estimate instantaneous amplitude from a time-frequency representation.

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `matrix_coeff` | ndarray | Time-frequency coefficient matrix |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `out` | ndarray | Instantaneous amplitude |

---

## `get_loc_freq(matrix_coeff)`

### Description

Estimate the energy distribution as a function of frequency.

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `matrix_coeff` | ndarray | Time-frequency coefficient matrix |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `out` | ndarray | Frequency-localized energy distribution |

---

## `scaleogram_visualisation(wx, freqs, t_inst, inst_freq, inst_amp)`

### Description

Generate a complete visualization including:

- Signal
- Wavelet scaleogram
- Global wavelet spectrum
- Instantaneous amplitude
- Instantaneous frequency

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `wx` | ndarray | CWT coefficients |
| `freqs` | ndarray | Frequency vector |
| `t_inst` | ndarray | Time vector for instantaneous features |
| `inst_freq` | ndarray | Instantaneous frequency |
| `inst_amp` | ndarray | Instantaneous amplitude |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `fig` | matplotlib.figure.Figure | Figure object containing the visualization |

---

## `signal_preparation()`

### Description

Complete signal preparation pipeline:

1. DWT denoising
2. Spline detrending
3. Peak-based trimming

### Returns

Dictionary containing:

| Key | Description |
|------|-------------|
| `t_denoised` | Denoised time vector |
| `signal_denoised` | Denoised signal |
| `trend` | Estimated trend |
| `signal_detrended` | Detrended signal |
| `t_trimmed` | Trimmed time vector |
| `signal_trimmed` | Trimmed signal |

---

## `sst_gmw_insync()`

### Description

Perform a synchrosqueezed Continuous Wavelet Transform (CWT) using Generalized Morse Wavelets (GMW) and extract the dominant ridge.

### Uses

| Attribute | Type | Description |
|------------|------|-------------|
| `self.t` | ndarray | Time vector (used to compute sampling frequency) |
| `self.sig` | ndarray | Signal to analyze |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `tx` | ndarray | Synchrosqueezed transform |
| `freqs` | ndarray | Frequencies associated with scales (Hz) |
| `inst_freq` | ndarray | Instantaneous frequency (ridge-based) |
| `inst_amp` | ndarray | Instantaneous amplitude (ridge-based) |
| `t_inst` | ndarray | Time vector |
| `wx` | ndarray | CWT coefficients |

---

## `trimming_signal(t, sig)`

### Description

Trim the signal according to detected peaks.

The number of peaks removed at the beginning and at the end is controlled through:

- `trim_pt_start`
- `trim_pt_end`

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `t` | ndarray | Time vector |
| `sig` | ndarray | Signal to trim |

### Uses

| Attribute | Type | Description |
|------------|------|-------------|
| `self.trim_pt_start` | int | Number of peaks removed at the beginning |
| `self.trim_pt_end` | int | Number of peaks removed at the end |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `out_t` | ndarray | Trimmed time vector |
| `out_sig` | ndarray | Trimmed and normalized signal |

---