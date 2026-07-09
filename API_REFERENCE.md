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
    save_csv=False,
    dir_save='./'
)
```

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `t` | ndarray | Time vector |
| `sig` | ndarray | Input signal |
| `coeff` | float | Smoothing spline parameter |
| `trim_pt_start` | int | Number of peaks removed at the beginning |
| `trim_pt_end` | int | Number of peaks removed at the end |
| `sig_name` | str | Signal name |
| `save_csv` | bool | Export results to CSV |
| `dir_save` | str | Output directory |

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
| `self.sig_name` | str | Signal name |
| `self.save_csv` | bool | Export denoised signal |
| `self.dir_save` | str | Export directory |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `out_t` | ndarray | Time vector |
| `out` | ndarray | Denoised signal |

### Output

If `save_csv` = True, a CSV file is generated:

```text
<sig_name>_denoised.csv
```

| Column |
|----------|
| time |
| denoised_signal |

---

## `detrend_smoothing_spline(sig)`

### Description

Estimate and remove the signal trend using a smoothing spline.

The detrended signal is normalized by its maximum absolute value.

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `sig` | ndarray | Signal to detrend |

### Uses

| Attribute | Type | Description |
|------------|------|-------------|
| `self.t` | ndarray | Time vector |
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

### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `amplitude` | ndarray | Instantaneous amplitude |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `params` | ndarray | Estimated decay parameter |
| `err_eps` | ndarray | Standard deviation of the fitted parameter |

---

## `full_analysis()`

### Description

Execute the complete signal analysis pipeline.

### Workflow

1. Signal preparation
2. Continuous Wavelet Transform
3. Ridge extraction
4. Instantaneous frequency estimation
5. Instantaneous amplitude estimation
6. Scaleogram visualization
7. Smoothing of instantaneous quantities
8. Exponential decay fitting

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
| `t_inst_freq` | Time vector |
| `inst_freq` | Instantaneous frequency |
| `inst_period` | Instantaneous period |
| `inst_amp` | Instantaneous amplitude |
| `inst_freq_smoothed` | Smoothed frequency |
| `inst_amp_smoothed` | Smoothed amplitude |
| `decay_rate` | Estimated decay rate |
| `decay_error` | Uncertainty on decay rate |

---

### Output

If `save_csv` = True, two CSV file are generated:

```text
<sig_name>_inst_period_amplitude.csv
```

| Column |
|----------|
| time |
| inst_period |
| inst_amplitude |

and

```text
<sig_name>_metrics.csv
```

| Column |
|----------|
| decay_rate |
| mean_period |
| min_period |
| max_period|


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

## `scaleogram_visualisation(wx, freqs, t_inst_freq, inst_freq, inst_amp)`

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
| `t_inst_freq` | ndarray | Time vector |
| `inst_freq` | ndarray | Instantaneous frequency |
| `inst_amp` | ndarray | Instantaneous amplitude |

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `ener_amp_cwt` | ndarray | Temporal energy evolution |

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

### Output


If `save_csv` = True, a CSV file is generated:

```text
<sig_name>_denoised.csv
```

| Column |
|----------|
| time |
| detrended_signal |

---

## `sst_gmw_insync()`

### Description

Perform a synchrosqueezed Continuous Wavelet Transform (CWT) using Generalized Morse Wavelets (GMW) and extract the dominant ridge.

### Returns

| Output | Type | Description |
|----------|------|-------------|
| `tx` | ndarray | Synchrosqueezed transform |
| `freqs` | ndarray | Frequencies associated with scales |
| `inst_freq` | ndarray | Instantaneous frequency |
| `inst_amp` | ndarray | Instantaneous amplitude |
| `t_inst_freq` | ndarray | Time vector |
| `wx` | ndarray | CWT coefficients |

---

## `trimming_signal()`

### Description

Trim the signal according to detected peaks.

The number of peaks removed at the beginning and at the end is controlled through:

- `trim_pt_start`
- `trim_pt_end`

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