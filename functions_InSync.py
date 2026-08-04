"""
Library containing the functions that preprocess a signal, compute its Continuous
Wavelet Transform (CWT), instantaneous frequency and amplitude.
Contains also the vizualisation tools.

Antoine FORTUNE
Anastasia MARECHAL
Mathieu MEZACHE
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spis
import scipy.interpolate as spii
import scipy.optimize as spio
import pandas as pd
from ssqueezepy import ssq_cwt
import ssqueezepy as ssq
from ssqueezepy.experimental import scale_to_freq
import pywt

def generate_test_signal(n_points=2000, dt=600,
                         signal_to_noise_ratio=1.5, show_plot=True):
    """Generate a test signal (harmonic oscillator with decaying amplitude and a parabolic trend)
    Args:
        n_points (int, optional): number of points of the signal. Defaults to 1000.
        dt (int, optional): time step of the signal. Defaults to 600 seconds.
        signal_to_noise_ratio (float, optional): ratio minimum signal amplitude 
                                    over noise amplitude.
                                    Lower values correspond to noisier signals. 
                                    Defaults to 1.5.
        show_plot (bool, optional): if True, the graphs are shown.
                                    Defaults to True. 

    Returns:
        time (array): Time array in hours.
        true_sig (array): Signal test without noise and trend.
        signal_data (array): Signal test noised and trended.
        trend (array): Trend of the signal test.
        inst_freq (array): Instantaneous frequency of the signal test.
        inst_amp (array): Instantaneous amplitude of the signal test.
    """

    time = np.arange(n_points) * dt

    # Base frequency
    base_freq = 0.75e-05

    # Time-varying frequency
    freq_variation = 0.75e-05 * np.exp(-2 * (1 / time[-1]) * time)  # decay
    inst_freq = base_freq + freq_variation

    # Time-varying amplitude with exponential decay
    decay_time = time[-1] / np.log(4)  # Decay over 3/4 of total time
    inst_amp = np.exp(-time / decay_time)

    # Generate signal
    phase = 2 * np.pi * np.cumsum(inst_freq * dt)
    signal_data = inst_amp * np.cos(phase + np.pi / 2)
    true_sig = np.copy(signal_data)
    # Add some noise
    sigma = 1
    noise_level = 1 / 4 * 1 / signal_to_noise_ratio
    signal_data += noise_level * np.random.normal(0, sigma, len(signal_data))

    # Add trend
    mu = time[int(len(time) / 2)]
    trend = ((time - mu) / mu) ** 2
    signal_data += trend

    if show_plot:
        figure, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        figure.suptitle(
            "Synthetic signal with corresponding instantaneous frequency "
            "and instantaneous amplitude",
            fontsize=16,
        )
        axes[0].plot(time / 3600, signal_data)
        axes[0].set_title("Signal")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)
        # Instantaneous frequency
        axes[1].plot(time / 3600, inst_freq * 1e5)
        axes[1].set_title("Instantaneous Frequency")
        axes[1].set_ylabel("Frequency (×10⁻⁵ Hz)")
        axes[1].grid(True)
        # Instantaneous amplitude
        axes[2].plot(time / 3600, inst_amp)
        axes[2].set_title("Instantaneous Amplitude")
        axes[2].set_xlabel("Time (hours)")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True)
        plt.tight_layout()
        plt.show()
    return time / 3600, true_sig, signal_data, trend, inst_freq, inst_amp

def _count_positive_peaks(signal: np.ndarray) -> int:
    """Return the number of positive peaks in a 1-D signal."""
    peaks, _ = spis.find_peaks(signal, height=0)
    return len(peaks)

class InSyncPy:
    """
    Signal processing and time-frequency analysis pipeline.

    The pipeline follows these steps:
        1. Signal preparation:
           - Denoising using discrete wavelet transform (DWT)
           - Smoothing spline detrending
           - Peak-based trimming
        2. Time-frequency analysis using Continuous Wavelet Transform (CWT)
        3. Extraction of instantaneous frequency and amplitude via ridge detection
        4. Visualization of the scaleogram and signal representations
        5. Post-processing:
           - Smoothing of instantaneous features
           - Exponential decay fitting of amplitude envelope
    
    Attributes:
        t (array): Time vector of the current processed signal.
        sig (array): Current working signal (updated through the pipeline).
        coeff (float): Smoothing parameter for the detrending.
        trim_pt_start (int): Number of initial peaks to remove.
        trim_pt_end (int): Number of final peaks to remove.
        sig_name (str): Name identifier for the signal.
        show_plot (bool): If True, enables plot visualization.
        save_plot (bool): If True, saves plots to jpg file.
        save_csv (bool): If True, saves results to CSV files. The results outputs are:
                            - the denoised signal,
                            - the detrended signal,
                            - the instantaneous frequency and amplitude,
                            - some metrics: decay, average period, minimale and maximale period.
        dir_save (str): Directory where outputs are saved.

    Results:
        t_denoised (array): Time vector after denoising.
        signal_denoised (array): Denoised signal.
        trend (array): Estimated trend removed during detrending.
        signal_detrended (array): Detrended signal.
        t_trimmed (array): Time vector after trimming.
        signal_trimmed (array): Trimmed signal.

        freqs (array): Frequency array associated with the wavelet scales.
        wx (array): CWT coefficients.
        tx (array): Synchrosqueezed CWT coefficients.

        t_inst (array): Time vector associated with instantaneous features.
        inst_freq (array): Instantaneous frequency extracted from the CWT ridge.
        inst_freq_smoothed (array): Smoothed instantaneous frequency.
        inst_period (array): Instantaneous period.

        inst_amp (array): Instantaneous amplitude extracted from the CWT ridge.
        inst_amp_smoothed (array): Smoothed instantaneous amplitude.

        amp_decay_rate (float): Estimated exponential decay rate of amplitude.
        amp_decay_error (float): Uncertainty on the amplitude estimated decay rate.
    """

    def __init__(self, t, sig,  coeff = 9 * 1e5, trim_pt_start = 0, trim_pt_end = 1,
                 sig_name = 'Synthetic signal', show_plot = True, save_plot = False,
                 save_csv = False, dir_save = './'):

        # Validate input arrays have same length
        if not isinstance(t, np.ndarray):
            raise TypeError("t must be a numpy array")
        if not isinstance(sig, np.ndarray):
            raise TypeError("sig must be a numpy array")
        if len(t) != len(sig):
            raise ValueError(
                f"Time vector (len={len(t)}) and signal (len={len(sig)}) must have the same length"
            )
        if len(t) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Validate arrays contain float values
        if t.dtype.kind != 'f':
            raise TypeError(f"Time vector must contain float values (unit: hour), got dtype={t.dtype}")
        if sig.dtype.kind != 'f':
            raise TypeError(f"Signal must contain float values, got dtype={sig.dtype}")

        # Validate coeff
        if not isinstance(coeff, (int, float)):
            raise TypeError(f"coeff must be numeric, got {type(coeff).__name__}")
        if coeff <= 0:
            raise ValueError(f"coeff must be positive, got {coeff}")

        # Validate trim points
        for name, val in (("trim_pt_start", trim_pt_start), ("trim_pt_end", trim_pt_end)):
            if not isinstance(val, int):
                raise TypeError(f"{name} must be an integer, got {type(val).__name__}")
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")
        if trim_pt_end <= 0:
            raise ValueError(f"trim_pt_end must be >= 1, got {trim_pt_end}")
        if trim_pt_start >= trim_pt_end:
            raise ValueError(
                f"trim_pt_start ({trim_pt_start}) must be less than trim_pt_end ({trim_pt_end})"
            )
        
        # Validate trim_pt_end does not exceed number of positive peaks
        n_peaks = _count_positive_peaks(sig)
        if n_peaks == 0:
            raise ValueError(
                "No positive peaks found in signal — cannot define trim points"
            )
        if trim_pt_end > n_peaks:
            raise ValueError(
                f"trim_pt_end ({trim_pt_end}) exceeds number of positive peaks ({n_peaks})"
            )

        # Validate strings
        for name, val in (("sig_name", sig_name), ("dir_save", dir_save)):
            if not isinstance(val, str):
                raise TypeError(f"{name} must be a string, got {type(val).__name__}")

        # Validate booleans
        for name, val in (
            ("show_plot", show_plot), 
            ("save_plot", save_plot), 
            ("save_csv", save_csv)
        ):
            if not isinstance(val, bool):
                raise TypeError(f"{name} must be a boolean, got {type(val).__name__}")

        self.t = t
        self.sig = sig
        self.coeff = coeff
        self.trim_pt_start = trim_pt_start
        self.trim_pt_end = trim_pt_end
        self.sig_name = sig_name
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.save_csv = save_csv
        self.dir_save = dir_save

    ####################################
    # Preprocessing denoise and detrend
    ####################################
    def denoise_dwt(self):
        """Return the denoised signal

        Uses class attributes:
            self.t (array): Time array
            self.signal (array): Noisy signal

        Returns:
            t_denoised (array):  Time array of the denoised signal
            out (array): Denoised signal
        """
        # Discrete Wavelet Transform
        wavelet = 'sym8'
        w = pywt.Wavelet(wavelet)

        t = self.t
        sig = self.sig

        res_lvl = pywt.dwt_max_level(len(t), w.dec_len)
        coeffs = pywt.wavedec(sig, wavelet, level=res_lvl)

        # Thresholding on the approximation coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 * 2
        uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
        denoised_coeffs = [pywt.threshold(c, value=uthresh, mode="hard") for c in coeffs]
        filtered_data_dwt = pywt.waverec(denoised_coeffs, wavelet, mode="smooth", axis=-1)

        if len(t) < len(filtered_data_dwt):
            out = filtered_data_dwt[:-1]
        else:
            out = filtered_data_dwt

        n = min(len(t), len(filtered_data_dwt))
        out = filtered_data_dwt[:n]
        out_t = t[:n]

        return out_t, out


    def detrend_smoothing_spline(self, t, sig):
        """Return the detrended and normalized signal

        Uses class attributes:
            self.coeff (float): Smoothing parameter

        Args: 
            sig (array): Trended signal

        Returns:
            out_sig (array): Normalized signal
            trend (array): Trend of the signal
        """

        coeff = self.coeff
        #t = self.t
        spl = spii.make_smoothing_spline(t, sig, lam=coeff)
        trend = spl(t)

        signal_detrended = sig - trend
        signal_detrended += -np.mean(signal_detrended)

        out_sig = signal_detrended / np.max(np.abs(signal_detrended))

        return out_sig, trend


    def trimming_signal(self, t, sig):
        """Return trimmed  and normalized signal 

        Uses class attributes:
            self.trim_pt_start (int): Number of initial peaks to remove.
            self.trim_pt_end (int): Number of final peaks to remove.

        Returns:
            out_t (array): Time array of the trimmed signal
            out_sig (array): Trimmed signal
        """
        #sig = self.sig

        # Supposed width of the peaks
        ind_width = 20 * 6
        peaks_ind, _ = spis.find_peaks(sig, distance=ind_width)

        #t = self.t
        num_peaks_start = self.trim_pt_start
        num_peaks_end = self.trim_pt_end

        if len(peaks_ind) <= 1:
            out_t = t
            out_sig = sig
        else:
            start_idx = (
                peaks_ind[num_peaks_start]
                if len(peaks_ind) > num_peaks_start
                else peaks_ind[0]
            )
            if num_peaks_end != 0:
                end_idx = (
                    peaks_ind[-num_peaks_end]
                    if len(peaks_ind) > num_peaks_end
                    else peaks_ind[-1]
                )
            else:
                end_idx = len(sig)

            # Avoid empty slices
            if start_idx >= end_idx:
                out_t = t
                out_sig = sig
            else:
                out_t = t[start_idx:end_idx]
                out_sig = sig[start_idx:end_idx]

        # Normalized signal
        out_sig = out_sig / np.max(np.abs(out_sig))
        return (out_t, out_sig)

    def signal_preparation(self):
        """ 
        Full preparation of the signal: denoising, detrending, trimming.
        If ``save_csv`` is True, the detrended signal is exported to a CSV file.

        Returns:
        dict: Preprocessing results containing:
            - t_denoised
            - sig_denoised
            - trend
            - sig_detrended
            - t_trimmed
            - sig_trimmed
        """
        t_denoised, sig_denoised = self.denoise_dwt()
        sig_detrended, trend = self.detrend_smoothing_spline(t_denoised, sig_denoised)
        # OMG !
        # self.sig = sig_detrended
        t_trimmed, sig_trimmed = self.trimming_signal(t_denoised, sig_detrended)

        return {
            "t_denoised": t_denoised,
            "signal_denoised": sig_denoised,
            "trend": trend,
            "signal_detrended": sig_detrended,
            "t_trimmed": t_trimmed,
            "signal_trimmed": sig_trimmed
        }


    ###############################
    # Continuous Wavelet Transform
    ###############################


    def sst_gmw_insync(self):
        """Return instantaneous amplitude and frequency

        Uses class attributes:
            self.t (array): Time array
            self.sig (array): Signal

        Returns:
            tx (array): Synchrosqueezed continuous wavelet transform
                of the input signal.
            freqs (array): Frequencies associated with the wavelet scales, in Hz.
            inst_freq (array): Estimated instantaneous frequency extracted from the
                dominant ridge of the wavelet transform.
            inst_amp (array): Estimated instantaneous amplitude extracted from the
                dominant ridge of the wavelet transform.
            t_inst (array): Time array associated with the instantaneous frequency
                and amplitude estimates.
            wx (array): Continuous wavelet transform coefficients.
        """
        sig = self.sig
        t = self.t
        N_sig = len(sig)

        # Parameters signal
        dfs = 1 / (t[1] - t[0]) / 3600  # Frequency sampling (t in Hours)
        # Parameters Generalized Morse Wavelets (GMW)
        padtype = "reflect"
        wavtype = (
            "gmw",
            {
                "gamma": 3,  # Frequency domain parameter
                "beta": 2,  # Time domain parameter
            },
        )

        # Continuous Wavelet Transform using GMW
        tx, wx, freqs, scales = ssq_cwt(
            sig, scales="log", wavelet=wavtype, fs=dfs, padtype=padtype
        )

        # Ridge extraction
        penalty = 1000
        n_ridges = 1
        bw = 15
        ridge_idxs = ssq.extract_ridges(
            np.abs(tx) ** 2,
            freqs,
            penalty=penalty,
            n_ridges=n_ridges,
            bw=bw,
            transform="cwt",
        )
        ridge_freq = freqs[ridge_idxs]

        freqs_amp = scale_to_freq(scales, 
                          wavtype,
                          N=N_sig, 
                          fs=dfs, 
                          padtype=padtype)

        ridge_idxs_amp = ssq.extract_ridges(
            np.abs(wx) ** 2,
            freqs_amp,
            penalty=penalty,
            n_ridges=n_ridges,
            bw=bw,
            transform="cwt",
        )

        inst_freq = ridge_freq.squeeze()
        #t_inst = t
        # Instantaneous amplitude as the ridge
        temp_wx = np.abs(wx)
        temp_amp = np.array(
            [temp_wx[ridge_idxs_amp[ie], ie] for ie in range(len(ridge_idxs_amp))]
        )
        inst_amp = temp_amp.ravel()

        return tx, freqs, inst_freq, inst_amp, t, wx

    #################################
    # Amplitude decay
    #################################

    def exponential_decay(self, time, decreasing_rate, coeff):
        """
        Exponential decay model.

        Args:
            t (ndarray): Time vector.
            decreasing_rate (float): Decay rate, 
                controls how fast the signal decreases.
            coeff (float): Initial amplitude.

        Returns:
            ndarray: Exponentially decaying signal: 
                coeff * exp(-decreasing_rate * t)
        """
        return coeff * np.exp(-decreasing_rate * time)

    def fit_exp_decay(self, amplitude):
        """
        Fit an exponential decay model to an instantaneous amplitude envelope.
        The model assumes:
            A(t) = A(t0) * exp(-epsilon t)
        where A(t0) is fixed to the first amplitude value and epsilon is estimated.

        Uses class attributes:
            self.t (ndarray): Time vector.

        Args: 
            amplitude (ndarray): Instantaneous amplitude of the signal.

        Returns:
            params (ndarray): Estimated decay parameter.
            err_eps (ndarray): Standard deviation of the estimated parameter.
        """
        
        dt = self.t[1] - self.t[0]

        t = np.arange(0, dt * len(self.t), dt)
        coeff = amplitude[0] # Fixed initial amplitude

        initial_guess = [0.001]

        params, covariance = spio.curve_fit(
            lambda t, eps: self.exponential_decay(t, eps, coeff),
            t,
            amplitude,
            p0=initial_guess,
            bounds=([0, 0.1]),
            method="trf"
        )

        err_eps = np.sqrt(np.diag(covariance))

        return params, err_eps


    #################################
    # Visualization
    #################################


    def get_inst_amplitude(self, matrix_coeff):
        """Estimate the instantaneous amplitude from a time-frequency representation.

        Args:
            matrix_coeff (array): Time-frequency coefficient matrix.

        Returns:
            out (array): Array containing the estimated
                instantaneous amplitude for each time sample.
        """
        temp = (np.abs(matrix_coeff)) ** 2
        n = len(matrix_coeff[:, 0])
        out = np.sum(temp, axis=0) / n
        return out


    def get_loc_freq(self, matrix_coeff):
        """Estimate frequency-localized energy distribution from a time-frequency representation.

        Args:
            Matrix_coeff (array): Time-frequency coefficient matrix.

        Returns:
            out (array): Array containing the energy associated
                with each frequency bin.
        """
        temp = (np.abs(matrix_coeff)) ** 2
        out = np.sum(temp, axis=1)

        return out


    def scaleogram_visualisation(
            self,
            wx,
            freqs,
            t_inst,
            inst_freq,
            inst_amp
    ):
        """
        This function generates the folowing figures:
        - the original signal,
        - the CWT power spectrum (scaleogram),
        - the global wavelet spectrum,
        - and the time evolution of instantaneous amplitude and frequency
        estimated from ridge extraction.

        Uses class attributes:
            self.t (array): Time array (in hours).
            self.sig (array): Signal.
        
        Args:
        wx (array): CWT coefficients.
        freqs (array): Frequencies associated with the wavelet scales (Hz).
        t_inst (array): Time vector corresponding to the ridge-based
            instantaneous frequency estimates.
        inst_freq (array): Instantaneous frequency extracted from ridge
            detection (Hz).
        inst_amp (array): Instantaneous amplitude extracted from ridge
            detection.

        Returns:
            fig: Matplotlib figure object.
        """
        t = self.t
        sig = self.sig
        # Create coordinate meshes
        x_coords = t  # Time in Hours
        y_coords = freqs * 1e5  # Log-spaced frequency scale
        x, y = np.meshgrid(x_coords, y_coords)

        # Get the global spectrum and Hour scale averaged power
        ener_amp_cwt = self.get_inst_amplitude(wx)
        ener_freq_cwt = self.get_loc_freq(wx)

        # Parameters for the plot

        units_spectrum = "Frequency (×10⁻⁵ Hz)"

        # Plot the Signal, scaleogram the power spectra
        # and the estimation of the instantaneous frequency
        plt.ioff()
        figprops = dict(figsize=(15, 12), dpi=102)

        fig = plt.figure(**figprops)

        # First sub-plot, the original time series
        ax = plt.axes([0.1, 0.75, 0.65, 0.2])
        ax.plot(x_coords, sig, "k", linewidth=1.5)
        ax.set_title("Normalized signal")
        ax.grid(True)

        # Second sub-plot, the normalized wavelet power spectrum and significance
        bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
        # Create pcolormesh plot
        bx.pcolormesh(x, y, (np.abs(wx)) ** 2, cmap="viridis", shading="auto")
        # Estimated Instantaneous Frequency
        ridge_f = inst_freq
        t_ridge = t_inst
        # Set log scale
        bx.set_yscale("log")
        bx.set_ylabel("Frequency ($\\times10^{-5}$ Hz)")
        bx.set_title('Scaleogram')

        # Third sub-plot, the global wavelet power spectra.
        cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
        cx.plot(ener_freq_cwt, y_coords, "k-", linewidth=1.5)
        cx.set_title("Global Wavelet Spectrum")
        cx.set_xlabel("Power")
        cx.grid(True)
        plt.setp(cx.get_yticklabels(), visible=True)

        # Fourth sub-plot, the scale averaged wavelet spectrum and
        # the instantaneous frequency.
        dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
        dx.plot(t_ridge, inst_amp, "k-", linewidth=1.5, label="Amplitude")
        dx.set_title("Hour scale-averaged power and instantaneous frequency estimation")
        dx.set_xlabel("Time (hour)")
        dx.set_ylabel("Power")
        # Instantaneous frequency in the COI

        dx2 = dx.twinx()
        dx2.set_ylim(
            np.array(
                [
                    np.min(ridge_f * 1e5) - 0.5 * np.min(ridge_f * 1e5),
                    np.max(ridge_f * 1e5) + 0.5 * np.max(ridge_f * 1e5),
                ]
            )
        )
        dx2.plot(t_ridge, ridge_f * 1e5, color="red", label="Estimated Freq.")
        dx2.plot(
            x_coords,
            np.mean(ridge_f * 1e5) * np.ones((len(x_coords),)),
            "r-.",
            label="Averaged Freq.",
        )
        dx2.spines["right"].set_color("red")
        dx2.set_ylabel(units_spectrum, color="red")
        dx2.set_yscale("log")
        dx2.tick_params(axis="y", colors="red")
        dx2.yaxis.label.set_color("red")
        dx2.legend()
        dx.grid(True)

        #plt.show()

        return fig

    def full_analysis(self):
        """
        Execute the full signal analysis pipeline.

        The pipeline includes the following steps:
            1. Signal preprocessing:
                - Denoising
                - Detrending
                - Normalizing
                - Trimming
            2. CWT analysis
            3. Extraction of instantaneous frequency and amplitude
            4. Smoothing of instantaneous features
            5. Exponential decay fitting of amplitude envelope
            6. Saving results to CSV files
                - signal processing
                - instantaneous features
                - summary metrics
            7. Plotting and saving figures
        
        If 'save_csv' is enabled, 4 CSV files are generated:
            - One file containing the signal preparation steps (denoising, trend, detrending)
            - One file containing the signal model (denoised, detrended, trimmed and normalized)
            - One file containing instantaneous period and amplitude.
            - One file containing summary metrics (period and amplitude descriptors).
        
        Returns:
        dict: Dictionary containing the preprocessing and analysis results.

            - 't_denoised' (array): Time vector after denoising.
            - 'signal_denoised' (array): Denoised signal.
            - 'trend' (array): Estimated trend removed during detrending.
            - 'signal_detrended' (array): Detrended signal.
            - 't_trimmed' (array): Time vector after trimming.
            - 'signal_trimmed' (array): Trimmed signal.

            - 'freqs' (array): Frequency array associated with the wavelet scales.
            - 'wx' (array): CWT coefficients.
            - 'tx' (array): Synchrosqueezed CWT coefficients.
    
            - 't_inst' (array): Time vector associated with instantaneous features.
            - 'inst_freq' (array): Instantaneous frequency extracted from the CWT ridge.
            - 'inst_freq_smoothed' (array): Smoothed instantaneous frequency.
            - 'inst_period' (array): Instantaneous period.

            - 'inst_amp' (array): Instantaneous amplitude extracted from the CWT ridge.
            - 'inst_amp_smoothed' (array): Smoothed instantaneous amplitude.

            - 'decay_rate' (float): Estimated exponential decay rate.
            - 'decay_error' (float): Uncertainty on the estimated decay rate.
        """

        prep = self.signal_preparation()

        # OMG !
        t = self.t
        sig = self.sig
        self.t = prep["t_trimmed"]
        self.sig = prep["signal_trimmed"]

        tx, freqs, inst_freq, inst_amp, t_inst, wx = self.sst_gmw_insync()

        # Instant frequency and amplitude smoothing
        inst_freq_sc = self.detrend_smoothing_spline(t_inst, inst_freq)[1]
        inst_period = 1 / (inst_freq_sc * 3600)
        inst_amp_sc = self.detrend_smoothing_spline(t_inst, inst_amp)[1]
        # Amplitude's decay rate estimation
        amp_decay, amp_decay_err = self.fit_exp_decay(inst_amp_sc)
        # Instantaneous Amplitude metrics
        mean_inst_amp = np.mean(inst_amp_sc)
        min_inst_amp = np.min(inst_amp_sc)
        max_inst_amp = np.max(inst_amp_sc)
        # Instentaneous Period metrics
        mean_inst_period = np.mean(inst_period)
        min_inst_period = np.min(inst_period)
        max_inst_period = np.max(inst_period)


        ###############
        # EXPORT TO CSV
        ###############

        if self.save_csv:
            if not os.path.exists(self.dir_save):
                os.makedirs(self.dir_save)
            
            df = pd.DataFrame({
                "time (hours)": prep["t_denoised"], 
                "denoised_signal": prep["signal_denoised"], 
                "signal_trend": prep["trend"],
                "detrended_signal": prep["signal_detrended"]
                })
            df.to_csv(os.path.join(self.dir_save, self.sig_name + "_detrended.csv"), index=False)

            df = pd.DataFrame({"time (hours)": prep["t_trimmed"], "signal_model": prep["signal_trimmed"]})
            df.to_csv( os.path.join(self.dir_save, self.sig_name + "_model.csv"), index=False)

            df = pd.DataFrame({"time": t_inst, "inst_period (h)": inst_period, 
                               "inst_amp": inst_amp_sc})
            df.to_csv(os.path.join(self.dir_save, self.sig_name + "_inst_period_amplitude.csv"), index=False)

            df = pd.DataFrame({
                "min_inst_amp": min_inst_amp, 
                "mean_inst_amp": mean_inst_amp, 
                "max_inst_amp": max_inst_amp, 
                "amp_decay": amp_decay,
                "amp_decay_err": amp_decay_err, 
                "min_inst_period (h)": min_inst_period, 
                "mean_inst_period (h)": mean_inst_period,
                "max_inst_period (h)": max_inst_period})
            df.to_csv(os.path.join(self.dir_save, self.sig_name + "_metrics.csv"), index=False)

        ###############
        # PLOT ANALYSIS
        ###############

        if (self.show_plot or self.save_plot):
        
        # signal processing plots
            fig_processing, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            fig_processing.suptitle(
                f"Preprocessing of signal {self.sig_name}",
                fontsize=16,
            )
            # Denoising
            ax[0].plot(t, sig, "r-", label="Raw signal")
            ax[0].plot(prep["t_denoised"], prep["signal_denoised"], "k-", label="Denoised signal")
            ax[0].set_title("Denoised signal using Discrete Wavelet Transform (sym8)")
            ax[0].set_ylabel("Amplitude")
            ax[0].grid(True)
            ax[0].legend()
            # Detrending
            ax[1].plot(t, prep["signal_denoised"], "k-", label="Denoised signal")
            ax[1].plot(t, prep["trend"], "r-", label="Estimated trend")
            ax[1].set_title("Trend approximated by smoothing splines")
            ax[1].set_ylabel("Amplitude")
            ax[1].grid(True)
            #ax[1].legend()
            # Trimming
            ax[2].plot(t, prep["signal_detrended"], "r-", label="Detrended normalized")
            ax[2].plot(prep["t_trimmed"], prep["signal_trimmed"], "k-", label="Trimmed")
            ax[2].axvline(x=prep["t_trimmed"][0], color="blue", linestyle="--", label="Trim points")
            ax[2].axvline(x=prep["t_trimmed"][-1], color="blue", linestyle="--")
            ax[2].grid()
            ax[2].set_xlabel("Time (hours)")
            ax[2].set_title("Detrended, normalized and trimmed signal")
            ax[2].legend()
            plt.tight_layout()

            #if self.show_plot:
            #    plt.show()
            
            # Signal power plots
            fig_scaleogram = self.scaleogram_visualisation(
                wx, 
                freqs,
                t_inst,
                inst_freq,
                inst_amp,
            )

            #if self.show_plot:
            #    plt.show()

            # Signal analysis plots
            fig_analysis, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

            ax[0].plot(prep["t_trimmed"], prep["signal_trimmed"], "k-", label="Processed signal")
            ax[0].set_title(self.sig_name + " modelled signal")
            ax[0].set_ylabel("Amplitude")
            ax[0].grid(True)
            ax[0].legend()
                
            # Instantaneous frequency
            ax[1].plot(
                t_inst, inst_freq * 1e5, "r--", label="Estimated frequency"
            )
            ax[1].plot(
                t_inst, inst_freq_sc * 1e5, "k-", label="Smoothed frequency estimate"
            )
            ax[1].set_title("Instantaneous frequency")
            ax[1].set_xlabel("Time (hours)")
            ax[1].set_ylabel("Frequency (×10⁻⁵ Hz)")
            ax[1].grid(True)
            ax[1].legend()
            # Instantaneous amplitude
            #ax[1, 1].plot(t, real_inst_amp, "r-", label="True amplitude")
            ax[2].plot(t_inst, inst_amp, "r--", label="Estimated amplitude")
            ax[2].plot(
                t_inst, inst_amp_sc, "k-", label="Smoothed amplitude estimate"
            )
            ax[2].set_title("Instantaneous amplitude")
            ax[2].set_xlabel("Time (hours)")
            ax[2].set_ylabel("Amplitude")
            ax[2].grid(True)
            ax[2].legend()
            plt.tight_layout()
            
            if self.show_plot:
                plt.show()

        ##############
        # EXPORT PLOTS
        ##############

        if self.save_plot:
            if not os.path.exists(self.dir_save):
                os.makedirs(self.dir_save)

            fig_processing.savefig(
                os.path.join(self.dir_save, self.sig_name) + "_processing.jpg", 
                format="jpg", 
                dpi=300, 
                bbox_inches="tight"
            )
            fig_scaleogram.savefig(
                os.path.join(self.dir_save, self.sig_name) + "_scaleogram.jpg", 
                format="jpg", 
                dpi=300, 
                bbox_inches="tight"
            )
            fig_analysis.savefig(
                os.path.join(self.dir_save, self.sig_name) + "_analysis.jpg", 
                format="jpg", 
                dpi=300, 
                bbox_inches="tight"
            )

        plt.close() 

        return {
            **prep, # signal denoised, detrended, trimmed, time array associated and trend
            "freqs": freqs,
            "wx": wx,
            "tx": tx,
            "t_inst": t_inst,
            "inst_freq": inst_freq,
            "inst_period": inst_period,
            "inst_amp": inst_amp,
            "inst_freq_smoothed": inst_freq_sc,
            "inst_amp_smoothed": inst_amp_sc,
            "amp_decay_rate": amp_decay,
            "amp_decay_error": amp_decay_err
        }


############
# DEMO START
############

if __name__ == "__main__":

    # Generate a test signal
    [t, real_sig, sig, real_trend, real_inst_freq, real_inst_amp] = generate_test_signal()

    # Parameters
    coeff_smoothing = 5 * 1e5
    num_peaks_start = 0
    num_peaks_end = 1
    sig_name = "Synthetic signal"

    # Launch the procedure
    model= InSyncPy(t, sig, coeff_smoothing, num_peaks_start, num_peaks_end,
                                  sig_name, show_plot=True, save_plot=False, save_csv=False, dir_save="./")
    model.full_analysis()
