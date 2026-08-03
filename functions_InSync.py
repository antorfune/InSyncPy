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
        boolshow (bool): If True, enables plot visualization.
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

        t_inst_freq (array): Time vector associated with instantaneous features.
        inst_freq (array): Instantaneous frequency extracted from the CWT ridge.
        inst_freq_smoothed (array): Smoothed instantaneous frequency.
        inst_period (array): Instantaneous period.

        inst_amp (array): Instantaneous amplitude extracted from the CWT ridge.
        inst_amp_smoothed (array): Smoothed instantaneous amplitude.

        decay_rate (float): Estimated exponential decay rate.
        decay_error (float): Uncertainty on the estimated decay rate.
    """

    def __init__(self, t, sig,  coeff = 9 * 1e5, trim_pt_start = 0, trim_pt_end = 1,
                 sig_name = 'Synthetic signal', boolshow = False,
                 save_csv = False, dir_save = './'):

        self.t = t
        self.sig = sig
        self.coeff = coeff
        self.trim_pt_start = trim_pt_start
        self.trim_pt_end = trim_pt_end
        self.sig_name = sig_name
        self.boolshow = boolshow
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
            self.signal_name (str): Name of the signal file
            self.save_csv (bool): if True, save denoised signal to CSV.
                                        Defaults to False.
            self.dir_save (str): Directory path where the figure can be saved.
                                        If None, uses ./signal_name.


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

        signal_name = self.sig_name
        save_csv = self.save_csv
        dir_save = self.dir_save
        if save_csv:
            if dir_save is None:
                dir_save = "./"

            df = pd.DataFrame({"time": out_t, "denoised_signal": out})
            df.to_csv(os.path.join(dir_save, signal_name + "_denoised.csv"))
        return out_t, out


    def detrend_smoothing_spline(self, sig):
        """Return the detrended and normalized signal

        Uses class attributes:
            self.t (array): Time array
            self.coeff (float): Smoothing parameter

        Args: 
            sig (array): Trended signal

        Returns:
            out_sig (array): Normalized signal
            trend (array): Trend of the signal
        """

        coeff = self.coeff
        t = self.t
        spl = spii.make_smoothing_spline(t, sig, lam=coeff)
        trend = spl(t)

        signal_detrended = sig - trend
        signal_detrended += -np.mean(signal_detrended)

        out_sig = signal_detrended / np.max(np.abs(signal_detrended))

        return out_sig, trend


    def trimming_signal(self):
        """Return trimmed signal

        Uses class attributes:
            self.t (array): Time vector associated with the signal.
            self.sig (array): Input signal to be trimmed.
            self.trim_pt_start (int): Number of initial peaks to remove.
            self.trim_pt_end (int): Number of final peaks to remove.

        Returns:
            out_t (array): Time array of the trimmed signal
            out_sig (array): Trimmed signal
        """
        sig = self.sig

        # Supposed width of the peaks
        ind_width = 20 * 6
        peaks_ind, _ = spis.find_peaks(sig, distance=ind_width)

        t = self.t
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

        sig_detrended, trend = self.detrend_smoothing_spline(sig_denoised)

        save_csv = self.save_csv
        dir_save = self.dir_save
        if save_csv:
            if dir_save is None:
                dir_save = "./"

            df = pd.DataFrame({"time": self.t, "detrended_signal": sig_detrended})
            df.to_csv(os.path.join(dir_save, self.sig_name + "_detrended.csv"))

        self.sig = sig_detrended

        t_trimmed, sig_trimmed = self.trimming_signal()

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
            t_inst_freq (array): Time array associated with the instantaneous frequency
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
        t_inst_freq = t
        # Instantaneous amplitude as the ridge
        temp_wx = np.abs(wx)
        temp_amp = np.array(
            [temp_wx[ridge_idxs_amp[ie], ie] for ie in range(len(ridge_idxs_amp))]
        )
        inst_amp = temp_amp.ravel()

        return tx, freqs, inst_freq, inst_amp, t_inst_freq, wx

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
            lambda t, eps: self.exponential_decay(t,  eps, coeff),
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

        Uses class attributes:
            self.boolshow (bool): If True, plot the instantaneous amplitude
                                            as a function of time.
                                            Defaults to False.
        
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

        Uses class attributes:
            self.boolshow (bool): If True, plot the energy distribution as a
                function of frequency.
                Defaults to False.

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
            t_inst_freq,
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
        t_inst_freq (array): Time vector corresponding to the ridge-based
            instantaneous frequency estimates.
        inst_freq (array): Instantaneous frequency extracted from ridge
            detection (Hz).
        inst_amp (array): Instantaneous amplitude extracted from ridge
            detection.

        Returns:
            array: Temporal evolution of the energy of the signal.
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

        plt.figure(**figprops)

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
        t_ridge = t_inst_freq
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

        plt.show()

        return ener_amp_cwt

    def full_analysis(self):
        """
        Execute the full signal analysis pipeline.

        The pipeline includes the following steps:
            1. Signal preprocessing:
                - Denoising
                - Detrending
                - Trimming
            2. CWT analysis
            3. Extraction of instantaneous frequency and amplitude
            4. Scaleogram visualization
            5. Smoothing of instantaneous features
            6. Exponential decay fitting of amplitude envelope
        
        If 'save_csv' is enabled, two CSV files are generated:
            - One file containing instantaneous period and amplitude.
            - One file containing summary metrics (decay rate, mean period,
            minimum period, and maximum period).
        
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
    
            - 't_inst_freq' (array): Time vector associated with instantaneous features.
            - 'inst_freq' (array): Instantaneous frequency extracted from the CWT ridge.
            - 'inst_freq_smoothed' (array): Smoothed instantaneous frequency.
            - 'inst_period' (array): Instantaneous period.

            - 'inst_amp' (array): Instantaneous amplitude extracted from the CWT ridge.
            - 'inst_amp_smoothed' (array): Smoothed instantaneous amplitude.

            - 'decay_rate' (float): Estimated exponential decay rate.
            - 'decay_error' (float): Uncertainty on the estimated decay rate.
        """

        prep = self.signal_preparation()

        self.t = prep["t_trimmed"]
        self.sig = prep["signal_trimmed"]

        tx, freqs, inst_freq, inst_amp, t_inst_freq, wx = self.sst_gmw_insync()

        self.t = t_inst_freq

        self.scaleogram_visualisation(
            wx,
            freqs,
            t_inst_freq,
            inst_freq,
            inst_amp,
        )

        signal_name = self.sig_name

        self.sig_name = "Instantaneous frequency"
        inst_freq_sc = self.detrend_smoothing_spline(inst_freq)[1]
        inst_period = 1 / (inst_freq_sc * 3600)

        self.sig_name = "Instantaneous amplitude"
        inst_amp_sc = self.detrend_smoothing_spline(inst_amp)[1]

        decay, err_decay = self.fit_exp_decay(inst_amp_sc)
        mean_period = np.mean(inst_period)
        min_period = np.min(inst_period)
        max_period = np.max(inst_period)

        save_csv = self.save_csv
        dir_save = self.dir_save
        if save_csv:
            if dir_save is None:
                dir_save = "./"

            df = pd.DataFrame({"time": t_inst_freq, "inst_period (h)": inst_period, 
                               "inst_amp": inst_amp_sc})
            df.to_csv(os.path.join(dir_save, signal_name + "_inst_period_amplitude.csv"))

            df2 = pd.DataFrame({"decay": decay, "mean_period (h)": mean_period,
                                "min_period (h)": min_period, "max_period (h)": max_period})
            df2.to_csv(os.path.join(dir_save, signal_name + "_metrics.csv"))

        return {
            **prep, # signal denoised, detrended, trimmed, time array associated and trend
            "freqs": freqs,
            "wx": wx,
            "tx": tx,
            "t_inst_freq": t_inst_freq,
            "inst_freq": inst_freq,
            "inst_period": inst_period,
            "inst_amp": inst_amp,
            "inst_freq_smoothed": inst_freq_sc,
            "inst_amp_smoothed": inst_amp_sc,
            "decay_rate": decay,
            "decay_error": err_decay
        }
