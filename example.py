"""
Python program computing the CWT of a signal.

Anastasia MARÉCHAL
Mathieu MEZACHE
"""

import numpy as np
import matplotlib.pyplot as plt
import functions_InSync as sinc


def generate_test_signal(n_points=2000, dt=600,
                         signal_to_noise_ratio=1.5, boolshow=True):
    """Generate a test signal (harmonic oscillator with decaying amplitude and a parabolic trend)
    Args:
        n_points (int, optional): number of points of the signal. Defaults to 1000.
        dt (int, optional): time step of the signal. Defaults to 600 seconds.
        signal_to_noise_ratio (float, optional): ratio minimum signal amplitude 
                                    over noise amplitude.
                                    Lower values correspond to noisier signals. 
                                    Defaults to 1.5.
        boolshow (bool, optional): if True, the graphs are shown.
                                    Defaboolshow=False, save_csv=True,
        dir_save="./"ults to False.

    Returns:
        time (array): Time array in seconds.
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

    if boolshow:
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


if __name__ == "__main__":

    # Generate a test signal
    [t, real_sig, sig, real_trend, real_inst_freq, real_inst_amp] = (
        generate_test_signal()
    )

    # Parameters
    coeff_smoothing = 5 * 1e5
    num_peaks_start = 0
    num_peaks_end = 1
    sig_name = "Synthetic signal"

    # Launch the procedure
    analysis = sinc.InSyncPy(t=t, sig=sig, coeff=coeff_smoothing, trim_pt_start=num_peaks_start, trim_pt_end=num_peaks_end,
                                  sig_name=sig_name)
    results = analysis.full_analysis(show_plot=False, save_plot=False, save_csv=False, dir_save="./")

    # Extract analysis outputs
    t_denoised = results["t_denoised"]
    sig_denoised = results["signal_denoised"]
    est_trend = results["trend"]
    sig_detrended = results["signal_detrended"]
    t_trimmed = results["t_trimmed"]
    sig_trimmed = results["signal_trimmed"]

    t_inst = results["t_inst"]
    est_inst_freq = results["inst_freq"]
    inst_freq_sc = results["inst_freq_smoothed"]
    est_inst_amp = results["inst_amp"]
    inst_amp_sc = results["inst_amp_smoothed"]

    #####################################
    # Preprocessing vizualisation
    #####################################

    fig, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(
        f"Preprocessing of the signal ({sig_name})",
        fontsize=16,
    )
    # Denoising
    ax[0].plot(t, sig, "r-", label="Raw signal")
    ax[0].plot(t_denoised, sig_denoised, "k-", label="Denoised signal")
    ax[0].set_title("Denoised signal using Discrete Wavelet Transform (sym8)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(True)
    # Detrending
    ax[1].plot(t_denoised, sig_denoised, "r-", label="Denoised signal")
    ax[1].plot(t_denoised, est_trend, "k-", label="Estimated trend")
    ax[1].set_title("Trend approximated by smoothing splines")
    ax[1].set_ylabel("Amplitude")
    ax[1].grid(True)
    # Trimming
    ax[2].plot(t_denoised, sig_detrended, "r-", label="Normalized signal")
    ax[2].plot(t_trimmed, sig_trimmed, "k-", label="Trimmed signal")
    ax[2].axvline(x=t_trimmed[0], color="blue", linestyle="--", label="Trim points")
    ax[2].axvline(
        x=t_trimmed[-1],
        color="blue",
        linestyle="--",
    )
    ax[2].grid()
    ax[2].set_xlabel("Time (hours)")
    ax[2].set_title("Trimmed and normalized signal")
    ax[2].legend()
    plt.tight_layout()
    plt.show()

    #####################################
    # Results
    #####################################

    # Signal
    fig, ax = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    ax[0, 0].plot(t, real_sig, "r-", label="True signal without trend")
    ax[0, 0].plot(t_denoised, sig_detrended, "k-", label="Estimated detrended signal")
    ax[0, 0].set_title("Signal")
    ax[0, 0].set_ylabel("Amplitude")
    ax[0, 0].grid(True)
    ax[0, 0].legend()
    # Trend
    ax[0, 1].plot(t, real_trend, "r-", label="True trend")
    ax[0, 1].plot(t_denoised, est_trend, "k-", label="Estimated trend")
    ax[0, 1].set_title("Trend")
    ax[0, 1].set_ylabel("Amplitude")
    ax[0, 1].grid(True)
    ax[0, 1].legend()
    # Instantaneous frequency
    ax[1, 0].plot(t, real_inst_freq * 1e5, "r-", label="True frequency")
    ax[1, 0].plot(
        t_inst, est_inst_freq * 1e5, "k--", label="Estimated frequency"
    )
    ax[1, 0].plot(
        t_inst, inst_freq_sc * 1e5, "k-", label="Smoothed frequency estimate"
    )
    ax[1, 0].set_title("Instantaneous frequency")
    ax[1, 0].set_xlabel("Time (hours)")
    ax[1, 0].set_ylabel("Frequency (×10⁻⁵ Hz)")
    ax[1, 0].grid(True)
    ax[1, 0].legend()
    # Instantaneous amplitude
    ax[1, 1].plot(t, real_inst_amp, "r-", label="True amplitude")
    ax[1, 1].plot(t_inst, est_inst_amp, "k--", label="Estimated amplitude")
    ax[1, 1].plot(
        t_inst, inst_amp_sc, "k-", label="Smoothed amplitude estimate"
    )
    ax[1, 1].set_title("Instantaneous amplitude")
    ax[1, 1].set_xlabel("Time (hours)")
    ax[1, 1].set_ylabel("Amplitude")
    ax[1, 1].grid(True)
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()
