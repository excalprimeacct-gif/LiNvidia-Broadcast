"""
Evaluation metrics for noise suppression quality assessment

Includes:
- SNR (Signal-to-Noise Ratio)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
"""

import numpy as np
from typing import Tuple, Dict


def calculate_snr(
    clean: np.ndarray,
    noisy: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB

    Args:
        clean: Clean signal
        noisy: Noisy signal
        epsilon: Small value to avoid division by zero

    Returns:
        SNR in dB
    """
    # Calculate noise
    noise = noisy - clean

    # Calculate powers
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    # SNR in dB
    snr = 10 * np.log10((signal_power / (noise_power + epsilon)) + epsilon)

    return float(snr)


def calculate_snr_improvement(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray
) -> float:
    """
    Calculate SNR improvement (delta SNR)

    Args:
        clean: Clean signal
        noisy: Noisy signal
        enhanced: Enhanced signal

    Returns:
        SNR improvement in dB
    """
    snr_input = calculate_snr(clean, noisy)
    snr_output = calculate_snr(clean, enhanced)

    return snr_output - snr_input


def calculate_pesq(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
    mode: str = 'wb'
) -> float:
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality)

    Args:
        reference: Reference (clean) signal
        degraded: Degraded (noisy/enhanced) signal
        sample_rate: Sample rate (8000 or 16000)
        mode: 'wb' (wideband) or 'nb' (narrowband)

    Returns:
        PESQ score (higher is better, range: -0.5 to 4.5)
    """
    try:
        from pesq import pesq as pesq_metric

        # Ensure correct sample rate
        if sample_rate not in [8000, 16000]:
            raise ValueError("PESQ requires sample rate of 8000 or 16000 Hz")

        # Ensure same length
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]

        # Calculate PESQ
        score = pesq_metric(sample_rate, reference, degraded, mode)

        return float(score)

    except ImportError:
        print("Warning: pesq package not installed. Install with: pip install pesq")
        return 0.0
    except Exception as e:
        print(f"Warning: PESQ calculation failed: {e}")
        return 0.0


def calculate_stoi(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
    extended: bool = False
) -> float:
    """
    Calculate STOI (Short-Time Objective Intelligibility)

    Args:
        reference: Reference (clean) signal
        degraded: Degraded (noisy/enhanced) signal
        sample_rate: Sample rate
        extended: Use extended STOI (better correlation with intelligibility)

    Returns:
        STOI score (range: 0 to 1, higher is better)
    """
    try:
        from pystoi import stoi as stoi_metric

        # Ensure same length
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]

        # Calculate STOI
        score = stoi_metric(reference, degraded, sample_rate, extended=extended)

        return float(score)

    except ImportError:
        print("Warning: pystoi package not installed. Install with: pip install pystoi")
        return 0.0
    except Exception as e:
        print(f"Warning: STOI calculation failed: {e}")
        return 0.0


def calculate_si_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: Reference signal
        estimate: Estimated signal
        epsilon: Small value to avoid division by zero

    Returns:
        SI-SDR in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    # Zero-mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    # Compute scaling factor
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + epsilon)

    # Project
    projection = alpha * reference

    # Noise
    noise = estimate - projection

    # SI-SDR
    si_sdr = 10 * np.log10(
        (np.sum(projection ** 2) + epsilon) / (np.sum(noise ** 2) + epsilon)
    )

    return float(si_sdr)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for noise suppression

    Calculates multiple metrics and provides summary statistics
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.reset()

    def reset(self):
        """Reset accumulated metrics"""
        self.snr_inputs = []
        self.snr_outputs = []
        self.snr_improvements = []
        self.pesq_scores = []
        self.stoi_scores = []
        self.si_sdr_scores = []

    def add_sample(
        self,
        clean: np.ndarray,
        noisy: np.ndarray,
        enhanced: np.ndarray
    ):
        """
        Add sample for evaluation

        Args:
            clean: Clean reference signal
            noisy: Noisy input signal
            enhanced: Enhanced output signal
        """
        # Resample if needed (for PESQ)
        if self.sample_rate not in [8000, 16000]:
            try:
                import resampy
                target_sr = 16000
                clean_resampled = resampy.resample(clean, self.sample_rate, target_sr)
                noisy_resampled = resampy.resample(noisy, self.sample_rate, target_sr)
                enhanced_resampled = resampy.resample(enhanced, self.sample_rate, target_sr)
            except ImportError:
                clean_resampled = clean
                noisy_resampled = noisy
                enhanced_resampled = enhanced
                target_sr = self.sample_rate
        else:
            clean_resampled = clean
            noisy_resampled = noisy
            enhanced_resampled = enhanced
            target_sr = self.sample_rate

        # SNR
        snr_input = calculate_snr(clean, noisy)
        snr_output = calculate_snr(clean, enhanced)
        snr_improvement = snr_output - snr_input

        self.snr_inputs.append(snr_input)
        self.snr_outputs.append(snr_output)
        self.snr_improvements.append(snr_improvement)

        # PESQ
        pesq_score = calculate_pesq(clean_resampled, enhanced_resampled, target_sr)
        if pesq_score > 0:
            self.pesq_scores.append(pesq_score)

        # STOI
        stoi_score = calculate_stoi(clean_resampled, enhanced_resampled, target_sr)
        if stoi_score > 0:
            self.stoi_scores.append(stoi_score)

        # SI-SDR
        si_sdr = calculate_si_sdr(clean, enhanced)
        self.si_sdr_scores.append(si_sdr)

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics

        Returns:
            Dictionary with mean values of all metrics
        """
        summary = {}

        if self.snr_inputs:
            summary['snr_input_mean'] = np.mean(self.snr_inputs)
            summary['snr_output_mean'] = np.mean(self.snr_outputs)
            summary['snr_improvement_mean'] = np.mean(self.snr_improvements)

        if self.pesq_scores:
            summary['pesq_mean'] = np.mean(self.pesq_scores)

        if self.stoi_scores:
            summary['stoi_mean'] = np.mean(self.stoi_scores)

        if self.si_sdr_scores:
            summary['si_sdr_mean'] = np.mean(self.si_sdr_scores)

        return summary

    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary()

        print("\n=== Evaluation Metrics Summary ===")
        print(f"Samples evaluated: {len(self.snr_inputs)}")

        if 'snr_input_mean' in summary:
            print(f"\nSNR:")
            print(f"  Input:       {summary['snr_input_mean']:.2f} dB")
            print(f"  Output:      {summary['snr_output_mean']:.2f} dB")
            print(f"  Improvement: {summary['snr_improvement_mean']:.2f} dB")

        if 'pesq_mean' in summary:
            print(f"\nPESQ: {summary['pesq_mean']:.2f}")

        if 'stoi_mean' in summary:
            print(f"STOI: {summary['stoi_mean']:.3f}")

        if 'si_sdr_mean' in summary:
            print(f"SI-SDR: {summary['si_sdr_mean']:.2f} dB")

        print()
