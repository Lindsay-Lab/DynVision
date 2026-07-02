"""
Efficient image noise implementations for PyTorch DataLoader usage.

This module provides optimized noise functions designed for use in PyTorch DataLoaders.
Each function takes a snr parameter (signal-to-noise ratio) for precise noise control.

Functions:
- salt_pepper_noise: Adds random salt (white) and pepper (black) pixels
- poisson_noise: Adds Poisson noise (photon noise simulation)
- uniform_noise: Adds uniformly distributed noise
- gaussian_noise: Adds zero-mean Gaussian noise with spatial correlation options
- phase_scrambled_noise: Adds spatially correlated Fourier phase-scrambled noise

Usage in DataLoader:
    from noise_module import salt_pepper_noise

    class NoisyDataset(Dataset):
        def __getitem__(self, idx):
            image = self.images[idx]  # shape: (C, H, W)
            image = image.unsqueeze(0)  # shape: (1, C, H, W)
            noisy_image = salt_pepper_noise(image, snr=10.0, seed=42)  # 10:1 SNR
            return noisy_image.squeeze(0), self.labels[idx]
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, Callable
import logging
from dynvision.utils import str_to_bool, alias_kwargs

logger = logging.getLogger(__name__)

__all__ = [
    "salt_pepper_noise",
    "poisson_noise",
    "uniform_noise",
    "gaussian_noise",
    "phase_scrambled_noise",
]


def _set_seed(seed: Optional[int], device: torch.device) -> None:
    """Set random seed for reproducible noise generation."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)


def _calculate_signal_power(images: torch.Tensor) -> torch.Tensor:
    """Calculate signal power (variance) of images."""
    return torch.var(images)


def _get_dtype_safe_max(dtype: torch.dtype) -> float:
    """
    Get maximum safe value for a given dtype to prevent overflow in clamp operations.

    Args:
        dtype: PyTorch dtype (e.g., torch.float32, torch.float16)

    Returns:
        Maximum safe value that can be represented in the dtype
    """
    if dtype == torch.float16:
        return 65000.0  # Slightly below float16 max (~65,504) for safety
    elif dtype == torch.bfloat16:
        return 3.3e38  # bfloat16 has same range as float32 but less precision
    else:  # float32, float64, etc.
        return 1e6  # Original safe value for higher precision types


def _validate_noise_parameters(
    snr: Optional[float] = None, ssnr: Optional[float] = None
) -> float:
    """
    Validate and convert noise parameters to SNR with robust floating-point handling.

    Args:
        snr: Signal-to-noise ratio (higher = less noise, inf = no noise)
        ssnr: Signal-to-signal-plus-noise ratio (0 = mainly noise, 1 = no noise)

    Returns:
        SNR value for internal use

    Raises:
        ValueError: If both or neither parameters are provided, or if values are invalid
    """
    EPSILON = 1e-10  # Tolerance for floating-point comparison

    if snr is not None and ssnr is not None:
        raise ValueError(
            "Cannot specify both 'snr' and 'ssnr' parameters. Choose one."
        )

    if snr is None and ssnr is None:
        raise ValueError("Must specify either 'snr' or 'ssnr' parameter.")

    if snr is not None:
        if snr < 0:
            raise ValueError("SNR must be non-negative")
        return snr

    if ssnr is not None:
        if not 0 <= ssnr <= 1:
            raise ValueError("SSNR must be between 0 and 1 (inclusive)")

        # Robust comparison for edge cases
        if abs(ssnr - 1.0) < EPSILON:  # Robust no-noise detection
            return float("inf")
        elif abs(ssnr - 0.0) < EPSILON:  # Robust all-noise detection
            return 0.0
        else:
            # Convert SSNR to SNR: SNR = SSNR / (1 - SSNR)
            return ssnr / (1 - ssnr)


def _is_no_noise_condition(internal_snr: float) -> bool:
    """Robust check for no-noise condition."""
    return torch.isinf(torch.tensor(internal_snr)) or internal_snr > 1000


def _standardize_noise_return(
    images: torch.Tensor,
    noise_state: Optional[dict] = None,
    cached_noise_state: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """Ensure consistent return types across all noise functions."""
    if cached_noise_state is not None:
        return images  # Always return tensor when using cache
    else:
        return images, (noise_state or {})  # Always return tuple when not cached


def handle_5d_input(noise_func: Callable) -> Callable:
    """
    Decorator to handle 5D input tensors for noise functions with temporal mode support.

    Args:
        noise_func: The noise function to apply to 4D tensors.

    Returns:
        A wrapped function that handles both 4D and 5D inputs with temporal modes.
    """

    def wrapper(
        images: torch.Tensor, *args, temporal_mode: str = "dynamic", **kwargs
    ) -> torch.Tensor:
        # Handle 4D tensors directly
        if images.dim() == 4:
            return noise_func(images, *args, **kwargs)

        if images.dim() != 5:
            raise RuntimeError(
                f"Expected 4D or 5D input, but got input of size: {images.shape}"
            )

        batch_size, time_steps, channels, height, width = images.shape

        if temporal_mode == "static":
            # Apply noise to first timestep, then replicate across time
            first_frame = images[:, 0]  # Shape: [batch, channels, height, width]
            noisy_first_frame = noise_func(first_frame, *args, **kwargs)

            # Handle case where noise function returns tuple (result, state)
            if isinstance(noisy_first_frame, tuple):
                noisy_first_frame = noisy_first_frame[0]

            # Replicate across all timesteps
            noisy_images = noisy_first_frame.unsqueeze(1).expand(
                -1, time_steps, -1, -1, -1
            )
            return noisy_images

        elif temporal_mode == "dynamic":
            # Apply noise dynamicly to each timestep (current behavior)
            if time_steps > 1:
                logger.warning(
                    f"{noise_func.__name__} applied with temporal_mode='dynamic' "
                    f"(time_steps={time_steps}). Processing each time step dynamicly."
                )

            noisy_frames = []
            for t in range(time_steps):
                frame_result = noise_func(images[:, t], *args, **kwargs)
                # Handle case where noise function returns tuple
                if isinstance(frame_result, tuple):
                    frame_result = frame_result[0]
                noisy_frames.append(frame_result)

            return torch.stack(noisy_frames, dim=1)

        elif temporal_mode == "correlated":
            # Placeholder for future correlated noise implementation
            # TODO: Implement smooth temporal transitions with varying parameters
            raise NotImplementedError(
                "Correlated temporal mode is not yet implemented. "
                "Use 'static' or 'dynamic' modes for now."
            )
        else:
            raise ValueError(
                f"Unknown temporal_mode: {temporal_mode}. "
                "Available modes: 'static', 'dynamic', 'correlated'"
            )

    return wrapper


@handle_5d_input
def salt_pepper_noise(
    images: torch.Tensor,
    snr: Optional[float] = None,
    ssnr: Optional[float] = 1.0,
    seed: Optional[int] = None,
    salt_vs_pepper: float = 0.5,
    cached_noise_state: Optional[dict] = None,
    temporal_mode: str = "dynamic",
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Apply salt and pepper noise to images with SNR or SSNR-based intensity control.

    Args:
        images: Input images tensor (B, C, H, W)
        snr: Signal-to-noise ratio (higher = less noise, inf = no noise)
        ssnr: Signal-to-signal-plus-noise ratio (0 = mainly noise, 1 = no noise)
        seed: Random seed for reproducibility
        salt_vs_pepper: Proportion of salt vs pepper (0.5 = equal)
        cached_noise_state: Optional cached noise state for consistent application
        temporal_mode: Temporal noise behavior - 'static', 'dynamic', or 'correlated'

    Returns:
        Noisy images tensor or tuple of (noisy_images, noise_state)
    """
    # Set default if neither specified
    if snr is None and ssnr is None:
        raise ValueError("no snr or ssnr value given!")

    # Convert to internal SNR
    internal_snr = _validate_noise_parameters(snr, ssnr)

    # Handle no-noise condition with robust detection
    if _is_no_noise_condition(internal_snr):
        return _standardize_noise_return(images, {}, cached_noise_state)

    noisy_images = images.clone()

    if cached_noise_state is not None:
        # Use cached masks
        noise_mask = cached_noise_state["noise_mask"]
        salt_mask = cached_noise_state["salt_mask"]
    else:
        # Calculate signal power
        signal_power = _calculate_signal_power(images)

        # For salt & pepper noise, we need to determine corruption probability
        # that achieves target SNR. The noise power comes from pixel value differences.
        img_min, img_max = images.min(), images.max()

        # Calculate average squared difference for corrupted pixels
        avg_salt_noise_power = (img_max - images).pow(2).mean()
        avg_pepper_noise_power = (img_min - images).pow(2).mean()
        avg_noise_power_per_pixel = (
            salt_vs_pepper * avg_salt_noise_power
            + (1 - salt_vs_pepper) * avg_pepper_noise_power
        )

        # Calculate corruption probability to achieve target SNR
        target_noise_power = signal_power / internal_snr
        corruption_prob = min(1.0, target_noise_power / avg_noise_power_per_pixel)

        # Generate masks
        _set_seed(seed, images.device)
        noise_mask = (
            torch.rand(images.shape, device=images.device, dtype=images.dtype)
            < corruption_prob
        )
        salt_mask = (
            torch.rand(images.shape, device=images.device, dtype=images.dtype)
            < salt_vs_pepper
        )

        # Return state for caching
        noise_state = {
            "noise_mask": noise_mask,
            "salt_mask": salt_mask,
            "function_type": "salt_pepper",
        }

    # Apply salt and pepper noise using masks
    salt_locations = noise_mask & salt_mask
    pepper_locations = noise_mask & (~salt_mask)

    noisy_images[salt_locations] = images.max()
    noisy_images[pepper_locations] = images.min()

    return _standardize_noise_return(noisy_images, noise_state, cached_noise_state)


@handle_5d_input
def poisson_noise(
    images: torch.Tensor,
    snr: Optional[float] = None,
    ssnr: Optional[float] = 1.0,
    seed: Optional[int] = None,
    cached_noise_state: Optional[dict] = None,
    temporal_mode: str = "dynamic",
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Apply Poisson noise to images with SNR or SSNR-based intensity control.

    Args:
        images: Input images tensor (B, C, H, W)
        snr: Signal-to-noise ratio (higher = less noise, inf = no noise)
        ssnr: Signal-to-signal-plus-noise ratio (0 = mainly noise, 1 = no noise)
        seed: Random seed for reproducibility
        cached_noise_state: Optional cached noise state for consistent application
        temporal_mode: Temporal noise behavior - 'static', 'dynamic', or 'correlated'

    Returns:
        Noisy images tensor or tuple of (noisy_images, noise_state)
    """
    # Set default if neither specified
    if snr is None and ssnr is None:
        raise ValueError("no snr or ssnr value given!")

    # Convert to internal SNR
    internal_snr = _validate_noise_parameters(snr, ssnr)

    # Handle no-noise condition with robust detection
    if _is_no_noise_condition(internal_snr):
        return _standardize_noise_return(
            images, {"function_type": "poisson"}, cached_noise_state
        )

    if cached_noise_state is not None:
        # Use cached noise pattern
        noise_pattern = cached_noise_state["noise_pattern"]
        noisy_images = images + noise_pattern
    else:
        # Generate new noise pattern with SNR control
        _set_seed(seed, images.device)

        # Calculate target noise power from SNR
        signal_power = _calculate_signal_power(images)
        target_noise_power = signal_power / internal_snr

        # For normalized images, shift to positive range for Poisson calculation
        # Use a larger shift to ensure all values are safely positive
        shift_amount = torch.abs(images.min()).item() + 0.1
        images_shifted = images + shift_amount
        images_positive = torch.clamp(images_shifted, min=0.1)

        # Use iterative approach to find correct scaling
        # Start with a reasonable lambda scaling based on target noise power
        lambda_scale = torch.sqrt(target_noise_power).clamp(min=0.01)

        # Generate Poisson noise with robust parameter checking
        lambda_param = images_positive * lambda_scale + 0.1

        # Ensure lambda_param is positive and finite (dtype-safe max to prevent overflow)
        max_value = _get_dtype_safe_max(images.dtype)
        lambda_param = torch.clamp(lambda_param, min=0.1, max=max_value)

        # Verify parameters before Poisson sampling
        if torch.any(torch.isnan(lambda_param)) or torch.any(
            torch.isinf(lambda_param)
        ):
            logger.warning(
                "Invalid lambda parameters detected, using fallback Gaussian noise"
            )
            # Fallback to Gaussian noise if Poisson parameters are invalid
            noise_pattern = torch.randn_like(images) * torch.sqrt(target_noise_power)
        else:
            poisson_samples = torch.poisson(lambda_param)

            # Convert to additive noise pattern
            noise_pattern = (poisson_samples - lambda_param) / torch.sqrt(
                lambda_param + 1e-8
            )

            # Scale to achieve target noise power
            actual_noise_power = torch.var(noise_pattern)
            if actual_noise_power > 1e-8:
                noise_pattern = noise_pattern * torch.sqrt(
                    target_noise_power / actual_noise_power
                )

        noisy_images = images + noise_pattern
        noise_state = {"noise_pattern": noise_pattern, "function_type": "poisson"}

    # Clamp to reasonable range (dtype-safe bounds)
    max_clamp = min(8.0, _get_dtype_safe_max(images.dtype))
    noisy_images = torch.clamp(noisy_images, -6.0, max_clamp)

    return _standardize_noise_return(noisy_images, noise_state, cached_noise_state)


@handle_5d_input
def uniform_noise(
    images: torch.Tensor,
    snr: Optional[float] = None,
    ssnr: Optional[float] = 1.0,
    seed: Optional[int] = None,
    cached_noise_state: Optional[dict] = None,
    temporal_mode: str = "dynamic",
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Apply uniform noise with SNR or SSNR-based intensity control.

    Args:
        images: Input images tensor (B, C, H, W)
        snr: Signal-to-noise ratio (higher = less noise, inf = no noise)
        ssnr: Signal-to-signal-plus-noise ratio (0 = mainly noise, 1 = no noise)
        seed: Random seed for reproducibility
        cached_noise_state: Optional cached noise state for consistent application
        temporal_mode: Temporal noise behavior - 'static', 'dynamic', or 'correlated'

    Returns:
        Noisy images tensor or tuple of (noisy_images, noise_state)
    """
    # Set default if neither specified
    if snr is None and ssnr is None:
        raise ValueError("no snr or ssnr value given!")

    # Convert to internal SNR
    internal_snr = _validate_noise_parameters(snr, ssnr)

    # Handle no-noise condition with robust detection
    if _is_no_noise_condition(internal_snr):
        return _standardize_noise_return(
            images, {"function_type": "uniform"}, cached_noise_state
        )

    if cached_noise_state is not None:
        # Use cached noise pattern
        noise = cached_noise_state["noise_pattern"]
    else:
        # Calculate target noise power from SNR
        signal_power = _calculate_signal_power(images)
        target_noise_power = signal_power / internal_snr

        # Generate uniform noise with target power
        _set_seed(seed, images.device)
        base_noise = (
            torch.rand(images.shape, device=images.device, dtype=images.dtype) - 0.5
        )

        # For uniform distribution, variance = (b-a)^2/12, where range is [a,b]
        # For [-0.5, 0.5], variance = 1/12, so std = 1/sqrt(12)
        uniform_std = 1.0 / torch.sqrt(torch.tensor(12.0))

        # Scale noise to achieve target power
        target_std = torch.sqrt(target_noise_power)
        noise = base_noise * (target_std / uniform_std)

        noise_state = {"noise_pattern": noise, "function_type": "uniform"}

    noisy_images = images + noise

    return _standardize_noise_return(noisy_images, noise_state, cached_noise_state)


@handle_5d_input
@alias_kwargs(spatialcorr="spatially_correlated")
def gaussian_noise(
    images: torch.Tensor,
    snr: Optional[float] = None,
    ssnr: Optional[float] = 1.0,
    seed: Optional[int] = None,
    spatially_correlated: bool = False,
    correlation_kernel_size: int = 7,
    cached_noise_state: Optional[dict] = None,
    temporal_mode: str = "dynamic",
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Apply zero-mean Gaussian noise with SNR or SSNR-based intensity control.

    Args:
        images: Input images tensor (B, C, H, W)
        snr: Signal-to-noise ratio (higher = less noise, inf = no noise)
        ssnr: Signal-to-signal-plus-noise ratio (0 = mainly noise, 1 = no noise)
        seed: Random seed for reproducibility
        spatially_correlated: If True, applies spatial smoothing to create correlated noise
        correlation_kernel_size: Size of Gaussian kernel for spatial correlation (odd number)
        cached_noise_state: Optional cached noise state for consistent application
        temporal_mode: Temporal noise behavior - 'static', 'dynamic', or 'correlated'

    Returns:
        Noisy images tensor or tuple of (noisy_images, noise_state)
    """
    spatially_correlated = str_to_bool(spatially_correlated)
    # Set default if neither specified
    if snr is None and ssnr is None:
        raise ValueError("no snr or ssnr value given!")

    # Convert to internal SNR
    internal_snr = _validate_noise_parameters(snr, ssnr)

    # Handle no-noise condition with robust detection
    if _is_no_noise_condition(internal_snr):
        return _standardize_noise_return(
            images, {"function_type": "gaussian"}, cached_noise_state
        )

    if cached_noise_state is not None:
        # Use cached noise pattern
        noise = cached_noise_state["noise_pattern"]
    else:
        # Calculate target noise power from SNR
        signal_power = _calculate_signal_power(images)
        target_noise_power = signal_power / internal_snr
        target_std = torch.sqrt(target_noise_power)

        # Generate Gaussian noise with target standard deviation
        _set_seed(seed, images.device)
        noise = torch.randn(images.shape, device=images.device, dtype=images.dtype)

        if spatially_correlated:
            # Apply spatial correlation via Gaussian smoothing
            if correlation_kernel_size % 2 == 0:
                correlation_kernel_size += 1  # Ensure odd kernel size

            # Create Gaussian kernel
            sigma = correlation_kernel_size / 6.0  # Standard heuristic
            kernel_1d = torch.exp(
                -0.5
                * (
                    (
                        torch.arange(
                            correlation_kernel_size,
                            dtype=torch.float32,
                            device=images.device,
                        )
                        - correlation_kernel_size // 2
                    )
                    / sigma
                )
                ** 2
            )
            kernel_1d = kernel_1d / kernel_1d.sum()

            # Create 2D kernel
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            kernel_2d = kernel_2d / kernel_2d.sum()

            # Apply correlation to each channel separately
            padding = correlation_kernel_size // 2
            noise_corr = F.conv2d(
                noise.view(-1, 1, images.shape[2], images.shape[3]),
                kernel_2d.unsqueeze(0).unsqueeze(0),
                padding=padding,
                groups=1,
            )
            noise = noise_corr.view(images.shape)

            # Renormalize to maintain target power
            actual_std = torch.std(noise)
            if actual_std > 1e-8:
                noise = noise * (target_std / actual_std)
        else:
            noise = noise * target_std

        noise_state = {"noise_pattern": noise, "function_type": "gaussian"}

    noisy_images = images + noise

    return _standardize_noise_return(noisy_images, noise_state, cached_noise_state)


@handle_5d_input
def phase_scrambled_noise(
    images: torch.Tensor,
    snr: Optional[float] = None,
    ssnr: Optional[float] = 1.0,
    seed: Optional[int] = None,
    use_image_spectrum: bool = False,
    spectral_exponent: float = -1.0,
    cached_noise_state: Optional[dict] = None,
    temporal_mode: str = "dynamic",
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Apply spatially correlated Fourier phase-scrambled noise.

    Creates noise by taking the average amplitude spectrum of input images,
    randomizing the phases, and performing inverse FFT. This preserves the
    original power spectrum (1/F-like) while creating cloud-like spatially
    correlated noise without coherent edges.

    Args:
        images: Input images tensor (B, C, H, W)
        snr: Signal-to-noise ratio (higher = less noise, inf = no noise)
        ssnr: Signal-to-signal-plus-noise ratio (0 = mainly noise, 1 = no noise)
        seed: Random seed for reproducibility
        use_image_spectrum: If True, use average amplitude from input images.
                           If False, use synthetic 1/F spectrum with spectral_exponent
        spectral_exponent: Power spectral exponent when use_image_spectrum=False
                          (-1.0 for pink noise, 0.0 for white noise)
        cached_noise_state: Optional cached noise state for consistent application
        temporal_mode: Temporal noise behavior - 'static', 'dynamic', or 'correlated'

    Returns:
        Noisy images tensor or tuple of (noisy_images, noise_state)
    """
    # Set default if neither specified
    if snr is None and ssnr is None:
        raise ValueError("no snr or ssnr value given!")

    # Convert to internal SNR
    internal_snr = _validate_noise_parameters(snr, ssnr)

    # Handle no-noise condition with robust detection
    if _is_no_noise_condition(internal_snr):
        return _standardize_noise_return(
            images, {"function_type": "phase_scrambled"}, cached_noise_state
        )

    if cached_noise_state is not None:
        # Use cached noise pattern
        noise = cached_noise_state["noise_pattern"]
    else:
        # Calculate target noise power from SNR
        signal_power = _calculate_signal_power(images)
        target_noise_power = signal_power / internal_snr

        _set_seed(seed, images.device)

        batch_size, channels, height, width = images.shape
        noise = torch.zeros_like(images)

        if use_image_spectrum:
            # Calculate average amplitude spectrum from input images
            # This follows the scientific method description
            amplitude_spectra = []

            for b in range(batch_size):
                for c in range(channels):
                    # Get FFT of original image
                    image_fft = torch.fft.fft2(images[b, c])
                    amplitude_spectra.append(torch.abs(image_fft))

            # Average amplitude spectrum across all images and channels
            avg_amplitude = torch.stack(amplitude_spectra).mean(dim=0)

            # Generate phase-scrambled noise for each image/channel
            for b in range(batch_size):
                for c in range(channels):
                    # Generate random phases uniformly distributed in [0, 2π)
                    random_phases = (
                        torch.rand(
                            height, width, device=images.device, dtype=images.dtype
                        )
                        * 2
                        * np.pi
                    )

                    # Create complex spectrum with average amplitude and random phases
                    # Make sure to create new tensors to avoid memory sharing issues
                    real_part = avg_amplitude * torch.cos(random_phases)
                    imag_part = avg_amplitude * torch.sin(random_phases)
                    scrambled_fft = torch.complex(real_part, imag_part)

                    # Ensure Hermitian symmetry for real-valued output
                    # This is important for getting real-valued noise after IFFT
                    scrambled_fft = _ensure_hermitian_symmetry(scrambled_fft)

                    # Convert back to spatial domain
                    phase_scrambled_noise_img = torch.fft.ifft2(scrambled_fft).real

                    noise[b, c] = phase_scrambled_noise_img
        else:
            # Fallback: Generate synthetic 1/F spectrum
            for b in range(batch_size):
                for c in range(channels):
                    # Generate white noise in frequency domain
                    real_noise = torch.randn(
                        height, width, device=images.device, dtype=images.dtype
                    )
                    imag_noise = torch.randn(
                        height, width, device=images.device, dtype=images.dtype
                    )
                    white_noise_fft = torch.complex(real_noise, imag_noise)

                    # Create frequency grid
                    freq_y = torch.fft.fftfreq(
                        height, d=1.0, device=images.device
                    ).unsqueeze(1)
                    freq_x = torch.fft.fftfreq(
                        width, d=1.0, device=images.device
                    ).unsqueeze(0)

                    # Compute radial frequency
                    freq_radial = torch.sqrt(freq_x**2 + freq_y**2)
                    freq_radial[0, 0] = 1e-8  # Avoid division by zero at DC

                    # Apply spectral shaping: S(f) ∝ f^exponent (1/F spectrum)
                    power_scaling = freq_radial ** (spectral_exponent / 2.0)

                    # Scale the white noise by the power spectrum
                    colored_noise_fft = white_noise_fft * power_scaling

                    # Ensure Hermitian symmetry
                    colored_noise_fft = _ensure_hermitian_symmetry(colored_noise_fft)

                    # Convert back to spatial domain
                    colored_noise_img = torch.fft.ifft2(colored_noise_fft).real

                    noise[b, c] = colored_noise_img

        # Normalize to target power
        actual_noise_power = torch.var(noise)
        if actual_noise_power > 1e-8:
            scaling_factor = torch.sqrt(target_noise_power / actual_noise_power)
            noise = noise * scaling_factor

        noise_state = {
            "noise_pattern": noise,
            "function_type": "phase_scrambled",
            "use_image_spectrum": use_image_spectrum,
        }

    noisy_images = images + noise

    return _standardize_noise_return(noisy_images, noise_state, cached_noise_state)


def _ensure_hermitian_symmetry(fft_tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure Hermitian symmetry for FFT tensor to guarantee real-valued IFFT output.

    For a real-valued signal, the FFT must satisfy: F(k) = F*(-k) where * denotes
    complex conjugate. This function enforces this property.

    Args:
        fft_tensor: Complex tensor of shape (H, W)

    Returns:
        Hermitian-symmetric complex tensor
    """
    height, width = fft_tensor.shape

    # Create a proper copy to modify (detach from computation graph and clone)
    symmetric_fft = fft_tensor.detach().clone()

    # Handle DC component (should be real)
    symmetric_fft[0, 0] = torch.complex(
        torch.real(symmetric_fft[0, 0]), torch.tensor(0.0, device=fft_tensor.device)
    )

    # Handle Nyquist frequencies (should be real for even dimensions)
    if height % 2 == 0:
        symmetric_fft[height // 2, 0] = torch.complex(
            torch.real(symmetric_fft[height // 2, 0]),
            torch.tensor(0.0, device=fft_tensor.device),
        )
        if width % 2 == 0:
            symmetric_fft[0, width // 2] = torch.complex(
                torch.real(symmetric_fft[0, width // 2]),
                torch.tensor(0.0, device=fft_tensor.device),
            )
            symmetric_fft[height // 2, width // 2] = torch.complex(
                torch.real(symmetric_fft[height // 2, width // 2]),
                torch.tensor(0.0, device=fft_tensor.device),
            )

    if width % 2 == 0:
        symmetric_fft[0, width // 2] = torch.complex(
            torch.real(symmetric_fft[0, width // 2]),
            torch.tensor(0.0, device=fft_tensor.device),
        )

    # Create a new tensor for the symmetric version to avoid in-place modifications
    result_fft = symmetric_fft.clone()

    # Enforce Hermitian symmetry: F(-k) = F*(k)
    for i in range(1, height):
        for j in range(1, width):
            # Get the symmetric indices
            sym_i = (height - i) % height
            sym_j = (width - j) % width

            # Only process upper triangle to avoid double processing
            if i < height // 2 or (i == height // 2 and j <= width // 2):
                # Set the symmetric element to be the complex conjugate
                if sym_i != i or sym_j != j:  # Avoid self-assignment
                    result_fft[sym_i, sym_j] = torch.conj(result_fft[i, j])

    return result_fft


# Update the visualization to show the new phase-scrambled noise
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torchvision.transforms as tv
    import os

    def create_ssnr_noise_visualization():
        """Create visualization showing noise progression with SSNR control."""
        print("Creating SSNR-based noise visualization...")

        # Try to load the specific test image
        test_image_path = "/home/rgutzen/01_PROJECTS/Modeling_Dynamical_Vision/data/interim/imagenette/test_all/n03394916/n03394916_24420.JPEG"

        if os.path.exists(test_image_path):
            print(f"Loading test image from: {test_image_path}")

            # Apply standard ImageNet preprocessing
            transform = tv.Compose(
                [
                    tv.Resize(256),
                    tv.CenterCrop(224),
                    tv.ToTensor(),
                ]
            )

            # Load and preprocess the image
            pil_image = Image.open(test_image_path).convert("RGB")
            test_image = transform(pil_image).unsqueeze(0)  # (1, 3, 224, 224)

            # Normalize to mean=0, std=1
            test_image_flat = test_image.reshape(-1)
            mean_val = test_image_flat.mean()
            std_val = test_image_flat.std()
            test_image = (test_image - mean_val) / (std_val + 1e-8)

            print(
                f"Loaded and normalized test image - Mean: {test_image.mean().item():.4f}, Std: {test_image.std().item():.4f}"
            )
        else:
            print(
                f"Test image not found at {test_image_path}, creating synthetic test image..."
            )
            # Create a test image with clear patterns (fallback)
            test_image = torch.zeros(1, 3, 224, 224)

            # Create a gradient background
            for i in range(224):
                test_image[0, :, i, :] = i / 223.0

            # Add geometric shapes for better noise visibility
            test_image[0, :, 40:80, 40:80] = 1.0  # White square
            test_image[0, :, 120:160, 120:200] = 0.5  # Gray rectangle

            # Black circle
            center_x, center_y = 180, 60
            for i in range(224):
                for j in range(224):
                    if (i - center_x) ** 2 + (j - center_y) ** 2 < 25**2:
                        test_image[0, :, i, j] = 0.0

            # Normalize the test image
            test_image_flat = test_image.reshape(-1)
            mean_val = test_image_flat.mean()
            std_val = test_image_flat.std()
            test_image = (test_image - mean_val) / (std_val + 1e-8)

        print(
            f"Value range: [{test_image.min().item():.2f}, {test_image.max().item():.2f}]"
        )

        # Noise functions (removed "Original")
        noise_functions = {
            "Salt & Pepper": salt_pepper_noise,
            "Poisson": poisson_noise,
            "Uniform": uniform_noise,
            "Gaussian": gaussian_noise,
            "Gaussian (Corr.)": lambda x, **kwargs: gaussian_noise(
                x, spatially_correlated=True, **kwargs
            ),
            "Phase Scrambled": lambda x, **kwargs: phase_scrambled_noise(
                x, use_image_spectrum=True, **kwargs
            ),
        }

        # SSNR values from no noise (1.0) to mainly noise (0.0)
        ssnr_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

        # Create figure
        fig, axes = plt.subplots(
            len(noise_functions), len(ssnr_values), figsize=(18, 10)
        )
        fig.suptitle(
            "Noise Types with SSNR-Based Control", fontsize=16, fontweight="bold"
        )

        # Column headers
        for j, ssnr in enumerate(ssnr_values):
            ssnr_label = "No Noise" if ssnr == 1.0 else f"SSNR: {ssnr}"
            axes[0, j].set_title(ssnr_label, fontsize=12, fontweight="bold")

        # Generate images
        for i, (noise_name, noise_func) in enumerate(noise_functions.items()):
            # Row label
            axes[i, 0].set_ylabel(noise_name, fontsize=12, fontweight="bold")

            for j, ssnr in enumerate(ssnr_values):
                if ssnr == 1.0:
                    # No noise case
                    result_image = test_image
                else:
                    try:
                        result = noise_func(test_image, ssnr=ssnr, seed=42)
                        if isinstance(result, tuple):
                            result_image, _ = result
                        else:
                            result_image = result
                    except Exception as e:
                        print(f"Error applying {noise_name} at SSNR {ssnr}: {e}")
                        result_image = test_image

                # Plot RGB image
                if result_image.shape[1] == 3:
                    img_np = result_image[0].detach().cpu().numpy().transpose(1, 2, 0)
                    img_display = (img_np - img_np.min()) / (
                        img_np.max() - img_np.min() + 1e-8
                    )
                    axes[i, j].imshow(img_display)
                else:
                    img_np = result_image[0, 0].detach().cpu().numpy()
                    img_display = (img_np - img_np.min()) / (
                        img_np.max() - img_np.min() + 1e-8
                    )
                    axes[i, j].imshow(img_display, cmap="gray", vmin=0, vmax=1)

                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

                # Add value range annotation
                min_val = result_image.min().item()
                max_val = result_image.max().item()
                axes[i, j].text(
                    5,
                    215,
                    f"[{min_val:.2f}, {max_val:.2f}]",
                    fontsize=7,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.8),
                )

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()

    # Run visualization
    create_ssnr_noise_visualization()

    print(f"\n{'='*50}")
    print("SSNR-BASED NOISE VISUALIZATION COMPLETED!")
    print(f"{'='*50}")
