"""Data loading

Usage:
    from dynvision.data.dataloader import get_data_loader

    loader = get_data_loader(
        dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )
"""

import logging
from typing import Any, Optional, Tuple
from collections import OrderedDict
import torch
import torch.jit
from torch.utils.data import DataLoader

from dynvision.data.operations import (
    _adjust_data_dimensions,
    _repeat_over_time,
    _adjust_label_dimensions,
)
from dynvision.data import noise
from dynvision.utils import alias_kwargs, filter_kwargs

logger = logging.getLogger(__name__)


# JIT-compiled helper functions
@torch.jit.script
def _fill_tensor_period(
    output_data: torch.Tensor,
    output_labels: torch.Tensor,
    input_data: torch.Tensor,
    input_labels: torch.Tensor,
    start_idx: int,
    duration: int,
) -> int:
    """JIT-compiled function for filling any tensor period (intro, stimulus, outro)."""
    end_idx = start_idx + duration
    output_data[:, start_idx:end_idx] = input_data
    output_labels[:, start_idx:end_idx] = input_labels
    return end_idx


@torch.jit.script
def _expand_tensor_optimized(
    tensor: torch.Tensor, repeat_dim: int, n_repeats: int
) -> torch.Tensor:
    """JIT-compiled optimized tensor expansion."""
    shape = list(tensor.shape)
    shape[repeat_dim] = n_repeats
    return tensor.expand(shape)


class StandardDataLoader(DataLoader):
    """Base data loader with performance optimizations.

    Performance Tips:
    - Use pin_memory=True for faster GPU transfer
    - Set num_workers based on CPU cores
    - Enable prefetch_factor for background loading
    - Use memory_format=torch.channels_last for GPU optimization
    - Enable mixed precision with dtype=torch.float16
    """

    @alias_kwargs(data_timesteps="n_timesteps")
    def __init__(
        self,
        *args,
        n_timesteps: int = 1,
        memory_format: torch.memory_format = torch.contiguous_format,
        dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
        use_channels_last: bool = False,
        use_cuda_streams: bool = True,
        stream_priority: int = 0,
        max_cache_size: int = 100,
        **kwargs,
    ):
        kwargs, _ = filter_kwargs(DataLoader, kwargs)

        super().__init__(*args, **kwargs)
        self.n_timesteps = int(n_timesteps)
        self.memory_format = memory_format
        self.dtype = dtype
        self.device = device
        self.use_channels_last = use_channels_last
        self.use_cuda_streams = use_cuda_streams
        self.max_cache_size = max_cache_size

        # Pre-compute optimal memory format based on data dimensions
        self._optimal_memory_format = (
            torch.channels_last if use_channels_last else torch.contiguous_format
        )

        # Initialize CUDA streams for overlapped computation
        if self.use_cuda_streams and torch.cuda.is_available():
            self._data_stream = torch.cuda.Stream(priority=stream_priority)
            self._copy_stream = torch.cuda.Stream(priority=stream_priority)
        else:
            self._data_stream = None
            self._copy_stream = None

        # Initialize caches with LRU eviction
        self._data_cache = {}
        self._label_cache = {}
        self._cache_access_order = OrderedDict()

    def _get_cached_tensors(self, data_shape, label_shape, device, dtype):
        """Get or create cached pre-allocated tensors with LRU eviction."""
        # Include memory format in cache key for proper separation
        cache_key = (
            data_shape,
            label_shape,
            device,
            dtype,
            self._optimal_memory_format,
        )

        # Update access order
        if cache_key in self._cache_access_order:
            self._cache_access_order.move_to_end(cache_key)
        else:
            self._cache_access_order[cache_key] = True

        if cache_key not in self._data_cache:
            # Evict old entries if cache is full
            if len(self._data_cache) >= self.max_cache_size:
                oldest_key = next(iter(self._cache_access_order))
                del self._data_cache[oldest_key]
                del self._label_cache[oldest_key]
                del self._cache_access_order[oldest_key]

            # Pre-allocate output tensors
            output_data_shape = (data_shape[0], self.n_timesteps, *data_shape[2:])
            output_label_shape = (label_shape[0], self.n_timesteps)

            # Create tensor with optimal memory layout (without pin_memory)
            self._data_cache[cache_key] = torch.empty(
                output_data_shape,
                dtype=dtype,
                device=device,
                memory_format=self._optimal_memory_format,
            )

            # For 4D+ tensors, ensure channels_last format if requested
            if len(output_data_shape) >= 4 and self.use_channels_last:
                self._data_cache[cache_key] = self._data_cache[cache_key].to(
                    memory_format=torch.channels_last
                )

            self._label_cache[cache_key] = torch.empty(
                output_label_shape, dtype=torch.long, device=device
            )

        return self._data_cache[cache_key], self._label_cache[cache_key]

    def _optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to optimal memory layout for GPU processing."""
        if tensor.dim() >= 4 and self.use_channels_last:
            return tensor.to(memory_format=torch.channels_last)
        return tensor.contiguous()

    def _prefetch_to_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        """Asynchronously transfer tensor to device using copy stream."""
        if self._copy_stream is not None and device:
            with torch.cuda.stream(self._copy_stream):
                return tensor.to(device, non_blocking=True)
        return tensor.to(device) if device else tensor

    def _async_tensor_operations(
        self,
        data,
        label_indices,
        output_data,
        output_labels,
        non_input_value,
        non_label_index,
    ):
        """Perform tensor operations asynchronously using CUDA streams."""
        if self._data_stream is not None:
            with torch.cuda.stream(self._data_stream):
                # Optimize data layout asynchronously
                data = self._optimize_tensor_layout(data)
                label_indices = self._optimize_tensor_layout(label_indices)

                # Pre-fill output tensors with default values
                output_data.fill_(non_input_value)
                output_labels.fill_(non_label_index)

            # Synchronize before returning
            self._data_stream.synchronize()
        else:
            # Fallback for CPU or when streams are disabled
            data = self._optimize_tensor_layout(data)
            label_indices = self._optimize_tensor_layout(label_indices)
            output_data.fill_(non_input_value)
            output_labels.fill_(non_label_index)

        return data, label_indices

    def __iter__(self):
        try:
            for sample in super(StandardDataLoader, self).__iter__():
                data, label_indices, *extra = sample

                label_indices = _adjust_label_dimensions(label_indices)
                data = _adjust_data_dimensions(data, self._optimal_memory_format)

                if self.n_timesteps > 1:
                    data = _repeat_over_time(data, self.n_timesteps)
                    label_indices = _repeat_over_time(label_indices, self.n_timesteps)

                yield [data, label_indices, *extra]

        except Exception as e:
            logger.error(f"Error in data loading: {str(e)}")
            raise


class StimulusRepetitionDataLoader(StandardDataLoader):
    @alias_kwargs(repeat="n_timesteps")
    def __init__(self, *args, n_timesteps=20, **kwargs):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)


class StimulusDurationDataLoader(StandardDataLoader):
    """DataLoader for stimulus duration experiments with performance optimizations."""

    @alias_kwargs(
        tsteps="n_timesteps",
        dsteps="n_timesteps",
        data_timesteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        voidid="non_label_index",
        voidinput="non_input_value",
    )
    def __init__(
        self,
        *args,
        n_timesteps: int = 30,
        stimulus_duration: int = 20,
        intro_duration: int = 1,
        non_label_index: int = -1,
        non_input_value: float = 0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)

        # Validate and store parameters
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.non_label_index = int(non_label_index)
        self.non_input_value = float(non_input_value)
        self.outro_duration = (
            self.n_timesteps - self.stimulus_duration - self.intro_duration
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough timesteps for stimulus presentation! "
                f"(n_timesteps={self.n_timesteps}, "
                f"stimulus_duration={self.stimulus_duration}, "
                f"intro_duration={self.intro_duration})"
            )

        # Pre-compile JIT functions during initialization
        self._fill_period_jit = _fill_tensor_period
        self._expand_jit = _expand_tensor_optimized

    def __iter__(self):
        try:
            for sample in DataLoader.__iter__(self):
                data, label_indices, *extra = sample

                # Asynchronous device transfer
                if self.device:
                    data = self._prefetch_to_device(data, self.device)
                    label_indices = self._prefetch_to_device(
                        label_indices, self.device
                    )

                # Apply performance optimizations
                data = _adjust_data_dimensions(data, self._optimal_memory_format)
                data = self._optimize_tensor_layout(data)

                if isinstance(data, torch.Tensor):
                    data = data.to(dtype=self.dtype)

                label_indices = _adjust_label_dimensions(label_indices)

                # Get pre-allocated tensors
                output_data, output_labels = self._get_cached_tensors(
                    data.shape, label_indices.shape, data.device, data.dtype
                )

                # Async tensor operations and pre-fill with void values
                data, label_indices = self._async_tensor_operations(
                    data,
                    label_indices,
                    output_data,
                    output_labels,
                    self.non_input_value,
                    self.non_label_index,
                )

                # Use JIT-compiled functions for critical path
                time_idx = 0

                # Skip intro period since output is already pre-filled with void values
                time_idx += self.intro_duration

                # Stimulus period (JIT-compiled)
                if self.stimulus_duration > 0:
                    expanded_data = self._expand_jit(data, 1, self.stimulus_duration)
                    expanded_labels = self._expand_jit(
                        label_indices, 1, self.stimulus_duration
                    )

                    time_idx = self._fill_period_jit(
                        output_data,
                        output_labels,
                        expanded_data,
                        expanded_labels,
                        time_idx,
                        self.stimulus_duration,
                    )

                # Skip outro period since output is already pre-filled with void values

                # Return without cloning for zero-copy (safe since we use pre-allocated tensors)
                yield [output_data, output_labels, *extra]

        except Exception as e:
            logger.error(f"Error in stimulus duration loading: {str(e)}")
            raise


class StimulusIntervalDataLoader(StandardDataLoader):
    """DataLoader for stimulus interval experiments with performance optimizations."""

    @alias_kwargs(
        tsteps="n_timesteps",
        dsteps="n_timesteps",
        data_timesteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        interval="interval_duration",
        voidid="non_label_index",
        voidinput="non_input_value",
    )
    def __init__(
        self,
        *args,
        n_timesteps: int = 30,
        stimulus_duration: int = 2,
        intro_duration: int = 0,
        interval_duration: int = 2,
        non_label_index: int = -1,
        non_input_value: float = 0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)

        # Validate and store parameters
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.interval_duration = int(interval_duration)
        self.non_label_index = int(non_label_index)
        self.non_input_value = float(non_input_value)
        self.outro_duration = (
            self.n_timesteps
            - 2 * self.stimulus_duration
            - self.interval_duration
            - self.intro_duration
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough timesteps for stimulus sequence! "
                f"(n_timesteps={self.n_timesteps}, "
                f"stimulus_duration={self.stimulus_duration} x2, "
                f"intro_duration={self.intro_duration}, "
                f"interval_duration={self.interval_duration})"
            )

        # Pre-compile JIT functions
        self._fill_period_jit = _fill_tensor_period
        self._expand_jit = _expand_tensor_optimized

    def __iter__(self):
        try:
            for sample in DataLoader.__iter__(self):
                data, label_indices, *extra = sample

                # Asynchronous device transfer
                if self.device:
                    data = self._prefetch_to_device(data, self.device)
                    label_indices = self._prefetch_to_device(
                        label_indices, self.device
                    )

                # Apply performance optimizations
                data = _adjust_data_dimensions(data, self._optimal_memory_format)
                data = self._optimize_tensor_layout(data)

                if isinstance(data, torch.Tensor):
                    data = data.to(dtype=self.dtype)

                label_indices = _adjust_label_dimensions(label_indices)

                # Get pre-allocated tensors
                output_data, output_labels = self._get_cached_tensors(
                    data.shape, label_indices.shape, data.device, data.dtype
                )

                # Async tensor operations and pre-fill with void values
                data, label_indices = self._async_tensor_operations(
                    data,
                    label_indices,
                    output_data,
                    output_labels,
                    self.non_input_value,
                    self.non_label_index,
                )

                # Pre-expand data once for reuse
                expanded_data = self._expand_jit(data, 1, self.stimulus_duration)
                expanded_labels = self._expand_jit(
                    label_indices, 1, self.stimulus_duration
                )

                # Use JIT-compiled functions for critical path
                time_idx = self.intro_duration  # Skip intro (pre-filled)

                # First stimulus period
                if self.stimulus_duration > 0:
                    time_idx = self._fill_period_jit(
                        output_data,
                        output_labels,
                        expanded_data,
                        expanded_labels,
                        time_idx,
                        self.stimulus_duration,
                    )

                # Skip interval period (pre-filled)
                time_idx += self.interval_duration

                # Second stimulus period (reuse expanded tensors)
                if self.stimulus_duration > 0:
                    self._fill_period_jit(
                        output_data,
                        output_labels,
                        expanded_data,
                        expanded_labels,
                        time_idx,
                        self.stimulus_duration,
                    )

                yield [output_data, output_labels, *extra]

        except Exception as e:
            logger.error(f"Error in stimulus interval loading: {str(e)}")
            raise


class StimulusContrastDataLoader(StandardDataLoader):
    @alias_kwargs(
        tsteps="n_timesteps",
        dsteps="n_timesteps",
        data_timesteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        contrast="stimulus_contrast",
        voidid="non_label_index",
        voidinput="non_input_value",
    )
    def __init__(
        self,
        *args,
        n_timesteps=15,
        stimulus_duration=10,
        intro_duration=2,
        stimulus_contrast=1.0,
        non_label_index=-1,
        non_input_value=0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)
        self.stimulus_duration = int(stimulus_duration)
        self.intro_duration = int(intro_duration)
        self.stimulus_contrast = float(stimulus_contrast)
        self.non_label_index = int(non_label_index)
        self.non_input_value = float(non_input_value)
        self.outro_duration = (
            self.n_timesteps - self.stimulus_duration - self.intro_duration
        )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough time steps for the stimulus duration and intro duration! "
                f"(n_timesteps={self.n_timesteps}, stimulus_duration={self.stimulus_duration}, intro_duration={self.intro_duration})"
            )

        # Pre-compile JIT functions
        self._fill_period_jit = _fill_tensor_period
        self._expand_jit = _expand_tensor_optimized

    def __iter__(self):
        for sample in DataLoader.__iter__(self):
            data, label_indices, *extra = sample

            # Apply contrast before other operations
            data = _adjust_data_dimensions(data) * self.stimulus_contrast
            data = self._optimize_tensor_layout(data)
            label_indices = _adjust_label_dimensions(label_indices)

            # Get pre-allocated tensors
            output_data, output_labels = self._get_cached_tensors(
                data.shape, label_indices.shape, data.device, data.dtype
            )

            # Async tensor operations and pre-fill with void values
            data, label_indices = self._async_tensor_operations(
                data,
                label_indices,
                output_data,
                output_labels,
                self.non_input_value,
                self.non_label_index,
            )

            # Use JIT-compiled functions
            time_idx = self.intro_duration  # Skip intro (pre-filled)

            # Stimulus period
            if self.stimulus_duration > 0:
                expanded_data = self._expand_jit(data, 1, self.stimulus_duration)
                expanded_labels = self._expand_jit(
                    label_indices, 1, self.stimulus_duration
                )

                self._fill_period_jit(
                    output_data,
                    output_labels,
                    expanded_data,
                    expanded_labels,
                    time_idx,
                    self.stimulus_duration,
                )

            yield [output_data, output_labels, *extra]


class StimulusNoiseDataLoader(StandardDataLoader):
    @alias_kwargs(
        tsteps="n_timesteps",
        dsteps="n_timesteps",
        data_timesteps="n_timesteps",
        stim="stimulus_duration",
        intro="intro_duration",
        noisetype="noise_type",
        noiselevel="ssnr",
        noiseseed="noise_seed",
        noisevoid="noise_void",
        tempmode="temporal_mode",
        voidid="non_label_index",
        voidinput="non_input_value",
    )
    def __init__(
        self,
        *args,
        n_timesteps=20,
        stimulus_duration=None,
        intro_duration=0,
        noise_type="uniform",
        ssnr=0.5,
        noise_seed=None,
        noise_void=True,
        temporal_mode="dynamic",
        noise_cache_size=50,
        non_label_index=-1,
        non_input_value=0,
        **kwargs,
    ):
        super().__init__(*args, n_timesteps=n_timesteps, **kwargs)

        # Validate and store parameters
        self.stimulus_duration = (
            int(n_timesteps - intro_duration)
            if stimulus_duration is None
            else int(stimulus_duration)
        )
        self.intro_duration = int(intro_duration)
        self.noise_type = str(noise_type).lower()
        self.ssnr = float(ssnr)
        self.noise_seed = noise_seed
        self.noise_void = bool(noise_void)
        self.temporal_mode = str(temporal_mode).lower()
        self.noise_cache_size = noise_cache_size
        self.non_label_index = int(non_label_index)
        self.non_input_value = float(non_input_value)
        self.outro_duration = (
            self.n_timesteps - self.stimulus_duration - self.intro_duration
        )

        # Validate temporal_mode
        valid_temporal_modes = ["static", "dynamic", "correlated"]
        if self.temporal_mode not in valid_temporal_modes:
            raise ValueError(
                f"Invalid temporal_mode: {self.temporal_mode}. "
                f"Available modes: {valid_temporal_modes}"
            )

        if self.outro_duration < 0:
            raise ValueError(
                f"{self.__class__}:\n"
                "Not enough time steps for the stimulus duration and intro duration! "
                f"(n_timesteps={self.n_timesteps}, stimulus_duration={self.stimulus_duration}, intro_duration={self.intro_duration})"
            )

        # Noise function mapping (optimized lookup)
        self._noise_functions = {
            "saltpepper": noise.salt_pepper_noise,
            "poisson": noise.poisson_noise,
            "uniform": noise.uniform_noise,
            "gaussian": noise.gaussian_noise,
            "phasescrambled": noise.phase_scrambled_noise,
        }

        if self.noise_type not in self._noise_functions:
            raise ValueError(
                f"Unknown noise type: {self.noise_type}. Available: {list(self._noise_functions.keys())}"
            )

        self.noise_function = self._noise_functions[self.noise_type]
        self.noise_kwargs, _ = filter_kwargs(self.noise_function, kwargs)

        # Noise caching (only for deterministic noise)
        self._should_cache_noise = self.noise_seed is not None
        if self._should_cache_noise:
            self._noise_state_cache = {}
            self._noise_cache_order = OrderedDict()

        # Sample counter for indexed seeding
        self._sample_counter = 0

        # Pre-compile JIT functions
        self._fill_period_jit = _fill_tensor_period
        self._expand_jit = _expand_tensor_optimized

    def _get_noise_seed(self, sample_idx=None):
        """Generate appropriate seed for current sample."""
        if self.noise_seed is None:
            return None
        elif self.noise_seed == "indexed":
            # Use sample index for reproducible per-sample noise
            return hash((sample_idx or self._sample_counter, self.noise_type)) % (
                2**31
            )
        else:
            # Global seed for all samples
            return self.noise_seed

    def _get_cached_noise_state(self, data_shape, device, dtype, seed):
        """Get or create cached noise state for any noise type."""
        if not self._should_cache_noise or seed is None:
            return None

        cache_key = (
            data_shape,
            self.noise_type,
            self.ssnr,
            self.temporal_mode,  # Include temporal_mode in cache key
            seed,
            device,
            dtype,
        )

        # Update access order
        if cache_key in self._noise_cache_order:
            self._noise_cache_order.move_to_end(cache_key)
            return self._noise_state_cache[cache_key]

        # Check if we need to evict old entries
        if len(self._noise_state_cache) >= self.noise_cache_size:
            oldest_key = next(iter(self._noise_cache_order))
            del self._noise_state_cache[oldest_key]
            del self._noise_cache_order[oldest_key]

        # Generate new noise state by calling function without cached_noise_state
        dummy_tensor = torch.zeros(data_shape, dtype=dtype, device=device)
        result = self.noise_function(
            dummy_tensor,
            self.ssnr,
            seed=seed,
            temporal_mode=self.temporal_mode,
            **self.noise_kwargs,
        )

        # Handle case where function returns tuple (result, state) or just result
        if isinstance(result, tuple):
            _, noise_state = result
        else:
            # For functions that don't return state (like motion_blur), create empty state
            noise_state = {"function_type": self.noise_type}

        # Cache the noise state
        self._noise_state_cache[cache_key] = noise_state
        self._noise_cache_order[cache_key] = True

        return noise_state

    def _should_apply_noise(self) -> bool:
        """Determine if noise should be applied based on parameters."""
        # Use epsilon-based comparison for robust no-noise detection
        EPSILON = 1e-4
        return not (abs(self.ssnr - 1.0) < EPSILON)

    def _apply_noise_optimized(
        self, data: torch.Tensor, sample_idx=None
    ) -> torch.Tensor:
        """Apply noise with robust result handling and early exit optimization."""
        # Early exit for no-noise case
        if not self._should_apply_noise():
            return data

        seed = self._get_noise_seed(sample_idx)

        # Try to get cached noise state first
        cached_state = None
        if self._should_cache_noise and seed is not None:
            cached_state = self._get_cached_noise_state(
                data.shape, data.device, data.dtype, seed
            )

        # Apply noise with temporal_mode
        if cached_state is not None:
            # Apply noise using cached state
            result = self.noise_function(
                data,
                ssnr=self.ssnr,
                seed=seed,
                cached_noise_state=cached_state,
                temporal_mode=self.temporal_mode,
                **self.noise_kwargs,
            )
        else:
            # Direct noise generation
            result = self.noise_function(
                data,
                ssnr=self.ssnr,
                seed=seed,
                temporal_mode=self.temporal_mode,
                **self.noise_kwargs,
            )

        # Robust result handling
        if isinstance(result, tuple):
            noisy_data, noise_state = result
            # Optionally store noise_state for debugging/analysis if needed
            return noisy_data
        elif isinstance(result, torch.Tensor):
            return result
        else:
            raise TypeError(f"Unexpected noise function return type: {type(result)}")

    def _get_cached_noise_state(self, data_shape, device, dtype, seed):
        """Get or create cached noise state with improved cache key generation."""
        if not self._should_cache_noise or seed is None:
            return None

        # Use rounded SSNR value to avoid floating-point precision issues in cache keys
        ssnr_key = round(self.ssnr, 10)

        cache_key = (
            data_shape,
            self.noise_type,
            ssnr_key,  # Use rounded value
            self.temporal_mode,
            seed,
            device,
            dtype,
            tuple(sorted(self.noise_kwargs.items())),  # Include all noise parameters
        )

        # Update access order
        if cache_key in self._noise_cache_order:
            self._noise_cache_order.move_to_end(cache_key)
            return self._noise_state_cache[cache_key]

        # Check if we need to evict old entries
        if len(self._noise_state_cache) >= self.noise_cache_size:
            oldest_key = next(iter(self._noise_cache_order))
            del self._noise_state_cache[oldest_key]
            del self._noise_cache_order[oldest_key]

        # Generate new noise state by calling function without cached_noise_state
        dummy_tensor = torch.zeros(data_shape, dtype=dtype, device=device)
        result = self.noise_function(
            dummy_tensor,
            ssnr=self.ssnr,
            seed=seed,
            temporal_mode=self.temporal_mode,
            **self.noise_kwargs,
        )

        # Handle case where function returns tuple (result, state) or just result
        if isinstance(result, tuple):
            _, noise_state = result
        else:
            # For functions that don't return state, create empty state
            noise_state = {"function_type": self.noise_type}

        # Cache the noise state
        self._noise_state_cache[cache_key] = noise_state
        self._noise_cache_order[cache_key] = True

        return noise_state

    def _apply_noise_to_void_periods(
        self, output_data: torch.Tensor, sample_idx=None
    ) -> torch.Tensor:
        """Apply noise to void periods (intro and outro) respecting temporal_mode."""
        if not self._should_apply_noise() or not self.noise_void:
            return output_data

        if self.temporal_mode == "static":
            # For static mode, generate noise once and replicate across time
            # This ensures consistent noise across all void timesteps

            # Apply noise to intro period (static)
            if self.intro_duration > 0:
                # Take a single frame slice for noise generation
                intro_frame = output_data[
                    :, 0:1
                ]  # Shape: [batch, 1, channels, height, width]
                noisy_intro_frame = self._apply_noise_optimized(
                    intro_frame, sample_idx
                )

                # Replicate the noisy frame across all intro timesteps
                output_data[:, : self.intro_duration] = noisy_intro_frame.expand(
                    -1, self.intro_duration, -1, -1, -1
                )

            # Apply noise to outro period (static)
            if self.outro_duration > 0:
                outro_start = self.intro_duration + self.stimulus_duration
                # Take a single frame slice for noise generation
                outro_frame = output_data[
                    :, outro_start : outro_start + 1
                ]  # Shape: [batch, 1, channels, height, width]
                noisy_outro_frame = self._apply_noise_optimized(
                    outro_frame, sample_idx
                )

                # Replicate the noisy frame across all outro timesteps
                output_data[:, outro_start : outro_start + self.outro_duration] = (
                    noisy_outro_frame.expand(-1, self.outro_duration, -1, -1, -1)
                )

        else:  # dynamic or correlated mode
            # Apply noise independently to each timestep

            # Apply noise to intro period (dynamic)
            if self.intro_duration > 0:
                intro_data = output_data[:, : self.intro_duration]
                noisy_intro = self._apply_noise_optimized(intro_data, sample_idx)
                output_data[:, : self.intro_duration] = noisy_intro

            # Apply noise to outro period (dynamic)
            if self.outro_duration > 0:
                outro_start = self.intro_duration + self.stimulus_duration
                outro_data = output_data[
                    :, outro_start : outro_start + self.outro_duration
                ]
                noisy_outro = self._apply_noise_optimized(outro_data, sample_idx)
                output_data[:, outro_start : outro_start + self.outro_duration] = (
                    noisy_outro
                )

        return output_data

    def __iter__(self):
        self._sample_counter = 0  # Reset counter

        try:
            for sample in DataLoader.__iter__(self):
                data, label_indices, *extra = sample

                # Strategic noise application based on temporal_mode
                if self.temporal_mode == "static" or not self._should_apply_noise():
                    # Apply noise to 4D tensor before temporal expansion (more efficient)
                    data = self._apply_noise_optimized(data, self._sample_counter)

                    # Apply performance optimizations
                    data = _adjust_data_dimensions(data, self._optimal_memory_format)
                    data = self._optimize_tensor_layout(data)

                    if isinstance(data, torch.Tensor):
                        data = data.to(dtype=self.dtype)

                    label_indices = _adjust_label_dimensions(label_indices)

                    # Get pre-allocated tensors
                    output_data, output_labels = self._get_cached_tensors(
                        data.shape, label_indices.shape, data.device, data.dtype
                    )

                    # Async tensor operations and pre-fill with void values
                    data, label_indices = self._async_tensor_operations(
                        data,
                        label_indices,
                        output_data,
                        output_labels,
                        self.non_input_value,
                        self.non_label_index,
                    )

                    # Use JIT-compiled functions for stimulus period
                    time_idx = self.intro_duration  # Skip intro (pre-filled)

                    if self.stimulus_duration > 0:
                        expanded_data = self._expand_jit(
                            data, 1, self.stimulus_duration
                        )
                        expanded_labels = self._expand_jit(
                            label_indices, 1, self.stimulus_duration
                        )

                        self._fill_period_jit(
                            output_data,
                            output_labels,
                            expanded_data,
                            expanded_labels,
                            time_idx,
                            self.stimulus_duration,
                        )

                else:  # dynamic or correlated mode
                    # Apply performance optimizations first
                    data = _adjust_data_dimensions(data, self._optimal_memory_format)
                    data = self._optimize_tensor_layout(data)

                    if isinstance(data, torch.Tensor):
                        data = data.to(dtype=self.dtype)

                    label_indices = _adjust_label_dimensions(label_indices)

                    # Get pre-allocated tensors
                    output_data, output_labels = self._get_cached_tensors(
                        data.shape, label_indices.shape, data.device, data.dtype
                    )

                    # Async tensor operations and pre-fill with void values
                    data, label_indices = self._async_tensor_operations(
                        data,
                        label_indices,
                        output_data,
                        output_labels,
                        self.non_input_value,
                        self.non_label_index,
                    )

                    # Expand to temporal dimensions first
                    time_idx = self.intro_duration

                    if self.stimulus_duration > 0:
                        expanded_data = self._expand_jit(
                            data, 1, self.stimulus_duration
                        )
                        expanded_labels = self._expand_jit(
                            label_indices, 1, self.stimulus_duration
                        )

                        self._fill_period_jit(
                            output_data,
                            output_labels,
                            expanded_data,
                            expanded_labels,
                            time_idx,
                            self.stimulus_duration,
                        )

                    # Apply noise to the stimulus period after temporal expansion
                    if self.stimulus_duration > 0:
                        stimulus_data = output_data[
                            :,
                            self.intro_duration : self.intro_duration
                            + self.stimulus_duration,
                        ]
                        noisy_stimulus = self._apply_noise_optimized(
                            stimulus_data, self._sample_counter
                        )
                        output_data[
                            :,
                            self.intro_duration : self.intro_duration
                            + self.stimulus_duration,
                        ] = noisy_stimulus

                # Apply noise to void periods (now respects temporal_mode)
                if self.noise_void:
                    output_data = self._apply_noise_to_void_periods(
                        output_data, self._sample_counter
                    )

                self._sample_counter += 1
                yield [output_data, output_labels, *extra]

        except Exception as e:
            logger.error(f"Error in stimulus noise loading: {str(e)}")
            raise


DATALOADER_CLASSES = {
    "StandardDataLoader": StandardDataLoader,
    "StimulusRepetitionDataLoader": StimulusRepetitionDataLoader,
    "StimulusDurationDataLoader": StimulusDurationDataLoader,
    "StimulusIntervalDataLoader": StimulusIntervalDataLoader,
    "StimulusContrastDataLoader": StimulusContrastDataLoader,
    "StimulusNoiseDataLoader": StimulusNoiseDataLoader,
}


def get_data_loader(
    dataset: torch.utils.data.Dataset,
    dataloader: Optional[str] = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Returns a DataLoader instance based on the specified dataloader name or class.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        dataloader (Optional[str]): The name of the DataLoader class or the class itself.
        data_timesteps (int): Number of timesteps for temporal data.
        **kwargs: Additional arguments for the DataLoader.

    Returns:
        torch.utils.data.DataLoader: An instance of the specified DataLoader.

    Raises:
        ValueError: If the specified DataLoader name is invalid.
        TypeError: If the provided dataloader is not a string or a valid DataLoader class.
        Exception: If there is an error during DataLoader initialization.
    """
    # Default to StandardDataLoader if no dataloader is specified
    dataloader = dataloader or StandardDataLoader

    # Validate and resolve the dataloader
    if isinstance(dataloader, str):
        dataloader_class = get_data_loader_class(dataloader)
    elif issubclass(dataloader, DataLoader):
        dataloader_class = dataloader
    else:
        raise TypeError(
            f"Invalid dataloader type: {type(dataloader)}. "
            "Expected a string or a DataLoader subclass."
        )

    filtered_kwargs, _ = filter_kwargs(dataloader_class, kwargs)

    return dataloader_class(dataset, **filtered_kwargs)


def get_data_loader_class(dataloader_name: str) -> torch.utils.data.DataLoader:

    if "DataLoader" not in dataloader_name:
        dataloader_name += "DataLoader"
    if dataloader_name in DATALOADER_CLASSES:
        dataloader_class = DATALOADER_CLASSES.get(dataloader_name)
    else:
        raise ValueError(f"Unknown DataLoader class: '{dataloader_name}'")

    return dataloader_class


def get_train_val_loaders(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 0,
    data_timesteps: int = 1,
    dataloader: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders from a dataset.

    Args:
        dataset: The dataset to split
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes
        data_timesteps: Number of timesteps for temporal data
        dataloader: Specific dataloader class to use
        **kwargs: Additional arguments for dataloaders

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Calculate split sizes
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Common loader arguments
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "data_timesteps": data_timesteps,
        "dataloader": dataloader,
        **kwargs,
    }
    common_kwargs.pop("shuffle", None)

    # Create loaders with appropriate shuffle settings
    train_loader = get_data_loader(train_dataset, shuffle=True, **common_kwargs)

    val_loader = get_data_loader(val_dataset, shuffle=False, **common_kwargs)

    return train_loader, val_loader
