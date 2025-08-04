import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .ffcv_operations import ExtendDataTimeFFCV, ExtendLabelTimeFFCV
import torch
import torch.distributed
import numpy as np
from dynvision.utils import handle_errors
from ffcv.fields.decoders import (
    IntDecoder,
    NDArrayDecoder,
    RandomResizedCropRGBImageDecoder,
    SimpleRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
    NormalizeImage,
    RandomResizedCrop,
)

from dynvision.data.transforms import get_data_transform, get_target_transform

logger = logging.getLogger(__name__)


def _build_image_pipeline(
    encoding: str,
    resolution: int,
    data_transform: Optional[List[Callable]],
    n_timesteps: int = 0,
    normalize: Optional[Tuple[List, List]] = None,
    dtype: Optional[torch.dtype] = torch.float16,
    device: Optional[torch.device] = None,
) -> List[Operation]:
    if encoding == "tensor":
        pipeline = [NDArrayDecoder(), ToTensor()]
    elif encoding == "image":
        # pipeline = [
        #     SimpleRGBImageDecoder(),
        #     RandomResizedCrop(scale=(1, 1), ratio=(1, 1), size=resolution),
        # ]
        pipeline = [RandomResizedCropRGBImageDecoder((resolution, resolution))]
    else:
        raise ValueError(f"Unsupported encoding type: {encoding}")

    if data_transform:
        pipeline.extend(data_transform)

    if normalize:
        pipeline.append(
            NormalizeImage(
                mean=np.array(normalize[0]) * 255,
                std=np.array(normalize[1]) * 255,
                type=float,
            )
        )

    # Convert to tensor and proper image format first
    pipeline.append(ToTensor())
    pipeline.append(ToTorchImage(convert_back_int16=False))

    # Finally apply dtype and device conversions
    if dtype:
        pipeline.append(Convert(dtype))

    if device:
        pipeline.append(ToDevice(device))

    # Then extend time dimension after tensor is in correct format
    # if n_timesteps > 1:
    #     pipeline.append(ExtendDataTimeFFCV(n_timesteps))

    return pipeline


def _build_label_pipeline(
    target_transform: Optional[List[Callable]],
    n_timesteps: int = 0,
    device: Optional[torch.device] = None,
) -> List[Operation]:
    pipeline = [IntDecoder(), ToTensor()]

    if target_transform:
        pipeline.extend(target_transform)

    if device:
        pipeline.append(ToDevice(device))

    # if n_timesteps > 1:
    #     pipeline.append(ExtendLabelTimeFFCV(n_timesteps))

    return pipeline


@handle_errors()
def get_ffcv_dataloader(
    path: Union[str, Path],
    batch_size: int = 1,
    data_timesteps: int = 0,
    num_workers: int = 4,
    data_transform: Optional[Union[str, List[str]]] = None,
    target_transform: Optional[str] = None,
    normalize: Optional[Tuple[List, List]] = None,  # (mean, std)
    order: OrderOption = OrderOption.RANDOM,
    os_cache: Optional[bool] = None,
    encoding: str = "image",
    resolution: int = 224,
    drop_last: bool = True,
    dtype: Optional[torch.dtype] = torch.float16,
    batches_ahead: int = 3,
    train: bool = True,
    verbose: bool = False,
    device: Optional[torch.device] = None,
    distributed: bool = False,
    **kwargs: Any,
) -> Loader:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    # Check if distributed mode is actually available and initialized
    if distributed:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            logger.warning(
                "Distributed mode requested but PyTorch distributed is not initialized. "
                "Falling back to non-distributed mode."
            )
            distributed = False

    target_transform = get_target_transform(target_transform) or []

    if train:    
        data_transform = get_data_transform(data_transform) or []
    else:
        encoding = "image"
        data_transform = []

    image_pipeline = _build_image_pipeline(
        encoding=encoding,
        resolution=resolution,
        data_transform=data_transform,
        device=device,
        dtype=dtype,
        n_timesteps=data_timesteps,
        normalize=normalize,
    )

    label_pipeline = _build_label_pipeline(
        target_transform=target_transform,
        n_timesteps=data_timesteps,
        device=device,
    )

    pipelines = {"image": image_pipeline, "label": label_pipeline}

    if verbose:
        logger.info(f"Image pipeline: {[type(p).__name__ for p in image_pipeline]}")
        logger.info(f"Label pipeline: {[type(p).__name__ for p in label_pipeline]}")

    logger.info(f"Creating FFCV loader from {path}")

    return Loader(
        str(path),
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        pipelines=pipelines,
        os_cache=os_cache,
        drop_last=drop_last,
        batches_ahead=batches_ahead,
        distributed=distributed,
        seed=0,
    )
