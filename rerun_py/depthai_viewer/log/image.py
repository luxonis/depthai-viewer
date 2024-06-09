from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from depthai_viewer import bindings
from depthai_viewer.components.tensor import Colormap, ImageEncoding
from depthai_viewer.log.error_utils import _send_warning
from depthai_viewer.log.log_decorator import log_decorator
from depthai_viewer.log.tensor import Tensor, _log_tensor, _to_numpy

__all__ = [
    "log_image",
    "log_depth_image",
    "log_segmentation_image",
    "log_encoded_image",
]


@log_decorator
def log_image(
    entity_path: str,
    image: Tensor,
    *,
    colormap: Optional[Colormap] = None,
    unit: Optional[str] = None,
    ext: Optional[Dict[str, Any]] = None,
    timeless: bool = False,
) -> None:
    """
    Log a gray or color image.

    The image should either have 1, 3 or 4 channels (gray, RGB or RGBA).

    Supported dtypes
    ----------------
     - uint8: color components should be in 0-255 sRGB gamma space, except for alpha which should be in 0-255 linear
    space.
     - uint16: color components should be in 0-65535 sRGB gamma space, except for alpha which should be in 0-65535
    linear space.
     - float32, float64: all color components should be in 0-1 linear space.

    Parameters
    ----------
    entity_path:
        Path to the image in the space hierarchy.
    image:
        A [Tensor][rerun.log.tensor.Tensor] representing the image to log.
    colormap:
        Optional colormap to apply to single channel images.
    unit:
        Optional unit of the single channel image.
    ext:
        Optional dictionary of extension components. See [rerun.log_extension_components][]
    timeless:
        If true, the image will be timeless (default: False).

    """

    image = _to_numpy(image)

    shape = image.shape
    non_empty_dims = [d for d in shape if d != 1]
    num_non_empty_dims = len(non_empty_dims)

    interpretable_as_image = True
    # Catch some errors early:
    if num_non_empty_dims < 2 or 3 < num_non_empty_dims:
        _send_warning(f"Expected image, got array of shape {shape}", 1)
        interpretable_as_image = False

    if num_non_empty_dims == 3:
        depth = shape[-1]
        if depth not in (1, 3, 4):
            _send_warning(
                f"Expected image depth of 1 (gray), 3 (RGB) or 4 (RGBA). Instead got array of shape {shape}", 1
            )
            interpretable_as_image = False

    # TODO(#672): Don't squeeze once the image view can handle extra empty dimensions
    if interpretable_as_image and num_non_empty_dims != len(shape):
        image = np.squeeze(image)

    _log_tensor(entity_path, image, ext=ext, timeless=timeless, colormap=colormap, unit=unit)


@log_decorator
def log_depth_image(
    entity_path: str,
    image: Tensor,
    *,
    min: Optional[float] = None,
    max: Optional[float] = None,
    meter: Optional[float] = None,
    ext: Optional[Dict[str, Any]] = None,
    timeless: bool = False,
) -> None:
    """
    Log a depth image.

    The image must be a 2D array.

    Supported dtypes
    ----------------
    uint8, uint16, float32, float64

    Parameters
    ----------
    entity_path:
        Path to the image in the space hierarchy.
    image:
        A [Tensor][rerun.log.tensor.Tensor] representing the depth image to log.
    min:
        Optional minimum depth value.
    max:
        Optional maximum depth value.
    meter:
        How long is a meter in the given dtype?
        For instance: with uint16, perhaps meter=1000 which would mean
        you have millimeter precision and a range of up to ~65 meters (2^16 / 1000).
    ext:
        Optional dictionary of extension components. See [rerun.log_extension_components][]
    timeless:
        If true, the image will be timeless (default: False).

    """

    image = _to_numpy(image)

    # TODO(#635): Remove when issue with displaying f64 depth images is fixed.
    if image.dtype == np.float64:
        image = image.astype(np.float32)

    shape = image.shape
    non_empty_dims = [d for d in shape if d != 1]
    num_non_empty_dims = len(non_empty_dims)

    # Catch some errors early:
    if num_non_empty_dims != 2:
        _send_warning(f"Expected 2D depth image, got array of shape {shape}", 1)
        _log_tensor(entity_path, image, timeless=timeless, meaning=bindings.TensorDataMeaning.Depth)
    else:
        # TODO(#672): Don't squeeze once the image view can handle extra empty dimensions.
        if num_non_empty_dims != len(shape):
            image = np.squeeze(image)
        _log_tensor(
            entity_path,
            image,
            meter=meter,
            ext=ext,
            timeless=timeless,
            meaning=bindings.TensorDataMeaning.Depth,
            depth_min=min,
            depth_max=max,
        )


@log_decorator
def log_segmentation_image(
    entity_path: str,
    image: npt.ArrayLike,
    *,
    ext: Optional[Dict[str, Any]] = None,
    timeless: bool = False,
) -> None:
    """
    Log an image made up of integer class-ids.

    The image should have 1 channel, i.e. be either `H x W` or `H x W x 1`.

    See: [rerun.log_annotation_context][] for information on how to map the class-ids to
    colors and labels.

    Supported dtypes
    ----------------
    uint8, uint16

    Parameters
    ----------
    entity_path:
        Path to the image in the space hierarchy.
    image:
        A [Tensor][rerun.log.tensor.Tensor] representing the segmentation image to log.
    ext:
        Optional dictionary of extension components. See [rerun.log_extension_components][]
    timeless:
        If true, the image will be timeless (default: False).

    """

    image = np.array(image, copy=False)
    if image.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        image = np.require(image, np.uint16)
    non_empty_dims = [d for d in image.shape if d != 1]
    num_non_empty_dims = len(non_empty_dims)

    # Catch some errors early:
    if num_non_empty_dims != 2:
        _send_warning(
            f"Expected single channel image, got array of shape {image.shape}. Can't interpret as segmentation image.",
            1,
        )
        _log_tensor(
            entity_path,
            tensor=image,
            ext=ext,
            timeless=timeless,
        )
    else:
        # TODO(#672): Don't squeeze once the image view can handle extra empty dimensions.
        if num_non_empty_dims != len(image.shape):
            image = np.squeeze(image)
        _log_tensor(
            entity_path,
            tensor=image,
            meaning=bindings.TensorDataMeaning.ClassId,
            ext=ext,
            timeless=timeless,
        )


@log_decorator
def log_encoded_image(
    entity_path: str,
    image: npt.ArrayLike,
    width: int,
    height: int,
    encoding: ImageEncoding,
    *,
    ext: Optional[Dict[str, Any]] = None,
    timeless: bool = False,
) -> None:
    """
    Log an image encoded as a string.

    The image should be encoded as a string, e.g. using base64.

    Parameters
    ----------
    entity_path:
        Path to the image in the space hierarchy.
    image:
        A [Tensor][rerun.log.tensor.Tensor] representing the image to log.
    width:
        The (RGB) width of the image.
    height:
        The (RGB) height of the image.
    encoding:
        The encoding of the image.
    ext:
        Optional dictionary of extension components. See [rerun.log_extension_components][]
    timeless:
        If true, the image will be timeless (default: False).
    """
    image = np.array(image, copy=False)
    tensor_height = height
    if encoding == ImageEncoding.NV12:  #  or encoding == ImageEncoding.Yuv420p:
        # TODO(filip): This doesn't seem to be completely true as the image is still displayed correctly
        # tmp_height = int(height * 1.5)
        # if tmp_height % 2 != 0:
        #     _send_warning(f"Invalid height {height} for NV12 encoded image: height * 1.5 must be divisible by 2.", 1)
        # tensor_height = int(tmp_height)
        tensor_height = int(height * 1.5)
    _log_tensor(entity_path, image.reshape(tensor_height, width), ext=ext, timeless=timeless, encoding=encoding)
