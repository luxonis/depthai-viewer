# NOTE: This file was autogenerated by re_types_builder; DO NOT EDIT.

from __future__ import annotations

from .. import datatypes
from .._baseclasses import (
    BaseDelegatingExtensionArray,
    BaseDelegatingExtensionType,
)

__all__ = ["Point2DArray", "Point2DType"]


class Point2DType(BaseDelegatingExtensionType):
    _TYPE_NAME = "rerun.point2d"
    _DELEGATED_EXTENSION_TYPE = datatypes.Vec2DType


class Point2DArray(BaseDelegatingExtensionArray[datatypes.Vec2DArrayLike]):
    _EXTENSION_NAME = "rerun.point2d"
    _EXTENSION_TYPE = Point2DType
    _DELEGATED_ARRAY_TYPE = datatypes.Vec2DArray


Point2DType._ARRAY_TYPE = Point2DArray

# TODO(cmc): bring back registration to pyarrow once legacy types are gone
# pa.register_extension_type(Point2DType())