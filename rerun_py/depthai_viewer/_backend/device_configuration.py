from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import depthai as dai
from depthai_sdk import Previews as QueueNames
from pydantic import BaseModel

# class PointcloudConfiguration(BaseModel):
#     enabled: bool = True


class StereoDepthConfiguration(BaseModel):  # type: ignore[misc]
    median: Optional[dai.MedianFilter] = dai.MedianFilter.KERNEL_7x7
    lr_check: Optional[bool] = True
    lrc_threshold: int = 5  # 0..10
    extended_disparity: Optional[bool] = False
    subpixel_disparity: Optional[bool] = True
    align: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_B
    sigma: int = 0  # 0..65535
    # pointcloud: PointcloudConfiguration | None = None
    confidence: int = 230
    stereo_pair: Tuple[dai.CameraBoardSocket, dai.CameraBoardSocket]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **v) -> None:  # type: ignore[no-untyped-def]
        if v.get("median", None) and isinstance(v["median"], str):
            v["median"] = getattr(dai.MedianFilter, v["median"])
        if v.get("align", None) and isinstance(v["align"], str):
            v["align"] = getattr(dai.CameraBoardSocket, v["align"])
        if v.get("stereo_pair", None) and all(isinstance(pair, str) for pair in v["stereo_pair"]):
            v["stereo_pair"] = (
                getattr(dai.CameraBoardSocket, v["stereo_pair"][0]),
                getattr(dai.CameraBoardSocket, v["stereo_pair"][1]),
            )
        return super().__init__(**v)  # type: ignore[no-any-return]

    def dict(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        return {
            "median": self.median.name if self.median else None,
            "lr_check": self.lr_check,
            "lrc_threshold": self.lrc_threshold,
            "extended_disparity": self.extended_disparity,
            "subpixel_disparity": self.subpixel_disparity,
            "align": self.align.name,
            "sigma": self.sigma,
            "confidence": self.confidence,
            "stereo_pair": [socket.name for socket in self.stereo_pair],
        }

    def to_runtime_controls(self) -> Dict[str, Any]:
        return {
            "algorithm_control": {
                "align": (
                    "RECTIFIED_LEFT"
                    if self.align == dai.CameraBoardSocket.LEFT
                    else "RECTIFIED_RIGHT"
                    if self.align == dai.CameraBoardSocket.RIGHT
                    else "CENTER"
                ),
                "lr_check": self.lr_check,
                "lrc_check_threshold": self.lrc_threshold,
                "extended": self.extended_disparity,
                "subpixel": self.subpixel_disparity,
            },
            "postprocessing": {
                "median": (
                    {
                        dai.MedianFilter.MEDIAN_OFF: 0,
                        dai.MedianFilter.KERNEL_3x3: 3,
                        dai.MedianFilter.KERNEL_5x5: 5,
                        dai.MedianFilter.KERNEL_7x7: 7,
                    }[self.median]
                    if self.median
                    else 0
                ),
                "bilateral_sigma": self.sigma,
            },
            "cost_matching": {
                "confidence_threshold": self.confidence,
            },
        }

    @property
    def out_queue_name(self) -> str:
        return str(QueueNames.depthRaw.name)


class AiModelConfiguration(BaseModel):  # type: ignore[misc]
    display_name: str = "Yolo V6"
    path: str = "yolov6nr3_coco_640x352"
    camera: dai.CameraBoardSocket

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **v) -> None:  # type: ignore[no-untyped-def]
        if v.get("camera", None) and isinstance(v["camera"], str):
            v["camera"] = getattr(dai.CameraBoardSocket, v["camera"])
        return super().__init__(**v)  # type: ignore[no-any-return]

    def dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "display_name": self.display_name,
            "path": self.path,
            "camera": self.camera.name,
        }


ALL_NEURAL_NETWORKS = [
    AiModelConfiguration(
        path="yolov8n_coco_640x352",
        display_name="Yolo V8",
        camera=dai.CameraBoardSocket.CAM_A,
    ),
    AiModelConfiguration(
        path="yolov6nr3_coco_640x352",
        display_name="Yolo V6",
        camera=dai.CameraBoardSocket.CAM_A,
    ),
    AiModelConfiguration(
        path="face-detection-retail-0004",
        display_name="Face Detection",
        camera=dai.CameraBoardSocket.CAM_A,
    ),
    AiModelConfiguration(
        path="age-gender-recognition-retail-0013",
        display_name="Age gender recognition",
        camera=dai.CameraBoardSocket.CAM_A,
    ),
    AiModelConfiguration(
        path="yolov6n_thermal_people_256x192",
        display_name="Thermal Person Detection",
        camera=dai.CameraBoardSocket.CAM_E,
    ),
]


class ImuConfiguration(BaseModel):  # type: ignore[misc]
    report_rate: int = 100
    batch_report_threshold: int = 5


class CameraSensorResolution(Enum):
    THE_256X192: str = "THE_256X192"
    THE_400_P: str = "THE_400_P"
    THE_480_P: str = "THE_480_P"
    THE_720_P: str = "THE_720_P"
    THE_800_P: str = "THE_800_P"
    THE_5_MP: str = "THE_5_MP"
    THE_1440X1080: str = "THE_1440X1080"
    THE_1080_P: str = "THE_1080_P"
    THE_1200_P: str = "THE_1200_P"
    THE_1280_P: str = "THE_1280_P"
    THE_1280X3848: str = "THE_1280X3848"
    THE_4_K: str = "THE_4_K"
    THE_4000X3000: str = "THE_4000X3000"
    THE_12_MP: str = "THE_12_MP"
    THE_13_MP: str = "THE_13_MP"
    THE_5312X6000: str = "THE_5312X6000"

    def dict(self, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        return self.value

    def as_sdk_resolution(self) -> str:
        return self.value.replace("_", "").replace("THE", "")


class ImuKind(Enum):
    SIX_AXIS = "SIX_AXIS"
    NINE_AXIS = "NINE_AXIS"


class CameraConfiguration(BaseModel):  # type: ignore[misc]
    fps: int = 30
    resolution: CameraSensorResolution
    kind: dai.CameraSensorType
    board_socket: dai.CameraBoardSocket
    stream_enabled: bool = True
    name: str = ""

    tof_align: Optional[dai.CameraBoardSocket] = None
    is_used_as_stereo_align: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **v) -> None:  # type: ignore[no-untyped-def]
        if v.get("board_socket", None):
            if isinstance(v["board_socket"], str):
                v["board_socket"] = getattr(dai.CameraBoardSocket, v["board_socket"])
        if v.get("kind", None):
            if isinstance(v["kind"], str):
                v["kind"] = getattr(dai.CameraSensorType, v["kind"])
        return super().__init__(**v)  # type: ignore[no-any-return]

    def dict(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        return {
            "fps": self.fps,
            "resolution": self.resolution.dict(),
            "kind": self.kind.name,
            "board_socket": self.board_socket.name,
            "name": self.name,
            "stream_enabled": self.stream_enabled,
        }

    @classmethod
    # type: ignore[no-untyped-def]
    def create_left(cls, **kwargs) -> "CameraConfiguration":
        if not kwargs.get("kind", None):
            kwargs["kind"] = dai.CameraSensorType.MONO
        if not kwargs.get("resolution", None):
            kwargs["resolution"] = CameraSensorResolution.THE_400_P
        return cls(board_socket="LEFT", **kwargs)

    @classmethod
    # type: ignore[no-untyped-def]
    def create_right(cls, **kwargs) -> "CameraConfiguration":
        if not kwargs.get("kind", None):
            kwargs["kind"] = dai.CameraSensorType.MONO
        if not kwargs.get("resolution", None):
            kwargs["resolution"] = CameraSensorResolution.THE_400_P
        return cls(board_socket="RIGHT", **kwargs)

    @classmethod
    # type: ignore[no-untyped-def]
    def create_color(cls, **kwargs) -> "CameraConfiguration":
        if not kwargs.get("kind", None):
            kwargs["kind"] = dai.CameraSensorType.COLOR
        if not kwargs.get("resolution", None):
            kwargs["resolution"] = CameraSensorResolution.THE_720_P
        return cls(board_socket="RGB", **kwargs)


class ToFConfig(BaseModel):  # type: ignore[misc]
    median: Optional[dai.MedianFilter] = dai.MedianFilter.MEDIAN_OFF
    phase_unwrapping_level: int = 4
    phase_unwrap_error_threshold: int = 100
    enable_phase_unwrapping: Optional[bool] = True
    enable_fppn_correction: Optional[bool] = True
    enable_optical_correction: Optional[bool] = True
    enable_temperature_correction: Optional[bool] = False
    enable_wiggle_correction: Optional[bool] = True
    enable_phase_shuffle_temporal_filter: Optional[bool] = True
    enable_burst_mode: Optional[bool] = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **v) -> None:  # type: ignore[no-untyped-def]
        if v.get("median", None):
            if isinstance(v["median"], str):
                v["median"] = getattr(dai.MedianFilter, v["median"])
        return super().__init__(**v)  # type: ignore[no-any-return]

    def dict(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        return {
            "median": self.median.name if self.median else None,
            "phase_unwrapping_level": self.phase_unwrapping_level,
            "phase_unwrap_error_threshold": self.phase_unwrap_error_threshold,
            "enable_fppn_correction": self.enable_fppn_correction,
            "enable_optical_correction": self.enable_optical_correction,
            "enable_temperature_correction": self.enable_temperature_correction,
            "enable_wiggle_correction": self.enable_wiggle_correction,
            "enable_phase_unwrapping": self.enable_phase_unwrapping,
            "enable_phase_shuffle_temporal_filter": self.enable_phase_shuffle_temporal_filter,
            "enable_burst_mode": self.enable_burst_mode,
        }

    def to_dai(self) -> dai.RawToFConfig:
        cfg = dai.RawToFConfig()
        cfg.median = self.median  # type: ignore[attr-defined, assignment]
        cfg.phaseUnwrappingLevel = self.phase_unwrapping_level  # type: ignore[attr-defined]
        cfg.phaseUnwrapErrorThreshold = self.phase_unwrap_error_threshold  # type: ignore[attr-defined]
        cfg.enableFPPNCorrection = self.enable_fppn_correction  # type: ignore[attr-defined, assignment]
        cfg.enableOpticalCorrection = self.enable_optical_correction  # type: ignore[attr-defined, assignment]
        cfg.enableTemperatureCorrection = self.enable_temperature_correction  # type: ignore[attr-defined, assignment]
        cfg.enableWiggleCorrection = self.enable_wiggle_correction  # type: ignore[attr-defined, assignment]
        cfg.enablePhaseUnwrapping = self.enable_phase_unwrapping  # type: ignore[attr-defined, assignment]
        cfg.enablePhaseShuffleTemporalFilter = (
            self.enable_phase_shuffle_temporal_filter  # type: ignore[attr-defined, assignment]
        )
        cfg.enableBurstMode = self.enable_burst_mode  # type: ignore[attr-defined, assignment]
        return cfg


class CameraFeatures(BaseModel):  # type: ignore[misc]
    resolutions: List[CameraSensorResolution] = []
    max_fps: int = 60
    board_socket: dai.CameraBoardSocket
    supported_types: List[dai.CameraSensorType]
    stereo_pairs: List[dai.CameraBoardSocket] = []
    """Which cameras can be paired with this one"""
    name: str
    tof_config: Optional[ToFConfig] = None

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True

    def dict(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        return {
            "resolutions": [r for r in self.resolutions],
            "max_fps": self.max_fps,
            "board_socket": self.board_socket.name,
            "supported_types": [sensor_type.name for sensor_type in self.supported_types],
            "stereo_pairs": [socket.name for socket in self.stereo_pairs],
            "name": self.name,
            "tof_config": self.tof_config.dict() if self.tof_config else None,
        }


class PipelineConfiguration(BaseModel):  # type: ignore[misc]
    auto: bool = False  # Should the backend automatically create a pipeline?
    cameras: List[CameraConfiguration] = []
    stereo: Optional[StereoDepthConfiguration]
    ai_model: Optional[AiModelConfiguration]
    imu: ImuConfiguration = ImuConfiguration()


class XLinkConnection(Enum):
    USB = "Usb"
    POE = "PoE"


class DeviceInfo(BaseModel):  # type: ignore[misc]
    name: str = ""
    connection: XLinkConnection = XLinkConnection.USB
    mxid: str = ""


class DeviceProperties(BaseModel):  # type: ignore[misc]
    id: str
    cameras: List[CameraFeatures] = []
    imu: Optional[ImuKind]
    stereo_pairs: List[
        Tuple[dai.CameraBoardSocket, dai.CameraBoardSocket]
    ] = []  # Which cameras can be paired for stereo
    default_stereo_pair: Optional[Tuple[dai.CameraBoardSocket, dai.CameraBoardSocket]] = None
    info: DeviceInfo = DeviceInfo()

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if kwargs.get("stereo_pairs", None) and all(isinstance(pair[0], str) for pair in kwargs["stereo_pairs"]):
            kwargs["stereo_pairs"] = [
                (getattr(dai.CameraBoardSocket, pair[0]), getattr(dai.CameraBoardSocket, pair[1]))
                for pair in kwargs["stereo_pairs"]
            ]
        return super().__init__(*args, **kwargs)  # type: ignore[no-any-return]

    def dict(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        return {
            "id": self.id,
            "cameras": [cam.dict() for cam in self.cameras],
            "imu": self.imu,
            "stereo_pairs": [(left.name, right.name) for left, right in self.stereo_pairs],
            "info": {
                "name": self.info.name,
                "connection": self.info.connection.value,
                "mxid": self.info.mxid,
            },
            "default_stereo_pair": (
                (self.default_stereo_pair[0].name, self.default_stereo_pair[1].name)
                if self.default_stereo_pair
                else None
            ),
        }


size_to_resolution = {
    (256, 192): CameraSensorResolution.THE_256X192,
    (640, 400): CameraSensorResolution.THE_400_P,
    (640, 480): CameraSensorResolution.THE_480_P,  # OV7251
    (1280, 720): CameraSensorResolution.THE_720_P,
    (1280, 962): CameraSensorResolution.THE_1280_P,  # TOF
    (1280, 800): CameraSensorResolution.THE_800_P,  # OV9782
    (1280, 3848): CameraSensorResolution.THE_1280X3848,  # TOF
    (2592, 1944): CameraSensorResolution.THE_5_MP,  # OV5645
    (1440, 1080): CameraSensorResolution.THE_1440X1080,
    (1920, 1080): CameraSensorResolution.THE_1080_P,
    (1920, 1200): CameraSensorResolution.THE_1200_P,  # AR0234
    (3840, 2160): CameraSensorResolution.THE_4_K,
    # IMX582 with binning enabled
    (4000, 3000): CameraSensorResolution.THE_4000X3000,
    (4056, 3040): CameraSensorResolution.THE_12_MP,  # IMX378, IMX477, IMX577
    (4208, 3120): CameraSensorResolution.THE_13_MP,  # AR214
    (5312, 6000): CameraSensorResolution.THE_5312X6000,  # IMX582 cropped
}


def get_size_from_resolution(resolution: CameraSensorResolution) -> Tuple[int, int]:
    for size, res in size_to_resolution.items():
        if res == resolution:
            return size
    raise ValueError(f"Unknown resolution {resolution}")


def compare_dai_camera_configs(cam1: dai.CameraSensorConfig, cam2: dai.CameraSensorConfig) -> bool:
    return (  # type: ignore[no-any-return]
        cam1.height == cam2.height
        and cam1.width == cam2.width
        and cam1.type == cam2.type
        and cam1.maxFps == cam2.maxFps
        and cam1.minFps == cam2.minFps
    )


def calculate_isp_scale(resolution_width: int) -> Tuple[int, int]:
    """Based on width, get ISP scale to target THE_800_P, aka 1280x800."""
    x = 1280 / resolution_width
    return Fraction.from_float(x).limit_denominator().as_integer_ratio()
