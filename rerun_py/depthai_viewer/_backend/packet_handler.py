from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import depthai as dai
import numpy as np
from ahrs.filters import Mahony
from depthai_sdk.classes.packets import (  # PointcloudPacket,
    DepthPacket,
    DetectionPacket,
    FramePacket,
    IMUPacket,
    TwoStagePacket,
    _Detection,
)
from numpy.typing import NDArray
from pydantic import BaseModel
from turbojpeg import TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE, TurboJPEG

import depthai_viewer as viewer
from depthai_viewer._backend.device_configuration import CameraConfiguration
from depthai_viewer._backend.store import Store
from depthai_viewer._backend.topic import Topic
from depthai_viewer.components.rect2d import RectFormat


class CallbackArgs(BaseModel):  # type: ignore[misc]
    pass


class DepthCallbackArgs(CallbackArgs):  # type: ignore[misc]
    alignment_camera: CameraConfiguration
    stereo_pair: Tuple[dai.CameraBoardSocket, dai.CameraBoardSocket]

    class Config:
        arbitrary_types_allowed = True


class AiModelCallbackArgs(CallbackArgs):  # type: ignore[misc]
    model_name: str
    camera: CameraConfiguration
    labels: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True


class SyncedCallbackArgs(BaseModel):  # type: ignore[misc]
    depth_args: Optional[DepthCallbackArgs] = None
    ai_args: Optional[AiModelCallbackArgs] = None


class PacketHandler:
    store: Store
    _ahrs: Mahony
    _get_camera_intrinsics: Callable[[dai.CameraBoardSocket, int, int], NDArray[np.float32]]
    _jpeg_decoder: TurboJPEG = TurboJPEG()

    def __init__(
        self, store: Store, intrinsics_getter: Callable[[dai.CameraBoardSocket, int, int], NDArray[np.float32]]
    ):
        viewer.init("Depthai Viewer")
        viewer.connect()
        self.store = store
        self._ahrs = Mahony(frequency=100)
        self._ahrs.Q = np.array([1, 0, 0, 0], dtype=np.float64)
        self.set_camera_intrinsics_getter(intrinsics_getter)

    def reset(self) -> None:
        self._ahrs = Mahony(frequency=100)
        self._ahrs.Q = np.array([1, 0, 0, 0], dtype=np.float64)

    def set_camera_intrinsics_getter(
        self, camera_intrinsics_getter: Callable[[dai.CameraBoardSocket, int, int], NDArray[np.float32]]
    ) -> None:
        self._get_camera_intrinsics = camera_intrinsics_getter  # type: ignore[assignment, misc]

    def build_sync_callback(self, args: SyncedCallbackArgs) -> Callable[[Any], None]:
        return lambda packets: self._on_synced_packets(args, packets)

    def _on_synced_packets(self, args: SyncedCallbackArgs, packets: Dict[str, Any]) -> None:
        for descriptor, packet in packets.items():
            if type(packet) is FramePacket:
                # Create dai.CameraBoardSocket from descriptor
                split_descriptor = descriptor.split(".")
                sock = getattr(dai, split_descriptor[0])
                for split in split_descriptor[1:]:
                    sock = getattr(sock, split)
                self._on_camera_frame(packet, sock)
            elif type(packet) is DepthPacket:
                if args.depth_args is None:
                    continue
                self._on_stereo_frame(packet, args.depth_args)
            elif type(packet) is DetectionPacket:
                if args.ai_args is None:
                    continue
                self._on_detections(packet, args.ai_args)
            elif type(packet) is TwoStagePacket:
                if args.ai_args is None:
                    continue
                self._on_age_gender_packet(packet, args.ai_args)

    def build_callback(
        self, args: Union[dai.CameraBoardSocket, DepthCallbackArgs, AiModelCallbackArgs]
    ) -> Callable[[Any], None]:
        if isinstance(args, dai.CameraBoardSocket):
            return lambda packet: self._on_camera_frame(packet, args)  # type: ignore[arg-type]
        elif isinstance(args, DepthCallbackArgs):
            return lambda packet: self._on_stereo_frame(packet, args)  # type: ignore[arg-type]
        elif isinstance(args, AiModelCallbackArgs):
            callback: Callable[[Any, AiModelCallbackArgs], None] = self._on_detections
            if args.model_name == "age-gender-recognition-retail-0013":
                callback = self._on_age_gender_packet
            return lambda packet: callback(packet, args)  # type: ignore[arg-type]
        raise ValueError(f"Unknown callback args type: {type(args)}")

    def _on_camera_frame(self, packet: FramePacket, board_socket: dai.CameraBoardSocket) -> None:
        viewer.log_rigid3(f"{board_socket.name}/transform", child_from_parent=([0, 0, 0], self._ahrs.Q), xyz="RDF")

        img_frame = packet.frame if packet.msg.getType() == dai.RawImgFrame.Type.RAW8 else packet.msg.getData()
        h, w = packet.msg.getHeight(), packet.msg.getWidth()
        if packet.msg.getType() == dai.ImgFrame.Type.BITSTREAM:
            img_frame = cv2.cvtColor(
                self._jpeg_decoder.decode(img_frame, flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT), cv2.COLOR_BGR2RGBA
            )
            h, w = img_frame.shape[:2]

        child_from_parent: NDArray[np.float32]
        try:
            child_from_parent = self._get_camera_intrinsics(  # type: ignore[call-arg, misc, arg-type]
                board_socket, w, h  # type: ignore[call-arg, misc, arg-type]
            )
        except Exception:
            f_len = (w * h) ** 0.5
            child_from_parent = np.array([[f_len, 0, w / 2], [0, f_len, h / 2], [0, 0, 1]])
        cam = cam_kind_from_frame_type(packet.msg.getType())
        viewer.log_pinhole(
            f"{board_socket.name}/transform/{cam}/",
            child_from_parent=child_from_parent,
            width=w,
            height=h,
        )
        entity_path = f"{board_socket.name}/transform/{cam}/Image"

        if packet.msg.getType() == dai.RawImgFrame.Type.NV12:
            viewer.log_encoded_image(
                entity_path,
                img_frame,
                width=w,
                height=h,
                encoding=viewer.ImageEncoding.NV12,
            )
        else:
            viewer.log_image(entity_path, img_frame)

    def on_imu(self, packet: IMUPacket) -> None:
        for data in packet.data:
            gyro: dai.IMUReportGyroscope = data.gyroscope
            accel: dai.IMUReportAccelerometer = data.acceleroMeter
            mag: dai.IMUReportMagneticField = data.magneticField
            # TODO(filip): Move coordinate mapping to sdk
            self._ahrs.Q = self._ahrs.updateIMU(
                self._ahrs.Q, np.array([gyro.z, gyro.x, gyro.y]), np.array([accel.z, accel.x, accel.y])
            )
        if Topic.ImuData not in self.store.subscriptions:
            return
        viewer.log_imu([accel.z, accel.x, accel.y], [gyro.z, gyro.x, gyro.y], self._ahrs.Q, [mag.x, mag.y, mag.z])

    def _on_stereo_frame(self, packet: DepthPacket, args: DepthCallbackArgs) -> None:
        depth_frame = packet.frame
        cam = cam_kind_from_sensor_kind(args.alignment_camera.kind)
        path = f"{args.alignment_camera.board_socket.name}/transform/{cam}" + "/Depth"
        if not self.store.pipeline_config or not self.store.pipeline_config.depth:
            # Essentially impossible to get here
            return
        viewer.log_depth_image(path, depth_frame, meter=1e3)

    def _on_detections(self, packet: DetectionPacket, args: AiModelCallbackArgs) -> None:
        rects, colors, labels = self._detections_to_rects_colors_labels(packet, args.labels)
        cam = cam_kind_from_sensor_kind(args.camera.kind)
        viewer.log_rects(
            f"{args.camera.board_socket.name}/transform/{cam}/Detections",
            rects,
            rect_format=RectFormat.XYXY,
            colors=colors,
            labels=labels,
        )

    def _detections_to_rects_colors_labels(
        self, packet: DetectionPacket, omz_labels: Optional[List[str]] = None
    ) -> Tuple[List[List[int]], List[List[int]], List[str]]:
        rects = []
        colors = []
        labels = []
        for detection in packet.detections:
            rects.append(self._rect_from_detection(detection))
            colors.append([0, 255, 0])
            label: str = detection.label
            # Open model zoo models output label index
            if omz_labels is not None and isinstance(label, int):
                label += omz_labels[label]
            label += ", " + str(int(detection.img_detection.confidence * 100)) + "%"
            labels.append(label)
        return rects, colors, labels

    def _on_age_gender_packet(self, packet: TwoStagePacket, args: AiModelCallbackArgs) -> None:
        for det, rec in zip(packet.detections, packet.nnData):
            age = int(float(np.squeeze(np.array(rec.getLayerFp16("age_conv3")))) * 100)
            gender = np.squeeze(np.array(rec.getLayerFp16("prob")))
            gender_str = "Woman" if gender[0] > gender[1] else "Man"
            label = f"{gender_str}, {age}"
            color = [255, 0, 0] if gender[0] > gender[1] else [0, 0, 255]
            # TODO(filip): maybe use viewer.log_annotation_context to log class colors for detections

            cam = cam_kind_from_sensor_kind(args.camera.kind)
            viewer.log_rect(
                f"{args.camera.board_socket.name}/transform/{cam}/Detection",
                self._rect_from_detection(det),
                rect_format=RectFormat.XYXY,
                color=color,
                label=label,
            )

    def _rect_from_detection(self, detection: _Detection) -> List[int]:
        return [
            *detection.bottom_right,
            *detection.top_left,
        ]


def cam_kind_from_frame_type(dtype: dai.RawImgFrame.Type) -> str:
    """Returns camera kind string for given dai.RawImgFrame.Type."""
    return "mono_cam" if dtype == dai.RawImgFrame.Type.RAW8 else "color_cam"


def cam_kind_from_sensor_kind(kind: dai.CameraSensorType) -> str:
    """Returns camera kind string for given sensor type."""
    return "mono_cam" if kind == dai.CameraSensorType.MONO else "color_cam"
