from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import depthai as dai
import numpy as np
from ahrs.filters import Mahony
from depthai_sdk.classes.packets import (
    DepthPacket,
    DetectionPacket,
    FramePacket,
    IMUPacket,
    # PointcloudPacket,
    TwoStagePacket,
    _Detection,
)
from numpy.typing import NDArray
from pydantic import BaseModel

import depthai_viewer as viewer
from depthai_viewer._backend.device_configuration import CameraConfiguration
from depthai_viewer._backend.store import Store
from depthai_viewer._backend.topic import Topic
from depthai_viewer.components.rect2d import RectFormat
from depthai_viewer import bindings


class QueueContext:
    cam: Optional[CameraConfiguration]
    encoding: bool = False

    def __init__(self, cam: Optional[CameraConfiguration] = None, encoding: bool = False):
        self.cam = cam
        self.encoding = encoding


class AiQueueContext(QueueContext):
    model_path = str
    labels = Optional[List[str]]

    def __init__(self, model_path: str, labels: Optional[List[str]] = None, cam: Optional[CameraConfiguration] = None):
        super().__init__(cam)
        self.model_path = model_path
        self.labels = labels


class DepthQueueContext(QueueContext):
    stereo_pair: Tuple[dai.CameraBoardSocket, dai.CameraBoardSocket]

    def __init__(
        self,
        stereo_pair: Tuple[dai.CameraBoardSocket, dai.CameraBoardSocket],
        cam: Optional[CameraConfiguration] = None,
    ):
        """
        cam is the camera that the depth is aligned to.
        """
        super().__init__(cam)
        self.stereo_pair = stereo_pair


class PacketHandler:
    store: Store
    _ahrs: Mahony
    _get_camera_intrinsics: Callable[[dai.CameraBoardSocket, int, int], NDArray[np.float32]]
    _queues: List[Tuple[QueueContext, Queue]] = []

    def __init__(
        self, store: Store, intrinsics_getter: Callable[[dai.CameraBoardSocket, int, int], NDArray[np.float32]]
    ):
        viewer.init("Depthai Viewer")
        viewer.connect()
        self.store = store
        self._ahrs = Mahony(frequency=100)
        self._ahrs.Q = np.array([1, 0, 0, 0], dtype=np.float64)
        self._get_camera_intrinsics = intrinsics_getter

    def update(self) -> None:
        for descriptor, q in self._queues:
            try:
                packet = q.get(timeout=0.1)
            except Empty:
                continue

            if type(packet) == FramePacket:
                self._on_camera_frame(packet, descriptor)
            elif type(packet) == DepthPacket:
                self._on_stereo_frame(packet, descriptor)
            elif type(packet) == DetectionPacket:
                self._on_detections(packet, descriptor)
            elif type(packet) == TwoStagePacket:
                if (
                    descriptor.model_name == "age-gender-recognition-retail-0013"
                ):  # TODO(filip): Not the best for performance, improve.
                    self._on_age_gender_packet(packet, descriptor)
                else:
                    print("Skipping unknown two stage packet", descriptor.model_name)
            elif type(packet) == IMUPacket:
                self._on_imu(packet)
            else:
                print("Skipping unknown packet type", type(packet))

    def add_queue(self, descriptor: QueueContext, queue: Queue) -> None:
        self._queues.append((descriptor, queue))

    def clear_queues(self) -> None:
        print("Clearing queues")
        self._queues = []

    # def _on_camera_frame(self, packet: dai.ImgFrame, ctx: QueueContext) -> None:
    #     viewer.log_rigid3(
    #         f"{ctx.cam.board_socket.name}/transform", child_from_parent=([0, 0, 0], self._ahrs.Q), xyz="RDF"
    #     )
    #     h, w = packet.getHeight(), packet.getWidth()
    #     child_from_parent: NDArray[np.float32]
    #     try:
    #         child_from_parent = self._get_camera_intrinsics(ctx.cam.board_socket, w, h)
    #     except Exception:
    #         f_len = (w * h) ** 0.5
    #         child_from_parent = np.array([[f_len, 0, w / 2], [0, f_len, h / 2], [0, 0, 1]])
    #     cam = cam_kind(ctx.cam.kind)
    #     viewer.log_pinhole(
    #         f"{ctx.cam.board_socket.name}/transform/{cam}/",
    #         child_from_parent=child_from_parent,
    #         width=w,
    #         height=h,
    #     )
    #     img_frame = packet.getCvFrame() if ctx.cam.kind == dai.CameraSensorType.MONO else packet.getData()
    #     entity_path = f"{ctx.cam.board_socket.name}/transform/{cam}/Image"
    #     if ctx.encoding:
    #         viewer.log_image_file(entity_path, img_bytes=img_frame, img_format=viewer.ImageFormat.JPEG)
    #     elif packet.getType() == dai.RawImgFrame.Type.NV12:
    #         viewer.log_encoded_image(
    #             entity_path,
    #             img_frame,
    #             width=w,
    #             height=h,
    #             encoding=None if ctx.cam.kind == dai.CameraSensorType.MONO else viewer.ImageEncoding.NV12,
    #         )
    #     else:
    #         viewer.log_image(entity_path, img_frame)

    def _on_camera_frame(self, packet: FramePacket, ctx: QueueContext) -> None:
        viewer.log_rigid3(
            f"{ctx.cam.board_socket.name}/transform", child_from_parent=([0, 0, 0], self._ahrs.Q), xyz="RDF"
        )
        h, w = packet.frame.shape[:2]
        child_from_parent: NDArray[np.float32]
        try:
            child_from_parent = self._get_camera_intrinsics(ctx.cam.board_socket, w, h)
        except Exception:
            f_len = (w * h) ** 0.5
            child_from_parent = np.array([[f_len, 0, w / 2], [0, f_len, h / 2], [0, 0, 1]])
        cam = cam_kind(ctx.cam.kind)
        viewer.log_pinhole(
            f"{ctx.cam.board_socket.name}/transform/{cam}/",
            child_from_parent=child_from_parent,
            width=w,
            height=h,
        )
        img_frame = packet.frame if ctx.cam.kind == dai.CameraSensorType.MONO else packet.msg.getData()
        entity_path = f"{ctx.cam.board_socket.name}/transform/{cam}/Image"
        if ctx.encoding:
            viewer.log_image_file(entity_path, img_bytes=img_frame, img_format=viewer.ImageFormat.JPEG)
        elif packet.msg.getType() == dai.RawImgFrame.Type.NV12:
            viewer.log_encoded_image(
                entity_path,
                img_frame,
                width=w,
                height=h,
                encoding=None if ctx.cam.kind == dai.CameraSensorType.MONO else viewer.ImageEncoding.NV12,
            )
        else:
            viewer.log_image(entity_path, img_frame)

    def _on_imu(self, packet: IMUPacket) -> None:
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

    def _on_stereo_frame(self, frame: DepthPacket, ctx: DepthQueueContext) -> None:
        depth_frame = frame.frame
        cam = cam_kind(ctx.cam.kind)
        path = f"{ctx.cam.board_socket.name}/transform/{cam}" + "/Depth"
        if not self.store.pipeline_config or not self.store.pipeline_config.depth:
            # Essentially impossible to get here
            return
        viewer.log_depth_image(path, depth_frame, meter=1e3)

    def _on_detections(self, packet: DetectionPacket, ctx: AiQueueContext) -> None:
        rects, colors, labels = self._detections_to_rects_colors_labels(packet, ctx.labels)
        cam = cam_kind(ctx.cam.kind)
        viewer.log_rects(
            f"{ctx.cam.board_socket.name}/transform/{cam}/Detections",
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

    def _on_age_gender_packet(self, packet: TwoStagePacket, ctx: AiQueueContext) -> None:
        for det, rec in zip(packet.detections, packet.nnData):
            age = int(float(np.squeeze(np.array(rec.getLayerFp16("age_conv3")))) * 100)
            gender = np.squeeze(np.array(rec.getLayerFp16("prob")))
            gender_str = "Woman" if gender[0] > gender[1] else "Man"
            label = f"{gender_str}, {age}"
            color = [255, 0, 0] if gender[0] > gender[1] else [0, 0, 255]
            # TODO(filip): maybe use viewer.log_annotation_context to log class colors for detections

            cam = cam_kind(ctx.cam.kind)
            viewer.log_rect(
                f"{ctx.cam.board_socket.name}/transform/{cam}/Detection",
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


def cam_kind(sensor: dai.CameraSensorType) -> str:
    """Returns camera kind string for given sensor type."""
    return "mono_cam" if sensor == dai.CameraSensorType.MONO else "color_cam"
