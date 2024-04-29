from typing import Callable, List, Optional, Tuple, Union

import cv2
import depthai as dai
import numpy as np
from ahrs.filters import Mahony
from depthai_sdk.classes.packets import (  # PointcloudPacket,
    BasePacket,
    BoundingBox,
    DepthPacket,
    Detection,
    DetectionPacket,
    DisparityDepthPacket,
    FramePacket,
    IMUPacket,
    TwoStagePacket,
)
from depthai_sdk.components import (
    CameraComponent,
    Component,
    NNComponent,
    StereoComponent,
)
from depthai_sdk.components.tof_component import ToFComponent
from numpy.typing import NDArray
from pydantic import BaseModel

import depthai_viewer as viewer
from depthai_viewer._backend.store import Store
from depthai_viewer._backend.topic import Topic
from depthai_viewer.components.rect2d import RectFormat


class PacketHandlerContext(BaseModel):  # type: ignore[misc]
    class Config:
        arbitrary_types_allowed = True


class DetectionContext(PacketHandlerContext):
    labels: List[str]
    frame_width: int
    frame_height: int
    board_socket: dai.CameraBoardSocket


class PacketHandler:
    store: Store
    _ahrs: Mahony
    _get_camera_intrinsics: Callable[[dai.CameraBoardSocket, int, int], NDArray[np.float32]]

    def __init__(
        self, store: Store, intrinsics_getter: Callable[[dai.CameraBoardSocket, int, int], NDArray[np.float32]]
    ):
        viewer.init(f"Depthai Viewer {store.viewer_address}")
        print("Connecting to viewer at", store.viewer_address)
        viewer.connect(store.viewer_address)
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
        # type: ignore[assignment, misc]
        self._get_camera_intrinsics = camera_intrinsics_getter

    def log_dai_packet(self, node: dai.Node, packet: dai.Buffer, context: Optional[PacketHandlerContext]) -> None:
        if isinstance(packet, dai.ImgFrame):
            board_socket = None
            if isinstance(node, dai.node.ColorCamera):
                board_socket = node.getBoardSocket()
            elif isinstance(node, dai.node.MonoCamera):
                board_socket = node.getBoardSocket()
            elif isinstance(node, dai.node.Camera):
                board_socket = node.getBoardSocket()
            if board_socket is not None:
                self._on_camera_frame(FramePacket("", packet), board_socket)
            else:
                print("Unknown node type:", type(node), "for packet:", type(packet))
        elif isinstance(packet, dai.ImgDetections):
            if context is None or not isinstance(context, DetectionContext):
                print("Invalid context for detections packet", context)
                return
            self._on_dai_detections(packet, context)
        else:
            print("Unknown dai packet type:", type(packet))

    def _dai_detections_to_rects_colors_labels(
        self, packet: dai.ImgDetections, context: DetectionContext
    ) -> Tuple[List[List[int]], List[List[int]], List[str]]:
        rects = []
        colors = []
        labels = []
        for detection in packet.detections:
            rects.append(self._rect_from_detection(detection, context.frame_height, context.frame_width))
            colors.append([0, 255, 0])
            label = ""
            # Open model zoo models output label index
            if context.labels is not None:
                label += context.labels[detection.label]
            label += ", " + str(int(detection.confidence * 100)) + "%"
            labels.append(label)
        return rects, colors, labels
        pass

    def _on_dai_detections(self, packet: dai.ImgDetections, context: DetectionContext) -> None:
        packet.detections
        rects, colors, labels = self._dai_detections_to_rects_colors_labels(packet, context)
        viewer.log_rects(
            f"{context.board_socket.name}/transform/color_cam/Detections",
            rects,
            rect_format=RectFormat.XYXY,
            colors=colors,
            labels=labels,
        )

    def log_packet(
        self,
        component: Component,
        packet: BasePacket,
    ) -> None:
        if type(packet) is FramePacket:
            if isinstance(component, CameraComponent):
                self._on_camera_frame(packet, component._socket)
            else:
                print("Unknown component type:", type(component), "for packet:", type(packet))
            # Create dai.CameraBoardSocket from descriptor
        elif type(packet) is DepthPacket:
            if isinstance(component, StereoComponent):
                self._on_stereo_frame(packet, component)
        elif type(packet) is DisparityDepthPacket:
            if isinstance(component, ToFComponent):
                self._on_tof_packet(packet, component)
            elif isinstance(component, StereoComponent):
                self._on_stereo_frame(packet, component)
            else:
                print("Unknown component type:", type(component), "for packet:", type(packet))
        elif type(packet) is DetectionPacket:
            self._on_detections(packet, component)
        elif type(packet) is TwoStagePacket:
            self._on_age_gender_packet(packet, component)
        else:
            print("Unknown packet type:", type(packet))

    def _on_camera_frame(self, packet: FramePacket, board_socket: dai.CameraBoardSocket) -> None:
        viewer.log_rigid3(
            f"{board_socket.name}/transform", child_from_parent=([0, 0, 0], [1, 0, 0, 0]), xyz="RDF"
        )  # TODO(filip): Enable the user to lock the camera rotation in the UI

        img_frame = packet.frame if packet.msg.getType() == dai.RawImgFrame.Type.RAW8 else packet.msg.getData()
        h, w = packet.msg.getHeight(), packet.msg.getWidth()
        if packet.msg.getType() == dai.ImgFrame.Type.BITSTREAM:
            img_frame = cv2.cvtColor(cv2.imdecode(img_frame, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
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
        elif packet.msg.getType() == dai.RawImgFrame.Type.GRAYF16:
            img = img_frame.view(np.float16).reshape(h, w)
            viewer.log_image(entity_path, img, colormap=viewer.Colormap.Magma, unit="°C")
        else:
            viewer.log_image(entity_path, img_frame)

    def on_imu(self, packet: IMUPacket) -> None:
        gyro: dai.IMUReportGyroscope = packet.gyroscope
        accel: dai.IMUReportAccelerometer = packet.acceleroMeter
        mag: dai.IMUReportMagneticField = packet.magneticField
        # TODO(filip): Move coordinate mapping to sdk
        self._ahrs.Q = self._ahrs.updateIMU(
            self._ahrs.Q, np.array([gyro.z, gyro.x, gyro.y]), np.array([accel.z, accel.x, accel.y])
        )
        if Topic.ImuData not in self.store.subscriptions:
            return
        viewer.log_imu([accel.z, accel.x, accel.y], [gyro.z, gyro.x, gyro.y], self._ahrs.Q, [mag.x, mag.y, mag.z])

    def _on_stereo_frame(self, packet: Union[DepthPacket, DisparityDepthPacket], component: StereoComponent) -> None:
        depth_frame = packet.frame
        cam = "color_cam" if component._align_component.is_color() else "mono_cam"
        path = f"{component._align_component._socket.name}/transform/{cam}" + "/Depth"
        if not self.store.pipeline_config or not self.store.pipeline_config.stereo:
            # Essentially impossible to get here
            return
        viewer.log_depth_image(path, depth_frame, meter=1e3)

    def _on_tof_packet(
        self,
        packet: DisparityDepthPacket,
        component: ToFComponent,
    ) -> None:
        depth_frame = packet.frame
        viewer.log_rigid3(
            f"{component.camera_socket.name}/transform", child_from_parent=([0, 0, 0], [1, 0, 0, 0]), xyz="RDF"
        )
        try:
            intrinsics = self._get_camera_intrinsics(component.camera_socket, 640, 480)
        except Exception:
            intrinsics = np.array([[471.451, 0.0, 317.897], [0.0, 471.539, 245.027], [0.0, 0.0, 1.0]])
        viewer.log_pinhole(
            f"{component.camera_socket.name}/transform/tof",
            child_from_parent=intrinsics,
            width=component.camera_node.getVideoWidth(),
            height=component.camera_node.getVideoHeight(),
        )

        path = f"{component.camera_socket.name}/transform/tof/Depth"
        viewer.log_depth_image(
            path,
            depth_frame,
            meter=1e3,
            min=200.0,
            max=1874
            * (
                (
                    self.store.tof_config.phaseUnwrappingLevel  # type: ignore[attr-defined]
                    if self.store.tof_config
                    else 4.0
                )
                + 1
            ),
        )

    def _on_detections(self, packet: DetectionPacket, component: NNComponent) -> None:
        rects, colors, labels = self._detections_to_rects_colors_labels(packet, component.get_labels())
        cam = "color_cam" if component._get_camera_comp().is_color() else "mono_cam"
        viewer.log_rects(
            f"{component._get_camera_comp()._socket.name}/transform/{cam}/Detections",
            rects,
            rect_format=RectFormat.XYXY,
            colors=colors,
            labels=labels,
        )

    def _rect_from_sdk_detection(
        self, packet_bbox: BoundingBox, detection: Detection, max_height: int, max_width: int
    ) -> List[int]:
        bbox = packet_bbox.get_relative_bbox(detection.bbox)
        (x1, y1), (x2, y2) = bbox.denormalize((max_height, max_width))
        return [
            max(x1, 0),
            max(y1, 0),
            min(x2, max_width),
            min(y2, max_height),
        ]

    def _detections_to_rects_colors_labels(
        self, packet: DetectionPacket, omz_labels: Optional[List[str]] = None
    ) -> Tuple[List[List[int]], List[List[int]], List[str]]:
        rects = []
        colors = []
        labels = []
        for detection in packet.detections:
            rects.append(
                self._rect_from_sdk_detection(packet.bbox, detection, packet.frame.shape[0], packet.frame.shape[1])
            )
            colors.append([0, 255, 0])
            label: str = detection.label_str
            # Open model zoo models output label index
            if omz_labels is not None and isinstance(label, int):
                label += omz_labels[label]
            label += ", " + str(int(detection.img_detection.confidence * 100)) + "%"
            labels.append(label)
        return rects, colors, labels

    def _on_age_gender_packet(self, packet: TwoStagePacket, component: NNComponent) -> None:
        for det, rec in zip(packet.detections, packet.nnData):
            age = int(float(np.squeeze(np.array(rec.getLayerFp16("age_conv3")))) * 100)
            gender = np.squeeze(np.array(rec.getLayerFp16("prob")))
            gender_str = "Woman" if gender[0] > gender[1] else "Man"
            label = f"{gender_str}, {age}"
            color = [255, 0, 0] if gender[0] > gender[1] else [0, 0, 255]
            # TODO(filip): maybe use viewer.log_annotation_context to log class colors for detections

            cam = "color_cam" if component._get_camera_comp().is_color() else "mono_cam"
            viewer.log_rect(
                f"{component._get_camera_comp()._socket.name}/transform/{cam}/Detection",
                self._rect_from_sdk_detection(packet.bbox, det, packet.frame.shape[0], packet.frame.shape[1]),
                rect_format=RectFormat.XYXY,
                color=color,
                label=label,
            )

    def _rect_from_detection(self, detection: dai.ImgDetection, max_height: int, max_width: int) -> List[int]:
        return [
            int(min(max(detection.xmin, 0.0), 1.0) * max_width),
            int(min(max(detection.ymin, 0.0), 1.0) * max_height),
            int(min(max(detection.xmax, 0.0), 1.0) * max_width),
            int(min(max(detection.ymax, 0.0), 1.0) * max_height),
        ]


def cam_kind_from_frame_type(dtype: dai.RawImgFrame.Type) -> str:
    """Returns camera kind string for given dai.RawImgFrame.Type."""
    return "mono_cam" if dtype == dai.RawImgFrame.Type.RAW8 else "color_cam"


def cam_kind_from_sensor_kind(kind: dai.CameraSensorType) -> str:
    """Returns camera kind string for given sensor type."""
    return "mono_cam" if kind == dai.CameraSensorType.MONO else "color_cam"
