from typing import Dict, List, Optional, Tuple, Union
import time
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
from depthai_viewer._backend.obscured_utilities.utilities.calibration_handler import Recalibration
from depthai_viewer._backend.obscured_utilities.utilities.display_handler import Display
import os

class PacketHandlerContext(BaseModel):  # type: ignore[misc]
    class Config:
        arbitrary_types_allowed = True


class DetectionContext(PacketHandlerContext):
    labels: List[str]
    frame_width: int
    frame_height: int
    board_socket: dai.CameraBoardSocket


class CachedCalibrationHandler:
    calibration_handler: dai.CalibrationHandler
    intrinsic_matrix: Dict[Tuple[dai.CameraBoardSocket, int, int], NDArray[np.float32]] = {}
    distortion_coefficients: Dict[dai.CameraBoardSocket, NDArray[np.float32]] = {}

    def __init__(self, calibration_handler: dai.CalibrationHandler):
        self.calibration_handler = calibration_handler

    def get_intrinsic_matrix(self, board_socket: dai.CameraBoardSocket, width: int, height: int) -> NDArray[np.float32]:
        if self.intrinsic_matrix.get((board_socket, width, height)) is not None:
            return self.intrinsic_matrix.get((board_socket, width, height))  # type: ignore[return-value]
        try:
            M = self.calibration_handler.getCameraIntrinsics(  # type: ignore[union-attr]
                board_socket, dai.Size2f(width, height)
            )
        except RuntimeError:
            print("No intrinsics found for camera: ", board_socket, " assuming default.")
            f_len = (height * width) ** 0.5
            M = [[f_len, 0, width / 2], [0, f_len, height / 2], [0, 0, 1]]
        self.intrinsic_matrix[(board_socket, width, height)] = np.array(M).reshape(3, 3)
        return self.intrinsic_matrix[(board_socket, width, height)]

    def get_distortion_coefficients(self, board_socket: dai.CameraBoardSocket) -> NDArray[np.float32]:
        if self.distortion_coefficients.get(board_socket) is not None:
            return self.distortion_coefficients.get(board_socket)  # type: ignore[return-value]
        try:
            D = self.calibration_handler.getDistortionCoefficients(board_socket)  # type: ignore[union-attr]
        except RuntimeError:
            print("No distortion coefficients found for camera: ", board_socket, " assuming default.")
            D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.distortion_coefficients[board_socket] = np.array(D)
        return self.distortion_coefficients[board_socket]


class PacketHandler:
    store: Store
    _ahrs: Mahony
    _calibration_handler: CachedCalibrationHandler
    _dynamic_recalibration: Recalibration = None
    _display: Display = None

    def __init__(self, store: Store, calibration_handler: dai.CalibrationHandler, factoryCalibration_handler: dai.CalibrationHandler, stereo: False):
        viewer.init(f"Depthai Viewer {store.viewer_address}")
        print("Connecting to viewer at", store.viewer_address)
        viewer.connect(store.viewer_address)
        self.store = store
        self._ahrs = Mahony(frequency=100)
        self._ahrs.Q = np.array([1, 0, 0, 0], dtype=np.float64)
        self._calibration_handler = CachedCalibrationHandler(calibration_handler)
        self.stereo = stereo
        self.save_diagnostics = ""
        self.diagnostics_display = False
        if self.stereo:
            self._dynamic_recalibration = Recalibration(calibration_handler, factoryCalibration_handler)
            self._dynamic_recalibration.min_pts_for_calib = 3500
            self._display = Display()
            self.new_calib = None
            self.flashCalibration = False
            self.resetFactoryCalibration = False
            self._display_flashing = ""
            self._calib_time = None
            self.display_bar = False
            self.display_text = None

    def reset(self) -> None:
        self._ahrs = Mahony(frequency=100)
        self._ahrs.Q = np.array([1, 0, 0, 0], dtype=np.float64)

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

    def _log_img_frame(
        self,
        frame: dai.ImgFrame,
        board_socket: dai.CameraBoardSocket,
        intrinsics_matrix: Optional[NDArray[np.float32]] = None,
        distortion_coefficients: Optional[NDArray[np.float32]] = None,
    ) -> None:
        """
        Log an image frame to the viewer.

        Optionally undistort and rectify the image using the provided intrinsic matrix
        and distortion coefficients.
        """
        viewer.log_rigid3(
            f"{board_socket.name}/transform", child_from_parent=([0, 0, 0], [1, 0, 0, 0]), xyz="RDF"
        )  # TODO(filip): Enable the user to lock the camera rotation in the UI

        img_frame = (
            frame.getCvFrame()
            if frame.getType() == dai.RawImgFrame.Type.RAW8 or frame.getType() == dai.RawImgFrame.Type.YUV420p
            else frame.getData()
        )
        h, w = frame.getHeight(), frame.getWidth()
        was_undistorted_and_rectified = False
        # If the image is a cv frame try to undistort and rectify it
        # Currently this is blocking the code and so it introcudes delay in stream, need to make it better (Matic)
        if intrinsics_matrix is not None and distortion_coefficients is not None:
            if frame.getType() == dai.RawImgFrame.Type.NV12:
                img_frame = cv2.cvtColor(frame.getCvFrame(), cv2.COLOR_BGR2RGB)
            map_x, map_y = cv2.initUndistortRectifyMap(
                intrinsics_matrix, distortion_coefficients, None, intrinsics_matrix, (w, h), cv2.CV_32FC1
            )
            img_frame = cv2.remap(img_frame, map_x, map_y, cv2.INTER_LINEAR)
            was_undistorted_and_rectified = True

        if frame.getType() == dai.ImgFrame.Type.BITSTREAM:
            img_frame = cv2.cvtColor(cv2.imdecode(img_frame, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            h, w = img_frame.shape[:2]

        child_from_parent: NDArray[np.float32]
        try:
            child_from_parent = (
                self._calibration_handler.get_intrinsic_matrix(  # type: ignore[call-arg, misc, arg-type]
                    board_socket, w, h  # type: ignore[call-arg, misc, arg-type]
                )
            )
        except Exception:
            f_len = (w * h) ** 0.5
            child_from_parent = np.array([[f_len, 0, w / 2], [0, f_len, h / 2], [0, 0, 1]])
        cam = cam_kind_from_frame_type(frame.getType())
        viewer.log_pinhole(
            f"{board_socket.name}/transform/{cam}/",
            child_from_parent=child_from_parent,
            width=w,
            height=h,
        )
        entity_path = f"{board_socket.name}/transform/{cam}/Image"
        if self.stereo:
            is_left_socket = board_socket.name == str(self._dynamic_recalibration.left_socket.name)
            is_right_socket = board_socket.name == str(self._dynamic_recalibration.right_socket.name)

            # Handle frames when feature collection is disabled
            if not self._dynamic_recalibration.collect_features and not self._dynamic_recalibration.recalibrating:
                if frame.getType() == dai.RawImgFrame.Type.NV12 and not was_undistorted_and_rectified:
                    encoding = viewer.ImageEncoding.NV12
                    if (is_left_socket or is_right_socket) and self.display_bar:
                        img_frame = self._display.draw_health_bar(cv2.cvtColor(img_frame, cv2.COLOR_GRAY2BGR),self.error_left * 1.2, display_text=self.display_text)
                        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
                    viewer.log_encoded_image(entity_path, img_frame, width=w, height=h, encoding=encoding)
                elif frame.getType() == dai.RawImgFrame.Type.YUV420p:
                    if (is_left_socket or is_right_socket) and self.display_bar:
                        img_frame = self._display.draw_health_bar(cv2.cvtColor(img_frame, cv2.COLOR_GRAY2BGR),self.error_left * 1.2, display_text=self.display_text)
                        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
                    viewer.log_image(entity_path, cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
                elif frame.getType() == dai.RawImgFrame.Type.GRAYF16:
                    img = img_frame.view(np.float16).reshape(h, w)
                    if (is_left_socket or is_right_socket) and self.display_bar:
                        self._display.draw_health_bar(img_frame, self.error_left*1.2,display_text=self.display_text)
                        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
                    viewer.log_image(entity_path, img, colormap=viewer.Colormap.Magma, unit="°C")
                else:
                    if (is_left_socket or is_right_socket) and self.display_bar:
                        img_frame = self._display.draw_health_bar(cv2.cvtColor(img_frame, cv2.COLOR_GRAY2BGR),self.error_left * 1.2, display_text=self.display_text)
                        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
                        self._dynamic_recalibration.resolution = img_frame.shape[1::-1]
                    elif (is_left_socket or is_right_socket) and self._display_flashing != "":
                        self._display.draw_center_center_box(img_frame,f"Flashing {self._display_flashing} calibration ...")

                    elif (is_left_socket or is_right_socket) and self.diagnostics_display:
                        self._display.draw_center_center_box(img_frame,f"Saving to:")
                        self.draw_center_center_box(img_frame, self.file_path)
                    viewer.log_image(entity_path, img_frame)
                if is_left_socket:
                    self._display.create_window((cv2.resize(frame.getCvFrame(), (640,400)).shape[1], cv2.resize(frame.getCvFrame(), (640,400)).shape[0]))
                    self._dynamic_recalibration.frame_left = cv2.cvtColor(cv2.resize(frame.getCvFrame(), (640,400)), cv2.COLOR_GRAY2BGR)
                if is_right_socket:
                    self._display.create_window((cv2.resize(frame.getCvFrame(), (640,400)).shape[1], cv2.resize(frame.getCvFrame(), (640,400)).shape[0]))
                    self._dynamic_recalibration.frame_right = cv2.cvtColor(cv2.resize(frame.getCvFrame(), (640,400)), cv2.COLOR_GRAY2BGR)
                    self._dynamic_recalibration.resolution = self._dynamic_recalibration.frame_right.shape[1::-1]

            # Handle frames when feature collection is enabled
            elif self._dynamic_recalibration.collect_features or self._dynamic_recalibration.recalibrating:
                if not is_left_socket and not is_right_socket:
                    # Process frames for sockets that are not left or right
                    if frame.getType() == dai.RawImgFrame.Type.NV12 and not was_undistorted_and_rectified:
                        encoding = viewer.ImageEncoding.NV12
                        viewer.log_encoded_image(entity_path, img_frame, width=w, height=h, encoding=encoding)
                    elif frame.getType() == dai.RawImgFrame.Type.YUV420p:
                        viewer.log_image(entity_path, cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
                    elif frame.getType() == dai.RawImgFrame.Type.GRAYF16:
                        img = img_frame.view(np.float16).reshape(h, w)
                        viewer.log_image(entity_path, img, colormap=viewer.Colormap.Magma, unit="°C")
                    else:
                        viewer.log_image(entity_path, img_frame)
                else:
                    # Handle left socket
                    if is_left_socket:
                        frame = cv2.cvtColor(frame.getCvFrame(), cv2.COLOR_GRAY2BGR)
                        self._display.create_window((cv2.resize(frame, (640,400)).shape[1], cv2.resize(frame, (640,400)).shape[0]))
                        self._dynamic_recalibration.frame_left = cv2.resize(frame, (640,400))
                        if self._dynamic_recalibration.collect_features:
                            self._display_and_log_frame(self._dynamic_recalibration.left_socket.name, cv2.resize(frame, (640,400)), self._dynamic_recalibration.new_pts_left)
                        elif self._dynamic_recalibration.recalibrating:
                            self._display.draw_center_center_box(self._dynamic_recalibration.frame_left,"Recalibration in progress ...")
                            entity_path = f"{self._dynamic_recalibration.left_socket.name}/transform/mono_cam/Image"
                            viewer.log_image(entity_path, cv2.cvtColor(self._dynamic_recalibration.frame_left, cv2.COLOR_BGR2RGB))
                    # Handle right socket
                    elif is_right_socket:
                        frame = cv2.cvtColor(frame.getCvFrame(), cv2.COLOR_GRAY2BGR)
                        self._display.create_window((cv2.resize(frame, (640,400)).shape[1], cv2.resize(frame, (640,400)).shape[0]))
                        self._dynamic_recalibration.frame_right = cv2.resize(frame, (640,400))
                        if self._dynamic_recalibration.collect_features:
                            self._display_and_log_frame(self._dynamic_recalibration.right_socket.name, cv2.resize(frame, (640,400)))
                        elif self._dynamic_recalibration.recalibrating:
                            self._display.draw_center_center_box(self._dynamic_recalibration.frame_right,"Recalibration in progress ...")
                            entity_path = f"{self._dynamic_recalibration.right_socket.name}/transform/mono_cam/Image"
                            viewer.log_image(entity_path, cv2.cvtColor(self._dynamic_recalibration.frame_right, cv2.COLOR_BGR2RGB))

            if self._calib_time is not None and np.abs(time.time() - self._calib_time) > 5:
                if self._display_flashing != "":
                    self._display_flashing = ""
                elif self.display_bar:
                    self.display_bar = False
                    self._dynamic_recalibration.reset_aggregation()
                self._calib_time = None
                self.diagnostics_display = False


            if not self._dynamic_recalibration.result_queue.empty():
                self.new_calib, out_before, out_after, fillrate_before, fillrate_after = self._dynamic_recalibration.result_queue.get()
                self._dynamic_recalibration.stop_optimization()
                self._display.reset()
                _, _, self.error_left, _ = self._dynamic_recalibration.get_features(self._dynamic_recalibration.frame_left , self._dynamic_recalibration.frame_right)
                self.calibration_error = [self.error_left, _]
                self._calib_time = time.time()
                self.display_bar = True
                self.display_text = "Recalibration done, new depth will be displayed."
                print(f"Previous reprojection error: {np.median(out_before[4])}px")
                print(f"New reprojection error: {np.median(out_after[4])}px")
                print(f"Previous fillrate: {fillrate_before}%")
                print(f"New fillrate: {fillrate_after}%")
        else:
            if frame.getType() == dai.RawImgFrame.Type.NV12 and not was_undistorted_and_rectified:
                encoding = viewer.ImageEncoding.NV12
                viewer.log_encoded_image(entity_path, img_frame, width=w, height=h, encoding=encoding)
            elif frame.getType() == dai.RawImgFrame.Type.YUV420p:
                viewer.log_image(entity_path, cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
            elif frame.getType() == dai.RawImgFrame.Type.GRAYF16:
                img = img_frame.view(np.float16).reshape(h, w)
                viewer.log_image(entity_path, img, colormap=viewer.Colormap.Magma, unit="°C")
            else:
                viewer.log_image(entity_path, img_frame)
    def _on_camera_frame(self, packet: FramePacket, board_socket: dai.CameraBoardSocket) -> None:
        if board_socket in list(
            map(lambda cam: cam.tof_align, self.store.pipeline_config.cameras if self.store.pipeline_config else [])
        ):
            # Skip tof aligned frames - they will be logged on_tof_packet
            return
        intrinsics = None
        distortion_coefficients = None
        if this_cam := list(filter(lambda cam: cam.board_socket == board_socket, self.store.pipeline_config.cameras)):
            if this_cam[0].is_used_as_stereo_align:
                intrinsics = self._calibration_handler.get_intrinsic_matrix(
                    board_socket, packet.msg.getWidth(), packet.msg.getHeight()
                )
                distortion_coefficients = self._calibration_handler.get_distortion_coefficients(board_socket)
        self._log_img_frame(packet.msg, board_socket, intrinsics, distortion_coefficients)

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
        if packet.aligned_frame:
            rgb_size = (packet.aligned_frame.getWidth(), packet.aligned_frame.getHeight())
            M = self._calibration_handler.get_intrinsic_matrix(
                dai.CameraBoardSocket(packet.aligned_frame.getInstanceNum()), *rgb_size
            )
            D = self._calibration_handler.get_distortion_coefficients(
                dai.CameraBoardSocket(packet.aligned_frame.getInstanceNum())
            )
            self._log_img_frame(
                packet.aligned_frame, dai.CameraBoardSocket(packet.aligned_frame.getInstanceNum()), M, D
            )
        depth_frame = packet.frame

        if packet.aligned_frame:
            ent_path_root = dai.CameraBoardSocket(packet.aligned_frame.getInstanceNum()).name
        else:
            ent_path_root = component.camera_socket.name
        ent_path_depth = f"{ent_path_root}/transform"
        if not packet.aligned_frame:
            viewer.log_rigid3(f"{ent_path_root}/transform", child_from_parent=([0, 0, 0], [1, 0, 0, 0]), xyz="RDF")
            try:
                intrinsics = self._calibration_handler.get_intrinsic_matrix(component.camera_socket, 640, 480)
            except Exception:
                intrinsics = np.array([[471.451, 0.0, 317.897], [0.0, 471.539, 245.027], [0.0, 0.0, 1.0]])
            viewer.log_pinhole(
                f"{ent_path_root}/transform/tof",
                child_from_parent=intrinsics,
                width=component.camera_node.getVideoWidth(),
                height=component.camera_node.getVideoHeight(),
            )
            ent_path_depth += "/tof/Depth"
        else:
            cam_kind = cam_kind_from_frame_type(packet.aligned_frame.getType())
            ent_path_depth += f"/{cam_kind}/Depth"
        viewer.log_depth_image(
            ent_path_depth,
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


    def _display_and_log_frame(self, side: str, frame: np.ndarray, features: Optional[np.ndarray] = None) -> None:
        """
        Display and log a frame with optional features overlay.
        """
        if features is not None:
            self._display.drawFeatures(frame, features)
            alpha = 0.25
            overlay_frame = cv2.addWeighted(self._display.overlay, alpha, frame, 1 - alpha, 0)
        else:
            overlay_frame = frame
        overlay_frame = self._display.draw_progress_bar_with_percentage(overlay_frame, len(self._dynamic_recalibration.aggregated_pts_left)/self._dynamic_recalibration.min_pts_for_calib, self._dynamic_recalibration.coverage_condition)

        entity_path = f"{side}/transform/mono_cam/Image"
        viewer.log_image(entity_path, cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB))

    def _start_calibration(self) -> None:
        """
        Start calibration when 'c' key is pressed.
        """
        _, _, self.error_left, _ = self._dynamic_recalibration.get_features(
            self._dynamic_recalibration.frame_left, self._dynamic_recalibration.frame_right
        )
        self._calib_time = time.time()
        self.display_bar = True
        self.display_text = "Calibration status."

    def _start_optimization(self) -> None:
        """
        Start optimization when 'r' key is pressed.
        """
        print("Starting feature collection and optimization...")
        self._dynamic_recalibration.start_optimization(8)

    def draw_center_center_box(self, image, text, box_color=(0, 0, 0, 128), font_scale=0.3, font_color=(255, 255, 255)):
        img_height, img_width = image.shape[:2]

        # Calculate text size
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        text_width, text_height = text_size

        # Define box dimensions
        box_padding = 10
        box_width = text_width + box_padding * 2
        box_height = text_height + box_padding * 2
        box_x = (img_width - box_width) // 2
        box_y = (img_height - box_height) // 2 +50

        # Draw semi-transparent background box
        overlay = image.copy()
        box_start = (box_x, box_y)
        box_end = (box_x + box_width, box_y + box_height)
        alpha = box_color[3] / 255.0
        cv2.rectangle(overlay, box_start, box_end, box_color[:3], -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Draw text at the center of the box
        text_x = box_x + (box_width - text_width) // 2
        text_y = box_y + (box_height + text_height) // 2
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)

        return image

    def _save_packet(self, name, frame):
        base_path = self.save_diagnostics
        self.file_path = os.path.join(base_path, f"{name}.png")

        counter = 1
        while os.path.exists(self.file_path):
            self.file_path = os.path.join(base_path, f"{name}_{counter}.png")
            counter += 1
        cv2.imwrite(self.file_path, frame)

def cam_kind_from_frame_type(dtype: dai.RawImgFrame.Type) -> str:
    """Returns camera kind string for given dai.RawImgFrame.Type."""
    return "mono_cam" if dtype == dai.RawImgFrame.Type.RAW8 else "color_cam"


def cam_kind_from_sensor_kind(kind: dai.CameraSensorType) -> str:
    """Returns camera kind string for given sensor type."""
    return "mono_cam" if kind == dai.CameraSensorType.MONO else "color_cam"
