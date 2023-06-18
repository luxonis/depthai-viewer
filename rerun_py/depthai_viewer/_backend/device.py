import itertools
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import depthai_viewer as viewer
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import *
from depthai_sdk.components import CameraComponent, NNComponent, StereoComponent
from depthai_viewer._backend import classification_labels
from depthai_viewer._backend.device_configuration import (
    CameraConfiguration,
    CameraFeatures,
    ImuKind,
    calculate_isp_scale,
    compare_dai_camera_configs,
    resolution_to_enum,
)
from depthai_viewer._backend.messages import *
from depthai_viewer._backend.packet_handler import (
    PacketHandler,
    CameraCallbackArgs,
    DepthCallbackArgs,
    AiModelCallbackArgs,
)
from depthai_viewer._backend.store import Store
from numpy.typing import NDArray


class XlinkStatistics:
    _device: dai.Device
    _time_of_last_update: float = 0  # s since epoch

    def __init__(self, device: dai.Device):
        self._device = device

    def update(self) -> None:
        if time.time() - self._time_of_last_update >= 32e-3:
            self._time_of_last_update = time.time()
            if hasattr(self._device, "getProfilingData"):  # Only on latest develop
                try:
                    xlink_stats = self._device.getProfilingData()
                    viewer.log_xlink_stats(
                        xlink_stats.numBytesWritten, xlink_stats.numBytesRead, self._time_of_last_update
                    )
                except Exception:
                    pass


class Device:
    id: str
    intrinsic_matrix: Dict[Tuple[dai.CameraBoardSocket, int, int], NDArray[np.float32]] = {}
    calibration_data: Optional[dai.CalibrationHandler] = None
    use_encoding: bool = False
    store: Store

    _packet_handler: PacketHandler
    _oak_cam: Optional[OakCamera] = None
    _cameras: List[CameraComponent] = []
    _stereo: StereoComponent = None
    _nnet: NNComponent = None
    _xlink_statistics: Optional[XlinkStatistics] = None

    def __init__(self, device_id: str, store: Store):
        self.id = device_id
        self.set_oak_cam(OakCamera(device_id))
        self.store = store
        self._packet_handler = PacketHandler(self.store, self.get_intrinsic_matrix)
        print("Oak cam: ", self._oak_cam)

    def set_oak_cam(self, oak_cam: Optional[OakCamera]) -> None:
        self._oak_cam = oak_cam
        self._xlink_statistics = None
        if self._oak_cam is not None:
            self._xlink_statistics = XlinkStatistics(self._oak_cam.device)

    def is_closed(self) -> bool:
        return self._oak_cam is not None and self._oak_cam.device.isClosed()

    def get_intrinsic_matrix(self, board_socket: dai.CameraBoardSocket, width: int, height: int) -> NDArray[np.float32]:
        if self.intrinsic_matrix.get((board_socket, width, height)) is not None:
            return self.intrinsic_matrix.get((board_socket, width, height))  # type: ignore[return-value]
        if self.calibration_data is None:
            raise Exception("Missing calibration data!")
        M_right = self.calibration_data.getCameraIntrinsics(  # type: ignore[union-attr]
            board_socket, dai.Size2f(width, height)
        )
        self.intrinsic_matrix[(board_socket, width, height)] = np.array(M_right).reshape(3, 3)
        return self.intrinsic_matrix[(board_socket, width, height)]

    def _get_possible_stereo_pairs_for_cam(
        self, cam: dai.CameraFeatures, connected_camera_features: List[dai.CameraFeatures]
    ) -> List[dai.CameraBoardSocket]:
        """Tries to find the possible stereo pairs for a camera."""
        if self._oak_cam is None:
            return []
        calib_data = self._oak_cam.device.readCalibration()
        try:
            calib_data.getCameraIntrinsics(cam.socket)
        except IndexError:
            return []
        possible_stereo_pairs = []
        if cam.name == "right":
            possible_stereo_pairs.extend(
                [features.socket for features in filter(lambda c: c.name == "left", connected_camera_features)]
            )
        elif cam.name == "left":
            possible_stereo_pairs.extend(
                [features.socket for features in filter(lambda c: c.name == "right", connected_camera_features)]
            )
        else:
            possible_stereo_pairs.extend(
                [
                    camera.socket
                    for camera in connected_camera_features
                    if camera != cam
                    and all(
                        map(
                            lambda confs: compare_dai_camera_configs(confs[0], confs[1]),
                            zip(camera.configs, cam.configs),
                        )
                    )
                ]
            )
        stereo_pairs = []
        for pair in possible_stereo_pairs:
            try:
                calib_data.getCameraIntrinsics(pair)
            except IndexError:
                continue
            stereo_pairs.append(pair)
        return stereo_pairs

    def get_device_properties(self) -> DeviceProperties:
        if self._oak_cam is None:
            raise Exception("No device selected!")
        connected_cam_features = self._oak_cam.device.getConnectedCameraFeatures()
        imu = self._oak_cam.device.getConnectedIMU()
        imu = ImuKind.NINE_AXIS if "BNO" in imu else None if imu == "NONE" else ImuKind.SIX_AXIS
        device_properties = DeviceProperties(id=self.id, imu=imu)
        try:
            calib = self._oak_cam.device.readCalibration2()
            left_cam = calib.getStereoLeftCameraId()
            right_cam = calib.getStereoRightCameraId()
            device_properties.default_stereo_pair = (left_cam, right_cam)
        except RuntimeError:
            pass
        for cam in connected_cam_features:
            prioritized_type = cam.supportedTypes[0]
            device_properties.cameras.append(
                CameraFeatures(
                    board_socket=cam.socket,
                    max_fps=60,
                    resolutions=[
                        resolution_to_enum[(conf.width, conf.height)]
                        for conf in cam.configs
                        if conf.type == prioritized_type  # Only support the prioritized type for now
                    ],
                    supported_types=cam.supportedTypes,
                    stereo_pairs=self._get_possible_stereo_pairs_for_cam(cam, connected_cam_features),
                    name=cam.name.capitalize(),
                )
            )
        device_properties.stereo_pairs = list(
            itertools.chain.from_iterable(
                [(cam.board_socket, pair) for pair in cam.stereo_pairs] for cam in device_properties.cameras
            )
        )
        return device_properties

    def close_oak_cam(self) -> None:
        if self._oak_cam is None:
            return
        if self._oak_cam.running():
            self._oak_cam.device.__exit__(0, 0, 0)

    def reconnect_to_oak_cam(self) -> Message:
        """

        Try to reconnect to the device with self.id.

        Timeout after 10 seconds.
        """
        if self._oak_cam is None:
            return ErrorMessage("No device selected, can't reconnect!")
        if self._oak_cam.device.isClosed():
            timeout_start = time.time()
            while time.time() - timeout_start < 10:
                available_devices = [
                    device.getMxId() for device in dai.Device.getAllAvailableDevices()  # type: ignore[call-arg]
                ]
                if self.id in available_devices:
                    break
            try:
                self.set_oak_cam(OakCamera(self.id))
                return InfoMessage("Successfully reconnected to device")
            except RuntimeError as e:
                print("Failed to create oak camera")
                print(e)
                self.set_oak_cam(None)
        return ErrorMessage("Failed to create oak camera")

    def _get_component_by_socket(self, socket: dai.CameraBoardSocket) -> Optional[CameraComponent]:
        component = list(filter(lambda c: c.node.getBoardSocket() == socket, self._cameras))
        if not component:
            return None
        return component[0]

    def _get_camera_config_by_socket(
        self, config: PipelineConfiguration, socket: dai.CameraBoardSocket
    ) -> Optional[CameraConfiguration]:
        print("Getting cam by socket: ", socket, " Cameras: ", config.cameras)
        camera = list(filter(lambda c: c.board_socket == socket, config.cameras))
        if not camera:
            return None
        return camera[0]

    def update_pipeline(self, config: PipelineConfiguration, runtime_only: bool) -> Message:
        if self._oak_cam is None:
            return ErrorMessage("No device selected, can't update pipeline!")
        if self._oak_cam.device.isPipelineRunning():
            if runtime_only:
                if config.depth is not None:
                    self._stereo.control.send_controls(config.depth.to_runtime_controls())
                    return InfoMessage("")
                return ErrorMessage("Depth is disabled, can't send runtime controls!")
            print("Cam running, closing...")
            self.close_oak_cam()
            message = self.reconnect_to_oak_cam()
            if isinstance(message, ErrorMessage):
                return message

        self._cameras = []
        # self._packet_handler.clear_queues()
        self.use_encoding = self._oak_cam.device.getDeviceInfo().protocol == dai.XLinkProtocol.X_LINK_TCP_IP
        if self.use_encoding:
            print("Connected device is PoE: Using encoding...")
        else:
            print("Connected device is USB: Not using encoding...")
        for cam in config.cameras:
            print("Creating camera: ", cam)
            sdk_cam = self._oak_cam.create_camera(
                cam.board_socket,
                cam.resolution.as_sdk_resolution(),
                cam.fps,
                encode=self.use_encoding,
                name=cam.name.capitalize(),
            )
            if cam.stream_enabled:
                callback_args = CameraCallbackArgs(
                    board_socket=cam.board_socket, image_kind=cam.kind, encoding=self.use_encoding
                )
                self._oak_cam.callback(
                    sdk_cam,
                    self._packet_handler.build_callback(callback_args),
                    main_thread=True,
                )
            self._cameras.append(sdk_cam)

        if config.depth:
            print("Creating depth")
            stereo_pair = config.depth.stereo_pair
            left_cam = self._get_component_by_socket(stereo_pair[0])
            right_cam = self._get_component_by_socket(stereo_pair[1])
            if not left_cam or not right_cam:
                return ErrorMessage(f"{cam} is not configured. Couldn't create stereo pair.")

            if left_cam.node.getResolutionWidth() > 1280:
                print("Left cam width > 1280, setting isp scale to get 800")
                left_cam.config_color_camera(isp_scale=calculate_isp_scale(left_cam.node.getResolutionWidth()))
            if right_cam.node.getResolutionWidth() > 1280:
                print("Right cam width > 1280, setting isp scale to get 800")
                right_cam.config_color_camera(isp_scale=calculate_isp_scale(right_cam.node.getResolutionWidth()))
            self._stereo = self._oak_cam.create_stereo(left=left_cam, right=right_cam, name="depth")

            # We used to be able to pass in the board socket to align to, but this was removed in depthai 1.10.0
            align_component = self._get_component_by_socket(config.depth.align)
            if not align_component:
                return ErrorMessage(f"{config.depth.align} is not configured. Couldn't create stereo pair.")
            self._stereo.config_stereo(
                lr_check=config.depth.lr_check,
                subpixel=config.depth.subpixel_disparity,
                confidence=config.depth.confidence,
                align=align_component,
                lr_check_threshold=config.depth.lrc_threshold,
                median=config.depth.median,
            )

            aligned_camera = self._get_camera_config_by_socket(config, config.depth.align)
            if not aligned_camera:
                return ErrorMessage(f"{config.depth.align} is not configured. Couldn't create stereo pair.")

            self._oak_cam.callback(
                self._stereo,
                self._packet_handler.build_callback(
                    DepthCallbackArgs(alignment_camera=aligned_camera, stereo_pair=config.depth.stereo_pair)
                ),
                main_thread=True,
            )

        if self._oak_cam.device.getConnectedIMU() != "NONE":
            print("Creating IMU")
            imu = self._oak_cam.create_imu()
            sensors = [
                dai.IMUSensor.ACCELEROMETER_RAW,
                dai.IMUSensor.GYROSCOPE_RAW,
            ]
            if "BNO" in self._oak_cam.device.getConnectedIMU():
                sensors.append(dai.IMUSensor.MAGNETOMETER_CALIBRATED)
            imu.config_imu(
                sensors, report_rate=config.imu.report_rate, batch_report_threshold=config.imu.batch_report_threshold
            )
            # self._oak_cam.callback(imu, self._packet_handler.on_imu, main_thread=True)
        else:
            print("Connected cam doesn't have IMU, skipping IMU creation...")

        if config.ai_model and config.ai_model.path:
            cam_component = self._get_component_by_socket(config.ai_model.camera)
            if not cam_component:
                return ErrorMessage(f"{config.ai_model.camera} is not configured. Couldn't create NN.")
            labels: Optional[List[str]] = None
            if config.ai_model.path == "age-gender-recognition-retail-0013":
                face_detection = self._oak_cam.create_nn("face-detection-retail-0004", cam_component)
                self._nnet = self._oak_cam.create_nn("age-gender-recognition-retail-0013", input=face_detection)
            else:
                self._nnet = self._oak_cam.create_nn(config.ai_model.path, cam_component)
                labels = getattr(classification_labels, config.ai_model.path.upper().replace("-", "_"), None)

            camera = self._get_camera_config_by_socket(config, config.ai_model.camera)
            if not camera:
                return ErrorMessage(f"{config.ai_model.camera} is not configured. Couldn't create NN.")

            self._oak_cam.callback(
                self._nnet,
                self._packet_handler.build_callback(
                    AiModelCallbackArgs(model_name=config.ai_model.path, camera=camera, labels=labels)
                ),
                main_thread=True,
            )
        try:
            self._oak_cam.start(blocking=False)
        except RuntimeError as e:
            print("Couldn't start pipeline: ", e)
            return ErrorMessage("Couldn't start pipeline")

        running = self._oak_cam.running()
        if running:
            self._oak_cam.poll()
            self.calibration_data = self._oak_cam.device.readCalibration()
            self.intrinsic_matrix = {}

        # print("Queues: ", self._oak_cam.device.getOutputQueueNames())
        # color_q = self._oak_cam.device.getOutputQueue("Color_video", maxSize=1, blocking=False)
        # color_ctx = QueueContext(list(filter(lambda c: c.name.capitalize() == "Color", config.cameras))[0])
        # cam_cpy = color_ctx.cam.copy()
        # color_ctx.cam = cam_cpy
        # color_ctx.cam.name = "Color_edited"
        # self._packet_handler.add_queue(color_ctx, color_q)
        return InfoMessage("Pipeline started" if running else ErrorMessage("Couldn't start pipeline"))

    def update(self) -> None:
        if self._oak_cam is None:
            return
        if not self._oak_cam.running():
            return
        self._oak_cam.poll()
        if self._xlink_statistics is not None:
            self._xlink_statistics.update()
        # self._packet_handler.update()
