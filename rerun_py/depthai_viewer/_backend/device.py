import itertools
import time
from queue import Queue, Empty as QueueEmpty
from typing import Dict, List, Optional, Tuple

import depthai as dai
import numpy as np
from depthai_sdk import OakCamera
from depthai_sdk.components import CameraComponent, NNComponent, StereoComponent
from depthai_sdk.components.tof_component import ToFComponent, Component
from depthai_sdk.components.camera_helper import (
    getClosestIspScale,
)
from depthai_sdk.classes.packet_handlers import QueuePacketHandler, ComponentOutput
from numpy.typing import NDArray

import depthai_viewer as viewer
from depthai_viewer._backend import classification_labels
from depthai_viewer._backend.device_configuration import (
    CameraConfiguration,
    CameraFeatures,
    CameraSensorResolution,
    DeviceInfo,
    DeviceProperties,
    ImuKind,
    PipelineConfiguration,
    XLinkConnection,
    calculate_isp_scale,
    compare_dai_camera_configs,
    get_size_from_resolution,
    size_to_resolution,
)
from depthai_viewer._backend.messages import (
    ErrorMessage,
    InfoMessage,
    Message,
    WarningMessage,
)
from depthai_viewer._backend.packet_handler import (
    AiModelCallbackArgs,
    DepthCallbackArgs,
    PacketHandler,
    SyncedCallbackArgs,
)
from depthai_viewer._backend.store import Store


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


# import cProfile
# import time


class Device:
    id: str
    intrinsic_matrix: Dict[Tuple[dai.CameraBoardSocket, int, int], NDArray[np.float32]] = {}
    calibration_data: Optional[dai.CalibrationHandler] = None
    use_encoding: bool = False
    store: Store

    _packet_handler: PacketHandler
    _oak: Optional[OakCamera] = None
    _cameras: List[CameraComponent] = []
    _stereo: StereoComponent = None
    _nnet: NNComponent = None
    _xlink_statistics: Optional[XlinkStatistics] = None
    _sys_info_q: Optional[Queue] = None  # type: ignore[type-arg]
    _pipeline_start_t: Optional[float] = None
    _queues: Dict[
        Component, QueuePacketHandler
    ] = {}

    # _profiler = cProfile.Profile()

    def __init__(self, device_id: str, store: Store):
        self.id = device_id
        self.set_oak(OakCamera(device_id))
        self.store = store
        self._packet_handler = PacketHandler(self.store, self.get_intrinsic_matrix)
        print("Oak cam: ", self._oak)
        # self.start = time.time()
        # self._profiler.enable()

    def set_oak(self, oak_cam: Optional[OakCamera]) -> None:
        self._oak = oak_cam
        self._xlink_statistics = None
        if self._oak is not None:
            self._xlink_statistics = XlinkStatistics(self._oak.device)

    def is_closed(self) -> bool:
        return self._oak is not None and self._oak.device.isClosed()

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
        if self._oak is None:
            return []
        calib_data = self._oak.device.readCalibration()
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
        if self._oak is None:
            raise Exception("No device selected!")
        connected_cam_features = self._oak.device.getConnectedCameraFeatures()
        imu = self._oak.device.getConnectedIMU()
        imu = ImuKind.NINE_AXIS if "BNO" in imu else None if imu == "NONE" else ImuKind.SIX_AXIS
        device_info = self._oak.device.getDeviceInfo()
        device_info = DeviceInfo(
            name=device_info.name,
            connection=XLinkConnection.POE
            if device_info.protocol == dai.XLinkProtocol.X_LINK_TCP_IP
            else XLinkConnection.USB,
            mxid=device_info.mxid,
        )
        device_properties = DeviceProperties(id=self.id, imu=imu, info=device_info)
        try:
            calib = self._oak.device.readCalibration2()
            left_cam = calib.getStereoLeftCameraId()
            right_cam = calib.getStereoRightCameraId()
            # A calibration can be present but the stereo pair sockets can be invalid
            # dai.CameraBoardSocket.???: 255
            if left_cam.value == 255 or right_cam.value == 255:
                device_properties.default_stereo_pair = None
            else:
                device_properties.default_stereo_pair = (left_cam, right_cam)
        except RuntimeError:
            pass

        ordered_resolutions = list(sorted(size_to_resolution.keys(), key=lambda res: res[0] * res[1]))
        for cam in connected_cam_features:
            if dai.CameraSensorType.TOF not in cam.supportedTypes:
                prioritized_type = cam.supportedTypes[0]
                biggest_width, biggest_height = [
                    (conf.width, conf.height) for conf in cam.configs[::-1] if conf.type == prioritized_type
                ][0]

                all_supported_resolutions = list(
                    filter(
                        lambda x: x,
                        [
                            size_to_resolution.get((w, h), None)
                            for w, h in ordered_resolutions
                            if (w * h) <= (biggest_height * biggest_width)
                        ],
                    )
                )
            else:
                all_supported_resolutions = []

            # Fill in lower resolutions that can be achieved with ISP scaling
            device_properties.cameras.append(
                CameraFeatures(
                    board_socket=cam.socket,
                    max_fps=60,
                    resolutions=all_supported_resolutions,
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
        print("Device properties: ", device_properties.default_stereo_pair)
        return device_properties

    def close_oak(self) -> None:
        if self._oak is None:
            return
        if self._oak.running():
            self._oak.device.__exit__(0, 0, 0)

    def reconnect_to_oak(self) -> Message:
        """

        Try to reconnect to the device with self.id.

        Timeout after 10 seconds.
        """
        if self._oak is None:
            return ErrorMessage("No device selected, can't reconnect!")
        if self._oak.device.isClosed():
            timeout_start = time.time()
            while time.time() - timeout_start < 10:
                available_devices = [
                    device.getMxId() for device in dai.Device.getAllAvailableDevices()  # type: ignore[call-arg]
                ]
                if self.id in available_devices:
                    break
            try:
                self.set_oak(OakCamera(self.id))
                return InfoMessage("Successfully reconnected to device")
            except RuntimeError as e:
                print("Failed to create oak camera")
                print(e)
                self.set_oak(None)
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

    def _create_auto_pipeline_config(self, config: PipelineConfiguration) -> Message:
        if self._oak is None:
            return ErrorMessage("Oak device unavailable, can't create auto pipeline config!")
        if self._oak.device is None:
            return ErrorMessage("No device selected, can't create auto pipeline config!")
        connected_cam_features = self._oak.device.getConnectedCameraFeatures()
        if not connected_cam_features:
            return ErrorMessage("No camera features found, can't create auto pipeline config!")
        n_cams = len(connected_cam_features)
        for cam in connected_cam_features:
            config.cameras.append(
                CameraConfiguration(
                )
            )


    def update_pipeline(self, runtime_only: bool) -> Message:
        if self._oak is None:
            return ErrorMessage("No device selected, can't update pipeline!")

        config = self.store.pipeline_config
        if config is None:
            return ErrorMessage("No pipeline config, can't update pipeline!")

        if self._oak.device.isPipelineRunning():
            if runtime_only:
                if config.depth is not None:
                    self._stereo.control.send_controls(config.depth.to_runtime_controls())
                    return InfoMessage("")
                return ErrorMessage("Depth is disabled, can't send runtime controls!")
            print("Cam running, closing...")
            self.close_oak()
            message = self.reconnect_to_oak()
            if isinstance(message, ErrorMessage):
                return message

        # if config.auto:
        #     self._create_auto_pipeline_config(config, self._oak.device)

        self._cameras = []
        self._stereo = None
        self._packet_handler.reset()
        self._sys_info_q = None
        self._pipeline_start_t = None

        synced_outputs: Dict[
            Component, ComponentOutput
        ] = {}
        synced_callback_args = SyncedCallbackArgs()

        is_poe = self._oak.device.getDeviceInfo().protocol == dai.XLinkProtocol.X_LINK_TCP_IP
        print("Usb speed: ", self._oak.device.getUsbSpeed())
        is_usb2 = self._oak.device.getUsbSpeed() == dai.UsbSpeed.HIGH
        if is_poe:
            self.store.send_message_to_frontend(
                WarningMessage("Device is connected via PoE. This may cause performance issues.")
            )
            print("Connected to a PoE device, camera streams will be JPEG encoded...")
        elif is_usb2:
            self.store.send_message_to_frontend(
                WarningMessage("Device is connected in USB2 mode. This may cause performance issues.")
            )
            print("Device is connected in USB2 mode, camera streams will be JPEG encoded...")
        self.use_encoding = is_poe or is_usb2

        connected_camera_features = self._oak.device.getConnectedCameraFeatures()
        for cam in config.cameras:
            print("Creating camera: ", cam)

            camera_features = next(filter(lambda feat: feat.socket == cam.board_socket, connected_camera_features))

            # When the resolution is too small, the ISP needs to scale it down
            res_x, res_y = get_size_from_resolution(cam.resolution)

            does_sensor_support_resolution = any(
                [
                    config.width == res_x and config.height == res_y
                    for config in camera_features.configs
                    if config.type == camera_features.supportedTypes[0]
                ]
            )

            # In case of ISP scaling, don't change the sensor resolution in the pipeline config
            # to keep it logical for the user in the UI
            sensor_resolution: Optional[CameraSensorResolution] = cam.resolution  # None for ToF
            if not does_sensor_support_resolution:
                smallest_supported_resolution = [
                    config for config in camera_features.configs if config.type == camera_features.supportedTypes[0]
                ][0]
                sensor_resolution = size_to_resolution.get(
                    (smallest_supported_resolution.width, smallest_supported_resolution.height), None
                )
            is_used_by_depth = config.depth is not None and (
                cam.board_socket == config.depth.align or cam.board_socket in config.depth.stereo_pair
            )
            is_used_by_ai = config.ai_model is not None and cam.board_socket == config.ai_model.camera
            cam.stream_enabled |= is_used_by_depth or is_used_by_ai

            # Only create a camera node if it is used by stereo or AI.
            if cam.stream_enabled:
                if dai.CameraSensorType.TOF in camera_features.supportedTypes:
                    sdk_cam = self._oak.create_tof(
                        cam.board_socket
                    )
                else:
                    sdk_cam = self._oak.create_camera(
                        cam.board_socket,
                        sensor_resolution.as_sdk_resolution(),
                        cam.fps,
                        encode=self.use_encoding,
                    )
                    if not does_sensor_support_resolution:
                        sdk_cam.config_color_camera(
                            isp_scale=getClosestIspScale(
                                (smallest_supported_resolution.width, smallest_supported_resolution.height), res_x
                            )
                        )
                    self._cameras.append(sdk_cam)
                synced_outputs[sdk_cam] = sdk_cam.out.main

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
            self._stereo = self._oak.create_stereo(left=left_cam, right=right_cam, name="depth")

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
            synced_callback_args.depth_args = DepthCallbackArgs(
                alignment_camera=aligned_camera, stereo_pair=config.depth.stereo_pair
            )
            synced_outputs[self._stereo] = self._stereo.out.main

        if self._oak.device.getConnectedIMU() != "NONE":
            print("Creating IMU")
            imu = self._oak.create_imu()
            sensors = [
                dai.IMUSensor.ACCELEROMETER_RAW,
                dai.IMUSensor.GYROSCOPE_RAW,
            ]
            if "BNO" in self._oak.device.getConnectedIMU():
                sensors.append(dai.IMUSensor.MAGNETOMETER_CALIBRATED)
            imu.config_imu(
                sensors, report_rate=config.imu.report_rate, batch_report_threshold=config.imu.batch_report_threshold
            )
            self._oak.callback(imu, self._packet_handler.on_imu)
        else:
            print("Connected cam doesn't have IMU, skipping IMU creation...")

        if config.ai_model and config.ai_model.path:
            cam_component = self._get_component_by_socket(config.ai_model.camera)
            if not cam_component:
                return ErrorMessage(f"{config.ai_model.camera} is not configured. Couldn't create NN.")
            labels: Optional[List[str]] = None
            if config.ai_model.path == "age-gender-recognition-retail-0013":
                face_detection = self._oak.create_nn("face-detection-retail-0004", cam_component)
                self._nnet = self._oak.create_nn("age-gender-recognition-retail-0013", input=face_detection)
            else:
                self._nnet = self._oak.create_nn(config.ai_model.path, cam_component)
                labels = getattr(classification_labels, config.ai_model.path.upper().replace("-", "_"), None)

            camera = self._get_camera_config_by_socket(config, config.ai_model.camera)
            if not camera:
                return ErrorMessage(f"{config.ai_model.camera} is not configured. Couldn't create NN.")

            synced_callback_args.ai_args = AiModelCallbackArgs(
                model_name=config.ai_model.path, camera=camera, labels=labels
            )
            synced_outputs[self._nnet] = self._nnet.out.main

        # Create the sdk queues and finalize the packet handler
        if synced_outputs:
            for component, synced_out in synced_outputs.items():
                self._queues[component] = self._oak.queue(synced_out)
        self._packet_handler.set_synced_callback_args(synced_callback_args)

        sys_logger_xlink = self._oak.pipeline.createXLinkOut()
        logger = self._oak.pipeline.createSystemLogger()
        logger.setRate(0.1)
        sys_logger_xlink.setStreamName("sys_logger")
        logger.out.link(sys_logger_xlink.input)

        try:
            self._oak.start(blocking=False)
        except RuntimeError as e:
            print("Couldn't start pipeline: ", e)
            return ErrorMessage("Couldn't start pipeline")

        running = self._oak.running()
        if running:
            self._pipeline_start_t = time.time()
            self._sys_info_q = self._oak.device.getOutputQueue("sys_logger", 1, False)
            self.store.set_pipeline_config(config)  # We might have modified the config, so store it
            try:
                self._oak.poll()
            except RuntimeError:
                return ErrorMessage("Runtime error when polling the device. Check the terminal for more info.")
            self.calibration_data = self._oak.device.readCalibration()
            self.intrinsic_matrix = {}
        return InfoMessage("Pipeline started") if running else ErrorMessage("Couldn't start pipeline")

    def update(self) -> None:
        if self._oak is None:
            return
        if not self._oak.running():
            return
        self._oak.poll()

        for component, queue in self._queues.items():
            try:
                packet = queue.get_queue().get_nowait()
                self._packet_handler.log_packet(component, packet)
            except QueueEmpty:
                continue

        if self._xlink_statistics is not None:
            self._xlink_statistics.update()

        if self._sys_info_q is None:
            return
        sys_info = self._sys_info_q.tryGet()  # type: ignore[attr-defined]
        if sys_info is not None and self._pipeline_start_t is not None:
            print("----------------------------------------")
            print(f"[{int(time.time() - self._pipeline_start_t)}s] System information")
            print("----------------------------------------")
            print_system_information(sys_info)
        # if time.time() - self.start > 10:
        #     print("Dumping profiling data")
        #     self._profiler.dump_stats("profile.prof")
        #     self._profiler.disable()
        #     self._profiler.enable()
        #     self.start = time.time()


def print_system_information(info: dai.SystemInformation) -> None:
    print(
        "Ddr used / total - %.2f / %.2f MiB"
        % (
            info.ddrMemoryUsage.used / (1024.0 * 1024.0),
            info.ddrMemoryUsage.total / (1024.0 * 1024.0),
        )
    )
    print(
        "Cmx used / total - %.2f / %.2f MiB"
        % (
            info.cmxMemoryUsage.used / (1024.0 * 1024.0),
            info.cmxMemoryUsage.total / (1024.0 * 1024.0),
        )
    )
    print(
        "LeonCss heap used / total - %.2f / %.2f MiB"
        % (
            info.leonCssMemoryUsage.used / (1024.0 * 1024.0),
            info.leonCssMemoryUsage.total / (1024.0 * 1024.0),
        )
    )
    print(
        "LeonMss heap used / total - %.2f / %.2f MiB"
        % (
            info.leonMssMemoryUsage.used / (1024.0 * 1024.0),
            info.leonMssMemoryUsage.total / (1024.0 * 1024.0),
        )
    )
    t = info.chipTemperature
    print(
        "Chip temperature - average: %.2f, css: %.2f, mss: %.2f, upa: %.2f, dss: %.2f"
        % (
            t.average,
            t.css,
            t.mss,
            t.upa,
            t.dss,
        )
    )
    print(
        "Cpu usage - Leon CSS: %.2f %%, Leon MSS: %.2f %%"
        % (
            info.leonCssCpuUsage.average * 100,
            info.leonMssCpuUsage.average * 100,
        )
    )
