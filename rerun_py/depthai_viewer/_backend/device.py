import itertools
import os
import time
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Dict, List, Optional, Tuple

import depthai as dai
from depthai_sdk import OakCamera
from depthai_sdk.classes.packet_handlers import ComponentOutput
from depthai_sdk.components import CameraComponent, NNComponent, StereoComponent
from depthai_sdk.components.camera_helper import getClosestIspScale
from depthai_sdk.components.tof_component import Component, ToFComponent

import depthai_viewer as viewer
from depthai_viewer._backend.device_configuration import (
    ALL_NEURAL_NETWORKS,
    CameraConfiguration,
    CameraFeatures,
    CameraSensorResolution,
    DeviceInfo,
    DeviceProperties,
    ImuKind,
    PipelineConfiguration,
    StereoDepthConfiguration,
    ToFConfig,
    XLinkConnection,
    calculate_isp_scale,
    compare_dai_camera_configs,
    get_size_from_resolution,
    size_to_resolution,
)
from depthai_viewer._backend.device_defaults import (
    oak_d_generic_default,
    oak_d_lr_default,
    oak_d_sr_default,
    oak_d_sr_poe_default,
    oak_t_default,
)
from depthai_viewer._backend.messages import (
    ErrorMessage,
    InfoMessage,
    Message,
    WarningMessage,
)
from depthai_viewer._backend.packet_handler import (
    DetectionContext,
    PacketHandler,
    PacketHandlerContext,
)
from depthai_viewer._backend.store import Store
from depthai_viewer.install_requirements import model_dir


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
    calibration_handler: Optional[dai.CalibrationHandler] = None
    use_encoding: bool = False
    store: Store

    _packet_handler: PacketHandler
    _oak: Optional[OakCamera] = None
    _stereo: StereoComponent = None
    _nnet: NNComponent = None
    _xlink_statistics: Optional[XlinkStatistics] = None
    _sys_info_q: Optional[Queue] = None  # type: ignore[type-arg]
    _pipeline_start_t: Optional[float] = None
    _queues: List[Tuple[Component, ComponentOutput]] = []
    _dai_queues: List[Tuple[dai.Node, dai.DataOutputQueue, Optional[PacketHandlerContext]]] = []
    _tof_component: Optional[ToFComponent] = None

    # _profiler = cProfile.Profile()

    def __init__(self, device_id: str, store: Store):
        self.id = device_id
        self.set_oak(OakCamera(device_id, args={"irFloodBrightness": 0, "irDotBrightness": 0}))
        self.store = store
        self._packet_handler = PacketHandler(self.store, self._oak.device.readCalibration())  # type: ignore[union-attr]
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

    def _get_possible_stereo_pairs_for_cam(
        self, cam: dai.CameraFeatures, connected_camera_features: List[dai.CameraFeatures]
    ) -> List[dai.CameraBoardSocket]:
        """Tries to find the possible stereo pairs for a camera."""
        if self._oak is None:
            return []
        try:
            calib_data = self._oak.device.readCalibration2()
        except RuntimeError:
            print("No calibration available.")
            return []
        try:
            calib_data.getCameraIntrinsics(cam.socket)
        except IndexError:
            print("No intrisics for cam: ", cam.socket)
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
            connection=(
                XLinkConnection.POE if device_info.protocol == dai.XLinkProtocol.X_LINK_TCP_IP else XLinkConnection.USB
            ),
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
            print("No calibration found while trying to fetch the default stereo pair.")

        ordered_resolutions = list(sorted(size_to_resolution.keys(), key=lambda res: res[0] * res[1]))
        for cam in connected_cam_features:
            prioritized_type = cam.supportedTypes[0]
            biggest_width_height = [
                (conf.width, conf.height) for conf in cam.configs[::-1] if conf.type == prioritized_type
            ]
            # Some sensors don't have configs, use the sensor width, height
            if not biggest_width_height:
                biggest_width, biggest_height = cam.width, cam.height
            else:
                biggest_width, biggest_height = biggest_width_height[0]

            if cam.supportedTypes[0] == dai.CameraSensorType.TOF:
                all_supported_resolutions = [size_to_resolution[(biggest_width, biggest_height)]]
            else:
                all_supported_resolutions = list(
                    filter(
                        lambda x: x,  # type: ignore[arg-type]
                        [
                            size_to_resolution.get((w, h), None)
                            for w, h in ordered_resolutions
                            if (w, h)
                            in [(conf.width, conf.height) for conf in cam.configs if conf.type == prioritized_type]
                        ],
                    )
                )

            # Fill in lower resolutions that can be achieved with ISP scaling
            device_properties.cameras.append(
                CameraFeatures(
                    board_socket=cam.socket,
                    max_fps=60,
                    resolutions=all_supported_resolutions,
                    supported_types=cam.supportedTypes,
                    stereo_pairs=self._get_possible_stereo_pairs_for_cam(cam, connected_cam_features),
                    name=cam.name.capitalize(),
                    tof_config=ToFConfig() if dai.CameraSensorType.TOF in cam.supportedTypes else None,
                )
            )
        device_properties.stereo_pairs = list(
            itertools.chain.from_iterable(
                [(cam.board_socket, pair) for pair in cam.stereo_pairs] for cam in device_properties.cameras
            )
        )
        print("Default stereo pair: ", device_properties.default_stereo_pair)
        return device_properties

    def close_oak(self) -> None:
        if self._oak is None:
            return
        if self._oak.device:
            self._oak.device.close()

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
                    # type: ignore[call-arg]
                    device.getMxId()
                    for device in dai.Device.getAllAvailableDevices()
                ]
                if self.id in available_devices:
                    break
            try:
                self.set_oak(OakCamera(self.id, args={"irFloodBrightness": 0, "irDotBrightness": 0}))
                return InfoMessage("Successfully reconnected to device")
            except RuntimeError as e:
                print("Failed to create oak camera")
                print(e)
                self.set_oak(None)
        return ErrorMessage("Failed to create oak camera")

    def _get_component_by_socket(self, socket: dai.CameraBoardSocket) -> Optional[CameraComponent]:
        component = list(
            filter(  # type: ignore[arg-type]
                lambda c: isinstance(c, CameraComponent) and c.node.getBoardSocket() == socket,
                self._oak._components if self._oak else [],
            )
        )
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
        print("Creating auto pipeline config")
        if self._oak is None:
            return ErrorMessage("Oak device unavailable, can't create auto pipeline config!")
        if self._oak.device is None:
            return ErrorMessage("No device selected, can't create auto pipeline config!")
        connected_cam_features = self._oak.device.getConnectedCameraFeatures()
        if not connected_cam_features:
            return ErrorMessage("No camera features found, can't create auto pipeline config!")

        print("Connected camera features: ", connected_cam_features)
        # Step 1: Create all the cameras. Try to find RGB cam, to align depth to it later
        # Step 2: Create stereo depth if calibration is present. Align to RGB if present, otherwise to left cam
        # Step 3: Create YOLO
        rgb_cam_socket = None
        # 1. Create all the cameras
        config.cameras = []
        has_tof = False
        for cam in connected_cam_features:
            if cam.name == "rgb":  # By convention
                rgb_cam_socket = cam.socket
            resolution = (
                CameraSensorResolution.THE_1080_P if cam.width >= 1920 else size_to_resolution[(cam.width, cam.height)]
            )
            resolution = CameraSensorResolution.THE_1200_P if cam.height == 1200 else resolution
            preferred_type = cam.supportedTypes[0]
            if preferred_type == dai.CameraSensorType.TOF:
                has_tof = True
            config.cameras.append(
                CameraConfiguration(resolution=resolution, kind=preferred_type, board_socket=cam.socket, name=cam.name)
            )
        # 2. Create stereo depth
        if not has_tof:
            try:
                calibration = self._oak.device.readCalibration2()
                left_cam = calibration.getStereoLeftCameraId()
                right_cam = calibration.getStereoRightCameraId()
                if left_cam.value != 255 and right_cam.value != 255:
                    config.stereo = StereoDepthConfiguration(
                        stereo_pair=(left_cam, right_cam),
                        align=rgb_cam_socket if rgb_cam_socket is not None else left_cam,
                    )
            except RuntimeError:
                calibration = None
        else:
            config.stereo = None
        # 3. Create YOLO
        nnet_cam_sock = rgb_cam_socket
        if nnet_cam_sock is None:
            # Try to find a color camera config
            nnet_cam_sock = next(
                filter(
                    lambda cam: cam.kind == dai.CameraSensorType.COLOR,  # type: ignore[arg-type,union-attr]
                    config.cameras,
                ),
                None,
            )  # type: ignore[assignment]
            if nnet_cam_sock is not None:
                nnet_cam_sock = nnet_cam_sock.board_socket
        if nnet_cam_sock is not None:
            config.ai_model = ALL_NEURAL_NETWORKS[1]  # Yolo V6
            config.ai_model.camera = nnet_cam_sock
        else:
            config.ai_model = ALL_NEURAL_NETWORKS[1]
        return InfoMessage("Created auto pipeline config")

    def update_pipeline(self, runtime_only: bool) -> Message:
        if self._oak is None:
            return ErrorMessage("No device selected, can't update pipeline!")

        config = self.store.pipeline_config
        if config is None:
            return ErrorMessage("No pipeline config, can't update pipeline!")

        if self._oak.device.isPipelineRunning():
            if runtime_only:
                if config.stereo is not None:
                    self._stereo.control.send_controls(config.stereo.to_runtime_controls())
                    return InfoMessage("")
                return ErrorMessage("Depth is disabled, can't send runtime controls!")
            print("Cam running, closing...")
            self.close_oak()
            message = self.reconnect_to_oak()
            if isinstance(message, ErrorMessage):
                return message

        self._queues = []
        self._dai_queues = []

        if config.auto:
            if self._oak.device.getDeviceName() == "OAK-T":
                config = oak_t_default.config
            elif self._oak.device.getDeviceName() == "OAK-D-SR-POE":
                config = oak_d_sr_poe_default.config
            elif self._oak.device.getDeviceName() == "OAK-D-SR":
                config = oak_d_sr_default.config
            elif self._oak.device.getDeviceName() == "OAK-D-LR":
                config = oak_d_lr_default.config
            elif "OAK-D" in self._oak.device.getDeviceName():
                config = oak_d_generic_default.config
            else:
                self._create_auto_pipeline_config(config)

        create_dai_queues_after_start: Dict[str, Tuple[dai.Node, Optional[PacketHandlerContext]]] = {}
        self._stereo = None
        self._packet_handler.reset()
        self._sys_info_q = None
        self._pipeline_start_t = None

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
        create_tof_on = None
        for cam in config.cameras:
            print("Creating camera: ", cam)

            camera_features = next(filter(lambda feat: feat.socket == cam.board_socket, connected_camera_features))

            # When the resolution is too small, the ISP needs to scale it down
            res_x, res_y = get_size_from_resolution(cam.resolution)

            does_sensor_support_resolution = (
                any(
                    [
                        config.width == res_x and config.height == res_y
                        for config in camera_features.configs
                        if config.type == camera_features.supportedTypes[0]
                    ]
                )
                or len(camera_features.configs) == 0
            )  # Some sensors don't have any configs... just assume the resolution is supported

            # In case of ISP scaling, don't change the sensor resolution in the pipeline config
            # to keep it logical for the user in the UI
            # None for ToF
            sensor_resolution: Optional[CameraSensorResolution] = cam.resolution
            if not does_sensor_support_resolution:
                smallest_supported_resolution = [
                    config for config in camera_features.configs if config.type == camera_features.supportedTypes[0]
                ][0]
                sensor_resolution = size_to_resolution.get(
                    (smallest_supported_resolution.width, smallest_supported_resolution.height), None
                )
            is_used_by_depth = config.stereo is not None and (
                cam.board_socket == config.stereo.align or cam.board_socket in config.stereo.stereo_pair
            )
            is_used_by_ai = config.ai_model is not None and cam.board_socket == config.ai_model.camera
            cam.stream_enabled |= is_used_by_depth or is_used_by_ai

            # Only create a camera node if it is used by stereo or AI.
            if cam.stream_enabled:
                if dai.CameraSensorType.TOF in camera_features.supportedTypes:
                    create_tof_on = cam.board_socket
                elif dai.CameraSensorType.THERMAL in camera_features.supportedTypes:
                    thermal_cam = self._oak.pipeline.create(dai.node.Camera)
                    # Hardcoded for OAK-T. The correct size is needed for correct detection parsing
                    thermal_cam.setSize(256, 192)
                    thermal_cam.setBoardSocket(cam.board_socket)
                    xout_thermal = self._oak.pipeline.create(dai.node.XLinkOut)
                    xout_thermal.setStreamName("thermal_cam")
                    thermal_cam.raw.link(xout_thermal.input)
                    create_dai_queues_after_start["thermal_cam"] = (thermal_cam, None)
                elif sensor_resolution is not None:
                    sdk_cam = self._oak.create_camera(
                        cam.board_socket,
                        # type: ignore[union-attr]
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
                    self._queues.append((sdk_cam, self._oak.queue(sdk_cam.out.main)))
                else:
                    print("Skipped creating camera:", cam.board_socket, "because no valid sensor resolution was found.")
                    continue

        if create_tof_on is not None:
            cam_cfg = list(filter(lambda c: c.board_socket == create_tof_on, config.cameras))
            if len(cam_cfg) != 0:
                cam_cfg = cam_cfg[0]
                cam_cfg.tof_align = dai.CameraBoardSocket.CAM_B  # OAK-D-SR-PoE default.
                tof_align = list(
                    filter(
                        lambda comp_and_q: (  # type: ignore[arg-type]
                            comp_and_q[0].node.getBoardSocket() == cam_cfg.tof_align
                            if isinstance(comp_and_q[0], CameraComponent)
                            else False
                        ),
                        self._queues,
                    )
                )
                tof_align = tof_align[0][0] if tof_align else None
                sdk_cam = self._oak.create_tof(create_tof_on, tof_align)
                self._tof_component = sdk_cam
                self._queues.append((sdk_cam, self._oak.queue(sdk_cam.out.main)))
            else:  # Should never happen
                print("Couldn't find camera config for ToF, can't create ToF.")

        # Make sure that it's possible to create stereo depth
        try:
            if config.stereo:
                stereo_pair = config.stereo.stereo_pair
                self._oak.device.readCalibration2().getCameraExtrinsics(stereo_pair[0], stereo_pair[1])
        except RuntimeError:
            config.stereo = None
        if config.stereo:
            print("Creating depth")
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

            align_component = self._get_component_by_socket(config.stereo.align)
            if not align_component:
                return ErrorMessage(f"{config.stereo.align} is not configured. Couldn't create stereo pair.")
            self._stereo.config_stereo(
                lr_check=config.stereo.lr_check,
                subpixel=config.stereo.subpixel_disparity,
                confidence=config.stereo.confidence,
                align=align_component,
                lr_check_threshold=config.stereo.lrc_threshold,
                median=config.stereo.median,
            )

            aligned_camera = self._get_camera_config_by_socket(config, config.stereo.align)
            if not aligned_camera:
                return ErrorMessage(f"{config.stereo.align} is not configured. Couldn't create stereo pair.")
            aligned_camera.is_used_as_stereo_align = True
            self._queues.append((self._stereo, self._oak.queue(self._stereo.out.main)))

        if self._oak.device.getConnectedIMU() != "NONE" and self._oak.device.getConnectedIMU() != "":
            print("Creating IMU, connected IMU: ", self._oak.device.getConnectedIMU())
            # TODO(someone): Handle IMU updates
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
            dai_camnode = [
                node
                for node, _ in create_dai_queues_after_start.values()
                if isinstance(node, dai.node.Camera) and node.getBoardSocket() == config.ai_model.camera
            ]
            model_path = config.ai_model.path
            if len(dai_camnode) > 0:
                model_path = os.path.join(
                    model_dir,
                    config.ai_model.path
                    + "_openvino_"
                    + dai.OpenVINO.getVersionName(dai.OpenVINO.DEFAULT_VERSION)
                    + "_6shave"
                    + ".blob",
                )
                cam_node = dai_camnode[0]
                if "yolo" in config.ai_model.path:
                    yolo = self._oak.pipeline.createYoloDetectionNetwork()
                    yolo.setBlobPath(model_path)
                    if "yolov6n_thermal_people_256x192" == config.ai_model.path:
                        yolo.setConfidenceThreshold(0.5)
                        yolo.setNumClasses(1)
                        yolo.setCoordinateSize(4)
                        yolo.setIouThreshold(0.5)
                cam_node.raw.link(yolo.input)
                xlink_out_yolo = self._oak.pipeline.createXLinkOut()
                xlink_out_yolo.setStreamName("yolo")
                yolo.out.link(xlink_out_yolo.input)
                create_dai_queues_after_start["yolo"] = (
                    yolo,
                    DetectionContext(
                        labels=["person"],
                        frame_width=cam_node.getWidth(),
                        frame_height=cam_node.getHeight(),
                        board_socket=config.ai_model.camera,
                    ),
                )
            elif not cam_component:
                self.store.send_message_to_frontend(
                    WarningMessage(f"{config.ai_model.camera} is not configured, won't create NNET.")
                )
            elif config.ai_model.path == "age-gender-recognition-retail-0013":
                face_detection = self._oak.create_nn("face-detection-retail-0004", cam_component)
                self._nnet = self._oak.create_nn(model_path, input=face_detection)
            else:
                self._nnet = self._oak.create_nn(model_path, cam_component)
            if self._nnet:
                self._queues.append((self._nnet, self._oak.queue(self._nnet.out.main)))

        sys_logger_xlink = self._oak.pipeline.createXLinkOut()
        logger = self._oak.pipeline.createSystemLogger()
        logger.setRate(0.1)
        sys_logger_xlink.setStreamName("sys_logger")
        logger.out.link(sys_logger_xlink.input)

        try:
            print("Starting pipeline")
            self._oak.start()
        except RuntimeError as e:
            print("Couldn't start pipeline: ", e)
            return ErrorMessage("Couldn't start pipeline")

        running = self._oak.running()
        if running:
            for q_name, (node, context) in create_dai_queues_after_start.items():
                self._dai_queues.append((node, self._oak.device.getOutputQueue(q_name, 2, False), context))
            self._pipeline_start_t = time.time()
            self._sys_info_q = self._oak.device.getOutputQueue("sys_logger", 1, False)
            # We might have modified the config, so store it
            self.store.set_pipeline_config(config)
            try:
                self._oak.poll()
            except RuntimeError:
                return ErrorMessage("Runtime error when polling the device. Check the terminal for more info.")
            self.calibration_handler = self._oak.device.readCalibration()
        return InfoMessage("Pipeline started") if running else ErrorMessage("Couldn't start pipeline")

    def update(self) -> None:
        if self._oak is None:
            return
        if not self._oak.running():
            return
        for component, queue in self._queues:
            try:
                packet = queue.get_queue().get_nowait()
                self._packet_handler.log_packet(component, packet)
            except QueueEmpty:
                continue

        for dai_node, queue, context in self._dai_queues:
            packet = queue.tryGet()
            if packet is not None:
                self._packet_handler.log_dai_packet(dai_node, packet, context)

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

    def get_tof_component(self) -> Optional[ToFComponent]:
        return self._tof_component


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
