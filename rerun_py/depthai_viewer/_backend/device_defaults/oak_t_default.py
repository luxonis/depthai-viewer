import depthai as dai
from depthai_viewer._backend.device_configuration import (
    AiModelConfiguration,
    CameraConfiguration,
    CameraSensorResolution,
    PipelineConfiguration,
)

config = PipelineConfiguration(
    cameras=[
        CameraConfiguration(
            resolution=CameraSensorResolution.THE_256X192,
            kind=dai.CameraSensorType.THERMAL,
            board_socket=dai.CameraBoardSocket.CAM_E,
            name="Thermal",
        ),
        CameraConfiguration(
            resolution=CameraSensorResolution.THE_1080_P,
            kind=dai.CameraSensorType.COLOR,
            board_socket=dai.CameraBoardSocket.CAM_A,
            stream_enabled=False,
            name="Color",
        ),
    ],
    depth=None,
    ai_model=AiModelConfiguration(
        display_name="Thermal Person Detection",
        path="yolov6n_thermal_people_256x192",
        camera=dai.CameraBoardSocket.CAM_E,
    ),
)
