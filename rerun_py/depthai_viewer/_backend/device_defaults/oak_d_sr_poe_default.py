import depthai as dai
from depthai_viewer._backend.device_configuration import (
    CameraConfiguration,
    CameraSensorResolution,
    PipelineConfiguration,
)

config = PipelineConfiguration(
    cameras=[
        CameraConfiguration(
            fps=15,
            resolution=CameraSensorResolution.THE_480_P,
            kind=dai.CameraSensorType.TOF,
            board_socket=dai.CameraBoardSocket.CAM_A,
            name="ToF",
        ),
        CameraConfiguration(
            fps=15,
            resolution=CameraSensorResolution.THE_720_P,
            kind=dai.CameraSensorType.COLOR,
            board_socket=dai.CameraBoardSocket.CAM_B,
            stream_enabled=True,
            name="Left",
        ),
        CameraConfiguration(
            fps=15,
            resolution=CameraSensorResolution.THE_720_P,
            kind=dai.CameraSensorType.COLOR,
            board_socket=dai.CameraBoardSocket.CAM_C,
            stream_enabled=True,
            name="Right",
        ),
    ],
    depth=None,
    ai_model=None,
)
