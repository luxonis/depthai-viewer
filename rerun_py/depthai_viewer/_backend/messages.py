import json
from enum import Enum
from typing import List, Optional

import depthai as dai

from depthai_viewer._backend.device_configuration import DeviceProperties, PipelineConfiguration


class MessageType:
    SUBSCRIPTIONS = "Subscriptions"  # Get or set subscriptions
    PIPELINE = "Pipeline"  # Get or Set pipeline
    DEVICES = "Devices"  # Get device list
    DEVICE = "DeviceProperties"  # Get or set device
    SET_FLOOD_BRIGHTNESS = "SetFloodBrightness"  # Set floodlight
    SET_DOT_BRIGHTNESS = "SetDotBrightness"  # Set floodlight
    SET_TOF_CONFIG = "SetToFConfig"  # Set ToF config
    ERROR = "Error"  # Error message
    INFO = "Info"  # Info message
    WARNING = "Warning"  # Warning message


class ErrorAction(Enum):
    NONE = "None"
    FULL_RESET = "FullReset"

    def __str__(self) -> str:
        return self.value


class Message:
    message: Optional[str] = None

    def __init__(self) -> None:
        raise NotImplementedError

    def json(self) -> str:
        raise NotImplementedError


class WarningMessage(Message):
    def __init__(self, message: str):
        self.message = message

    def json(self) -> str:
        return json.dumps({"type": MessageType.WARNING, "data": {"message": self.message}})


class ErrorMessage(Message):
    def __init__(self, message: str, action: ErrorAction = ErrorAction.FULL_RESET):
        self.action = action
        self.message = message

    def json(self) -> str:
        return json.dumps({"type": MessageType.ERROR, "data": {"action": str(self.action), "message": self.message}})


class DevicesMessage(Message):
    def __init__(self, devices: List[dai.DeviceInfo], message: Optional[str] = None):
        self.devices = [
            {
                "mxid": d.getMxId(),
                "connection": "PoE" if d.protocol == dai.XLinkProtocol.X_LINK_TCP_IP else "Usb",
                "name": d.name,
            }
            for d in devices
        ]
        self.message = message

    def json(self) -> str:
        return json.dumps({"type": MessageType.DEVICES, "data": self.devices})


class DeviceMessage(Message):
    def __init__(self, device_props: Optional[DeviceProperties], message: Optional[str] = None):
        self.device_props = device_props
        self.message = message

    def json(self) -> str:
        return json.dumps(
            {
                "type": MessageType.DEVICE,
                "data": self.device_props.dict() if self.device_props else DeviceProperties(id="").dict(),
            }
        )


class SubscriptionsMessage(Message):
    def __init__(self, subscriptions: List[str], message: Optional[str] = None):
        self.subscriptions = subscriptions
        self.message = message

    def json(self) -> str:
        return json.dumps({"type": MessageType.SUBSCRIPTIONS, "data": self.subscriptions})


class PipelineMessage(Message):
    def __init__(
        self,
        pipeline_config: Optional[PipelineConfiguration],
        runtime_only: bool = False,
        message: Optional[str] = None,
    ):
        self.pipeline_config = pipeline_config
        self.runtime_only = runtime_only
        self.message = message

    def json(self) -> str:
        return json.dumps(
            {
                "type": MessageType.PIPELINE,
                "data": (self.pipeline_config.dict(), self.runtime_only) if self.pipeline_config else None,
                "message": self.message,
            }
        )


class InfoMessage(Message):
    def __init__(self, message: str):
        self.message = message

    def json(self) -> str:
        return json.dumps({"type": MessageType.INFO, "data": self.message})
