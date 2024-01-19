from multiprocessing import Queue
from typing import List, Optional

from depthai_viewer._backend.device_configuration import PipelineConfiguration
from depthai_viewer._backend.messages import Message
from depthai_viewer._backend.topic import Topic


class Store:
    """Used to store common data that is used by the backend."""

    _pipeline_config: Optional[PipelineConfiguration] = None
    _subscriptions: List[Topic] = []
    _send_message_queue: Queue  # type: ignore[type-arg]
    _dot_brightness: int = 0
    _flood_brightness: int = 0

    def __init__(self) -> None:
        self._send_message_queue = Queue()

    def set_pipeline_config(self, pipeline_config: PipelineConfiguration) -> None:
        self._pipeline_config = pipeline_config

    def set_subscriptions(self, subscriptions: List[Topic]) -> None:
        self._subscriptions = subscriptions

    def set_dot_brightness(self, brightness: int) -> None:
        self._dot_brightness = brightness

    def set_flood_brightness(self, brightness: int) -> None:
        self._flood_brightness = brightness

    def reset(self) -> None:
        self._pipeline_config = None
        self._subscriptions = []

    @property
    def pipeline_config(self) -> Optional[PipelineConfiguration]:
        return self._pipeline_config

    @property
    def subscriptions(self) -> List[Topic]:
        return self._subscriptions

    @property
    def dot_brightness(self) -> int:
        return self._dot_brightness

    @property
    def flood_brightness(self) -> int:
        return self._flood_brightness

    def send_message_to_frontend(self, message: Message) -> None:
        self._send_message_queue.put(message)
