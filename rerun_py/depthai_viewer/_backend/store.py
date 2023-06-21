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

    def __init__(self) -> None:
        self._send_message_queue = Queue()

    def set_pipeline_config(self, pipeline_config: PipelineConfiguration) -> None:
        self._pipeline_config = pipeline_config

    def set_subscriptions(self, subscriptions: List[Topic]) -> None:
        self._subscriptions = subscriptions

    def reset(self) -> None:
        self._pipeline_config = None
        self._subscriptions = []

    @property
    def pipeline_config(self) -> Optional[PipelineConfiguration]:
        return self._pipeline_config

    @property
    def subscriptions(self) -> List[Topic]:
        return self._subscriptions

    def send_message_to_frontend(self, message: Message) -> None:
        self._send_message_queue.put(message)
