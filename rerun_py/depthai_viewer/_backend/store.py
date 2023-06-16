from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from depthai_viewer._backend.device_configuration import PipelineConfiguration
from depthai_viewer._backend.topic import Topic
from depthai_viewer._backend.messages import *


class Store:
    _pipeline_config: Optional[PipelineConfiguration] = PipelineConfiguration()
    _subscriptions: List[Topic] = []

    def set_pipeline_config(self, pipeline_config: PipelineConfiguration) -> None:
        self._pipeline_config = pipeline_config

    def set_subscriptions(self, subscriptions: List[Topic]) -> None:
        self._subscriptions = subscriptions

    def reset(self) -> None:
        self._pipeline_config = None
        self._subscriptions = []

    @property
    def pipeline_config(self) -> PipelineConfiguration:
        return self._pipeline_config

    @property
    def subscriptions(self) -> List[Topic]:
        return self._subscriptions
