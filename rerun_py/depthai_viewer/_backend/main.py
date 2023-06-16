import threading
from multiprocessing import Queue
from queue import Empty as QueueEmptyException
from typing import Optional

import depthai_viewer as viewer
from depthai_viewer._backend.config_api import start_api, Action
from depthai_viewer._backend.device_configuration import (
    DeviceProperties,
)
from depthai_viewer._backend.messages import *
from depthai_viewer._backend.store import Store
from depthai_viewer._backend.device import Device

viewer.init("Depthai Viewer")
viewer.connect()

import time


class DepthaiViewerBack:
    _device: Optional[Device] = None

    # Queues for communicating with the API process
    action_queue: Queue  # type: ignore[type-arg]
    result_queue: Queue  # type: ignore[type-arg]
    send_message_queue: Queue  # type: ignore[type-arg]

    def __init__(self) -> None:
        self.action_queue = Queue()
        self.result_queue = Queue()
        self.send_message_queue = Queue()

        self.store = Store()
        self.api_process = threading.Thread(
            target=start_api, args=(self.action_queue, self.result_queue, self.send_message_queue)
        )
        self.api_process.start()
        self.run()

    def set_device(self, device: Optional[Device] = None) -> None:
        self._device = device

    def on_reset(self) -> Message:
        print("Resetting...")
        if self._device:
            print("Closing device...")
            self._device.close_oak_cam()
            self.set_device(None)
        self.store.reset()
        print("Done")
        return InfoMessage("Reset successful")

    def on_select_device(self, device_id: str) -> Message:
        print("Selecting device: ", device_id)
        if self._device:
            self.on_reset()
        if device_id == "":
            return DeviceMessage(DeviceProperties(id=""), "Device successfully unselected")
        try:
            self.set_device(Device(device_id, self.store))
        except RuntimeError as e:
            print("Failed to select device:", e)
            return ErrorMessage(f"{str(e)}, Try to connect the device to a different port.")
        try:
            if self._device is not None:
                device_properties = self._device.get_device_properties()
                return DeviceMessage(device_properties, "Device selected successfully")
            return ErrorMessage("Couldn't select device")
        except RuntimeError as e:
            print("Failed to get device properties:", e)
            self.on_reset()
            self.send_message_queue.put(ErrorMessage("Device disconnected!"))
            print("Restarting backend...")
            # For now exit the backend, the frontend will restart it
            # (TODO(filip): Why does "Device already closed or disconnected: Input/output error happen")
            exit(-1)

    def on_update_pipeline(self, runtime_only: bool) -> Message:
        if not self._device:
            print("No device selected, can't update pipeline!")
            return ErrorMessage("No device selected, can't update pipeline!")
        print("Updating pipeline...")
        message = ErrorMessage("Couldn't update pipeline")
        if self.store.pipeline_config is not None:
            message = self._device.update_pipeline(self.store.pipeline_config, runtime_only)
        if isinstance(message, InfoMessage):
            return PipelineMessage(self.store.pipeline_config)
        return message

    def handle_action(self, action: str, **kwargs) -> Message:
        if action == Action.UPDATE_PIPELINE:
            pipeline_config = kwargs.get("pipeline_config", None)
            if pipeline_config is not None:
                old_pipeline_config = self.store.pipeline_config
                self.store.set_pipeline_config(pipeline_config)
                message = self.on_update_pipeline(kwargs.get("runtime_only"))  # type: ignore[arg-type]
                if isinstance(message, ErrorMessage):
                    self.store.set_pipeline_config(old_pipeline_config)
                return message
            else:
                return ErrorMessage("Pipeline config not provided")
        elif action == Action.SELECT_DEVICE:
            device_id = kwargs.get("device_id", None)
            if device_id is not None:
                self.device_id = device_id
                return self.on_select_device(device_id)
            else:
                return ErrorMessage("Device id not provided")

        # TODO(filip): Fully deprecate subscriptions (Only IMU uses subscriptions)
        elif action == Action.GET_SUBSCRIPTIONS:
            return self.store.subscriptions  # type: ignore[return-value]
        elif action == Action.SET_SUBSCRIPTIONS:
            self.store.set_subscriptions(kwargs.get("subscriptions", []))
            return SubscriptionsMessage([topic.name for topic in self.store.subscriptions])
        elif action == Action.GET_PIPELINE:
            return self.store.pipeline_config  # type: ignore[return-value]
        elif action == Action.RESET:
            return self.on_reset()
        return ErrorMessage(f"Action: {action} not implemented")

    def run(self) -> None:
        """Handles ws messages and polls OakCam."""
        while True:
            try:
                action, kwargs = self.action_queue.get(timeout=0.0001)
                print("Handling action: ", action)
                self.result_queue.put(self.handle_action(action, **kwargs))
            except QueueEmptyException:
                pass

            if self._device:
                self._device.update()
                if self._device.is_closed():
                    self.on_reset()
                    self.send_message_queue.put(ErrorMessage("Device disconnected"))


if __name__ == "__main__":
    DepthaiViewerBack()
