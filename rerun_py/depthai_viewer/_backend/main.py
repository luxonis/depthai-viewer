import threading
import traceback
from multiprocessing import Queue
from queue import Empty as QueueEmptyException
from typing import Optional

import sentry_sdk

from depthai_viewer import version as depthai_viewer_version
from depthai_viewer._backend.config_api import Action, start_api
from depthai_viewer._backend.device import Device
from depthai_viewer._backend.device_configuration import DeviceProperties
from depthai_viewer._backend.messages import (
    DeviceMessage,
    ErrorMessage,
    InfoMessage,
    Message,
    PipelineMessage,
    SubscriptionsMessage,
)
from depthai_viewer._backend.store import Store

try:
    sentry_sdk.init(  # type: ignore[abstract]
        dsn="https://37decdc44d584dca906e43ebd7fd1508@sentry.luxonis.com/16",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
        release=f"depthai-viewer@{depthai_viewer_version()}",
    )
except Exception:  # Be generic, a sentry failure should not crash the app
    print("Failed to initialize sentry")


class DepthaiViewerBack:
    _device: Optional[Device] = None
    port: int = 9001

    # Queues for communicating with the API process
    action_queue: Queue  # type: ignore[type-arg]
    result_queue: Queue  # type: ignore[type-arg]

    def __init__(self, port: int, sdk_port: int) -> None:
        self.action_queue = Queue()
        self.result_queue = Queue()

        self.store = Store()
        self.store.viewer_address = f"127.0.0.1:{sdk_port}"
        self.api_process = threading.Thread(
            target=start_api,
            args=(self.action_queue, self.result_queue, self.store._send_message_queue, port),  # This is a bit ugly
        )
        self.api_process.start()
        self.run()

    def set_device(self, device: Optional[Device] = None) -> None:
        self._device = device

    def on_reset(self) -> Message:
        print("Resetting...")
        if self._device:
            print("Closing device...")
            self._device.close_oak()
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
            self.store.send_message_to_frontend(ErrorMessage("Device disconnected!"))
            print("Restarting backend...")
            # For now exit the backend, the frontend will restart it
            # TODO(filip): Why does "Device already closed or disconnected: Input/output error happen"
            exit(-1)

    def on_update_pipeline(self, runtime_only: bool) -> Message:
        if not self._device:
            print("No device selected, can't update pipeline!")
            return ErrorMessage("No device selected, can't update pipeline!")
        print("Updating pipeline...")
        try:
            message = self._device.update_pipeline(runtime_only)
        except RuntimeError as e:
            print("Failed to update pipeline:", e)
            return ErrorMessage(str(e))
        if isinstance(message, InfoMessage):
            return PipelineMessage(self.store.pipeline_config)
        return message

    def handle_action(self, action: Action, **kwargs) -> Message:  # type: ignore[no-untyped-def]
        if action == Action.UPDATE_PIPELINE:
            pipeline_config = kwargs.get("pipeline_config", None)
            if pipeline_config is not None:
                old_pipeline_config = self.store.pipeline_config
                self.store.set_pipeline_config(pipeline_config)  # type: ignore[arg-type]
                message = self.on_update_pipeline(kwargs.get("runtime_only"))  # type: ignore[arg-type]
                if isinstance(message, ErrorMessage):
                    self.store.set_pipeline_config(old_pipeline_config)  # type: ignore[arg-type]
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
        elif action == Action.SET_FLOOD_BRIGHTNESS:
            self.store.set_flood_brightness(kwargs.get("flood_brightness", 0))
            print("Set flood: ", kwargs.get("flood_brightness", 0))
            if self._device and self._device._oak:
                self._device._oak.device.setIrFloodLightBrightness(self.store.flood_brightness)
                return InfoMessage("Floodlight set successfully")
            return ErrorMessage("No device selected")
        elif action == Action.SET_DOT_BRIGHTNESS:
            print("Set dot: ", kwargs.get("dot_brightness", 0))
            self.store.set_dot_brightness(kwargs.get("dot_brightness", 0))
            if self._device and self._device._oak:
                self._device._oak.device.setIrLaserDotProjectorBrightness(self.store.dot_brightness)
                return InfoMessage("Dot projector set successfully")
            return ErrorMessage("No device selected")
        elif action == Action.SET_TOF_CONFIG:
            if self._device and self._device._oak:
                if tof_component := self._device.get_tof_component():
                    if tof_config := kwargs.get("tof_config", None):
                        tof_cfg = tof_config.to_dai()
                        self.store.set_tof_config(tof_cfg)
                        tof_component.control.send_controls(tof_cfg)
                        return InfoMessage("ToF config updated successfully")
                    return ErrorMessage("ToF config not provided")
                return ErrorMessage("Failed to update ToF config. ToF node wasn't found.")
            return ErrorMessage("No device selected")
        return ErrorMessage(f"Action: {action} not implemented")

    def run(self) -> None:
        """Handles ws messages and polls OakCam."""
        while True:
            try:
                action, kwargs = self.action_queue.get(timeout=0.0001)
                # print("Handling action: ", action)
                self.result_queue.put(self.handle_action(action, **kwargs))
            except QueueEmptyException:
                pass

            if self._device:
                try:
                    self._device.update()
                except Exception as e:
                    print("Error while updating device:", end="")
                    # Print exception with traceback
                    print("\n".join(traceback.format_exception(type(e), e, e.__traceback__)))
                    self.store.send_message_to_frontend(ErrorMessage("Depthai error: " + str(e)))
                    self.on_reset()
                    continue
                if self._device.is_closed():
                    self.on_reset()
                    self.store.send_message_to_frontend(ErrorMessage("Device disconnected"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Depthai Viewer backend")
    parser.add_argument("--port", required=False, type=int, default=9001, help="Port to run the backend on")
    parser.add_argument("--sdk-port", required=False, type=int, default=9876, help="The rerun port")
    args = parser.parse_args()

    DepthaiViewerBack(args.port, args.sdk_port)
