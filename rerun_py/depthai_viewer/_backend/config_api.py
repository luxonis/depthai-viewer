import asyncio
import json
from enum import Enum
from multiprocessing import Queue
from queue import Empty as QueueEmptyException
from signal import SIGINT, signal
from typing import Any, Dict, Optional, Tuple

import depthai as dai
import websockets
from websockets.server import WebSocketServerProtocol

from depthai_viewer._backend.device_configuration import (
    DeviceProperties,
    PipelineConfiguration,
)
from depthai_viewer._backend.store import Action
from depthai_viewer._backend.topic import Topic

signal(SIGINT, lambda *args, **kwargs: exit(0))

# Definitions for linting
# send actions to back
dispatch_action_queue: Queue  # type: ignore[type-arg]

# bool indicating action success
result_queue: Queue  # type: ignore[type-arg]

# Send messages from backend to frontend without the frontend sending a message first
send_message_queue: Queue  # type: ignore[type-arg]


def dispatch_action(action: Action, **kwargs) -> Tuple[bool, Dict[str, Any]]:  # type: ignore[no-untyped-def]
    """
    Dispatches an action that will be executed by store.py.

    Returns: (success: bool, result: Dict[str, Any]).
    """
    dispatch_action_queue.put((action, kwargs))
    return result_queue.get()  # type: ignore[no-any-return]


class MessageType:
    SUBSCRIPTIONS = "Subscriptions"  # Get or set subscriptions
    PIPELINE = "Pipeline"  # Get or Set pipeline
    DEVICES = "Devices"  # Get device list
    DEVICE = "DeviceProperties"  # Get or set device
    ERROR = "Error"  # Error message


class ErrorAction(Enum):
    NONE = "None"
    FULL_RESET = "FullReset"

    def __str__(self) -> str:
        return self.value


def error(message: str, action: ErrorAction) -> str:
    """Create an error message to send via ws."""
    return json.dumps({"type": MessageType.ERROR, "data": {"action": str(action), "message": message}})


async def ws_api(websocket: WebSocketServerProtocol) -> None:
    while True:
        raw_message = None
        try:
            raw_message = await asyncio.wait_for(websocket.recv(), 1)
        except asyncio.TimeoutError:
            pass
        except websockets.exceptions.ConnectionClosed:
            success, _ = dispatch_action(Action.RESET)  # type: ignore[assignment]
            if success:
                return
            raise Exception("Couldn't reset backend after websocket disconnect!")

        if raw_message:
            try:
                message: Dict[str, Any] = json.loads(raw_message)
            except json.JSONDecodeError:
                print("Failed to parse message: ", message)
                continue
            message_type = message.get("type", None)
            if not message_type:
                print("Missing message type")
                continue
            print("Got message: ", message)
            if message_type == MessageType.SUBSCRIPTIONS:
                data = message.get("data", {})
                subscriptions = [Topic.create(topic_name) for topic_name in data.get(MessageType.SUBSCRIPTIONS, [])]
                dispatch_action(Action.SET_SUBSCRIPTIONS, subscriptions=subscriptions)
                print("Subscriptions: ", subscriptions)
                active_subscriptions = [
                    topic.name  # type: ignore[attr-defined]
                    for topic in dispatch_action(Action.GET_SUBSCRIPTIONS)
                    if topic
                ]
                await websocket.send(json.dumps({"type": MessageType.SUBSCRIPTIONS, "data": active_subscriptions}))
            elif message_type == MessageType.PIPELINE:
                data = message.get("data", {})
                pipeline_config_json, runtime_only = data.get("Pipeline", ({}, False))
                pipeline_config = PipelineConfiguration(**pipeline_config_json)
                print("Pipeline config: ", pipeline_config)

                success, result = dispatch_action(
                    Action.UPDATE_PIPELINE, pipeline_config=pipeline_config, runtime_only=runtime_only
                )
                if runtime_only:
                    # Send a full reset if setting a runtime config fails.
                    # Don't send pipeline config to save bandwidth.
                    if not success:
                        await websocket.send(error("Failed to set runtime config", ErrorAction.FULL_RESET))
                    continue
                if success:
                    active_config: Optional[PipelineConfiguration] = dispatch_action(
                        Action.GET_PIPELINE
                    )  # type: ignore[assignment]
                    await websocket.send(
                        json.dumps(
                            {
                                "type": MessageType.PIPELINE,
                                "data": (active_config.dict(), False) if active_config is not None else None,
                            }
                        )
                    )
                else:
                    await websocket.send(error(result["message"], ErrorAction.FULL_RESET))
            elif message_type == MessageType.DEVICES:
                await websocket.send(
                    json.dumps(
                        {
                            "type": MessageType.DEVICES,
                            "data": [
                                d.getMxId() for d in dai.Device.getAllAvailableDevices()  # type: ignore[call-arg]
                            ],
                        }
                    )
                )

            elif message_type == MessageType.DEVICE:
                data = message.get("data", {})
                device_repr = data.get(message_type, {})
                device_id = device_repr.get("id", None)
                if device_id is None:
                    print("Missing device id")
                    continue
                success, result = dispatch_action(Action.SELECT_DEVICE, device_id=device_id)
                if success:
                    print("Selected device properties: ", result.get("device_properties", None))
                    if result.get("device_properties", None) is None:
                        print("Device props is non!")
                        await websocket.send(error("Unknown error", ErrorAction.FULL_RESET))
                        continue
                    device_properties = result.get("device_properties", None)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": MessageType.DEVICE,
                                "data": device_properties.dict()
                                if device_properties
                                else DeviceProperties(id="").dict(),
                            }
                        )  # type: ignore[union-attr]
                    )
                else:
                    await websocket.send(error(result.get("message", "Unknown error"), ErrorAction.FULL_RESET))

            else:
                print("Unknown message type: ", message_type)
                continue
        send_message = None
        try:
            send_message = send_message_queue.get(timeout=0.01)
        except QueueEmptyException:
            pass
        if send_message:
            print("Sending message: ", send_message)
            await websocket.send(send_message)


async def main() -> None:
    async with websockets.serve(ws_api, "localhost", 9001):  # type: ignore[attr-defined]
        await asyncio.Future()  # run forever


def start_api(
    _dispatch_action_queue: Queue, _result_queue: Queue, _send_message_queue: Queue  # type: ignore[type-arg]
) -> None:
    """
    Starts the websocket API.

    _dispatch_action_queue: Queue to send actions to store.py
    _result_queue: Queue to get results from store.py
    _send_message_queue: Queue to send messages to frontend.
    """
    global dispatch_action_queue
    dispatch_action_queue = _dispatch_action_queue
    global result_queue
    result_queue = _result_queue
    global send_message_queue
    send_message_queue = _send_message_queue

    asyncio.run(main())