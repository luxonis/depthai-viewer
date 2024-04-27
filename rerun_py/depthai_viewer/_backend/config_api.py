import asyncio
import atexit
import json
from enum import Enum, auto
from multiprocessing import Queue
from queue import Empty as QueueEmptyException
from signal import SIGINT, signal
from typing import Any, Dict

import depthai as dai
import websockets
from websockets.server import WebSocketServerProtocol

from depthai_viewer._backend.device_configuration import PipelineConfiguration, ToFConfig
from depthai_viewer._backend.messages import (
    DevicesMessage,
    ErrorMessage,
    InfoMessage,
    Message,
    MessageType,
)
from depthai_viewer._backend.topic import Topic

atexit.register(lambda: print("Exiting..."))
signal(SIGINT, lambda *args, **kwargs: exit(0))

# Definitions for linting
# send actions to back
dispatch_action_queue: Queue  # type: ignore[type-arg]

# bool indicating action success
result_queue: Queue  # type: ignore[type-arg]

# Send messages from backend to frontend without the frontend sending a message first
send_message_queue: Queue  # type: ignore[type-arg]


class Action(Enum):
    UPDATE_PIPELINE = 0
    SELECT_DEVICE = auto()
    GET_SUBSCRIPTIONS = auto()
    SET_SUBSCRIPTIONS = auto()
    GET_PIPELINE = auto()
    SET_FLOOD_BRIGHTNESS = auto()
    SET_DOT_BRIGHTNESS = auto()
    RESET = auto()  # When anything bad happens, a reset occurs (like closing ws connection)
    GET_AVAILABLE_DEVICES = auto()
    SET_TOF_CONFIG = auto()


def dispatch_action(action: Action, **kwargs) -> Message:  # type: ignore[no-untyped-def]
    """
    Dispatches an action that will be executed by main.py.

    Returns: Message that will be sent to the frontend
    """
    dispatch_action_queue.put((action, kwargs))
    return result_queue.get()  # type: ignore[no-any-return]


async def send_message(websocket: WebSocketServerProtocol, message: Message) -> None:
    """Sends a message to the frontend without the frontend sending a message first."""
    if isinstance(message, InfoMessage) and not message.message:
        return
    await websocket.send(message.json())


async def ws_api(websocket: WebSocketServerProtocol) -> None:
    """
    Receives messages from the frontend, dispatches them to the backend and sends the result back to the frontend.

    Received Messages include the wanted state of the backend,
    e.g.: A DeviceMessage received from the frontend includes the device the user wants to select.
    The backend then tries to select the device and sends back a DeviceMessage
    with the selected device (selected device can be None if the selection failed).
    """
    while True:
        raw_message = None
        try:
            raw_message = await asyncio.wait_for(websocket.recv(), 1)
        except asyncio.TimeoutError:
            pass
        except websockets.exceptions.ConnectionClosed:
            if isinstance(dispatch_action(Action.RESET), ErrorMessage):
                raise Exception("Couldn't reset backend after websocket disconnect!")
            return

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

            if message_type == MessageType.SUBSCRIPTIONS:
                data = message.get("data", {})
                subscriptions = [Topic.create(topic_name) for topic_name in data.get(MessageType.SUBSCRIPTIONS, [])]
                await send_message(websocket, dispatch_action(Action.SET_SUBSCRIPTIONS, subscriptions=subscriptions))

            elif message_type == MessageType.PIPELINE:
                data = message.get("data", {})
                pipeline_config_json, runtime_only = data.get("Pipeline", ({}, False))
                pipeline_config = PipelineConfiguration(**pipeline_config_json)
                print("Pipeline config: ", pipeline_config)

                await send_message(
                    websocket,
                    dispatch_action(Action.UPDATE_PIPELINE, pipeline_config=pipeline_config, runtime_only=runtime_only),
                )

            elif message_type == MessageType.DEVICES:
                await send_message(
                    websocket,
                    DevicesMessage(dai.Device.getAllAvailableDevices()),  # type: ignore[call-arg]
                )

            elif message_type == MessageType.DEVICE:
                data = message.get("data", {})
                device_repr = data.get(message_type, {})
                device_id = device_repr.get("id", None)
                if device_id is None:
                    print("Missing device id")
                    continue
                await send_message(websocket, dispatch_action(Action.SELECT_DEVICE, device_id=device_id))
            elif message_type == MessageType.SET_FLOOD_BRIGHTNESS:
                data = message.get("data", {})
                flood_brightness = data.get(message_type, None)
                if flood_brightness is None:
                    print("Missing", message_type)
                    continue
                await send_message(
                    websocket, dispatch_action(Action.SET_FLOOD_BRIGHTNESS, flood_brightness=flood_brightness)
                )
            elif message_type == MessageType.SET_DOT_BRIGHTNESS:
                data = message.get("data", {})
                dot_brightness = data.get(message_type, None)
                if dot_brightness is None:
                    print("Missing dot", message_type)
                    continue
                await send_message(websocket, dispatch_action(Action.SET_DOT_BRIGHTNESS, dot_brightness=dot_brightness))
            elif message_type == MessageType.SET_TOF_CONFIG:
                data = message.get("data", {})
                tof_config = data.get(message_type, None)
                if len(tof_config) == 2:  # Disregard the camera board socket for now. Only support one.
                    tof_config = tof_config[1]
                else:
                    print("Invalid tof config: ", tof_config)
                    continue
                if tof_config is None:
                    print("Missing tof config")
                    continue
                try:
                    tof_config = ToFConfig(**tof_config)
                except Exception as e:
                    print(f"Failed to deserialize tof_config: {tof_config}: {e}")
                    continue
                await send_message(websocket, dispatch_action(Action.SET_TOF_CONFIG, tof_config=tof_config))
            else:
                print("Unknown message type: ", message_type)
                continue

        message_to_send = None
        try:
            message_to_send = send_message_queue.get(timeout=0.01)
            print("Got message to send: ", message_to_send)
        except QueueEmptyException:
            pass
        if message_to_send:
            print("Sending message: ", message_to_send)
            await send_message(websocket, message_to_send)


async def main(port: int = 9001) -> None:
    async with websockets.serve(ws_api, "localhost", port):  # type: ignore[attr-defined]
        await asyncio.Future()  # run forever


def start_api(
    _dispatch_action_queue: Queue,  # type: ignore[type-arg]
    _result_queue: Queue,  # type: ignore[type-arg]
    _send_message_queue: Queue,  # type: ignore[type-arg]
    port: int = 9001,
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

    asyncio.run(main(port))
