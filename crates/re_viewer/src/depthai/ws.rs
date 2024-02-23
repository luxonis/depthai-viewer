use crossbeam_channel;
use ewebsock::{WsEvent, WsMessage};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::ControlFlow;
use std::process::exit;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use super::depthai;

async fn spawn_ws_client(
    port: u32,
    recv_tx: crossbeam_channel::Sender<WsMessage>,
    send_rx: crossbeam_channel::Receiver<WsMessage>,
    shutdown: Arc<AtomicBool>,
    connected: Arc<AtomicBool>,
) {
    let (error_tx, error_rx) = crossbeam_channel::unbounded();
    // Retry connection until successful
    loop {
        let recv_tx = recv_tx.clone();
        let error_tx = error_tx.clone();
        let connected = connected.clone();
        if let Ok(sender) = ewebsock::ws_connect(
            String::from(format!("ws://localhost:{port}")),
            Box::new(move |event| {
                match event {
                    WsEvent::Opened => {
                        re_log::info!("Websocket opened");
                        connected.store(true, std::sync::atomic::Ordering::SeqCst);
                        ControlFlow::Continue(())
                    }
                    WsEvent::Message(message) => {
                        // re_log::debug!("Websocket message");
                        recv_tx.send(message);
                        ControlFlow::Continue(())
                    }
                    WsEvent::Error(e) => {
                        // re_log::info!("Websocket Error: {:?}", e);
                        connected.store(false, std::sync::atomic::Ordering::SeqCst);
                        error_tx.send(e);
                        ControlFlow::Break(())
                    }
                    WsEvent::Closed => {
                        // re_log::info!("Websocket Closed");
                        error_tx.send(String::from("Websocket Closed"));
                        ControlFlow::Break(())
                    }
                }
            }),
        )
        .as_mut()
        {
            while error_rx.is_empty() {
                if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                    re_log::debug!("Shutting down websocket client");
                    exit(0);
                }
                if let Ok(message) = send_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    re_log::debug!("Sending message: {:?}", message);
                    sender.send(message);
                }
            }
            for error in error_rx.try_iter() {
                re_log::debug!("Websocket error: {:}", error);
            }
        } else {
            re_log::error!("Coudln't create websocket");
        }
        if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
            re_log::debug!("Shutting down websocket client");
            exit(0);
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

type RuntimeOnly = bool;

#[derive(Serialize, Deserialize, fmt::Debug)]
pub enum WsMessageData {
    Subscriptions(Vec<depthai::ChannelId>),
    Devices(Vec<depthai::DeviceInfo>),
    DeviceProperties(depthai::DeviceProperties),
    Pipeline((depthai::DeviceConfig, RuntimeOnly)),
    SetFloodBrightness(u32),
    SetDotBrightness(u32),
    Error(depthai::Error),
    Info(depthai::Info),
    Warning(depthai::Warning),
}

#[derive(Deserialize, Serialize, fmt::Debug)]
pub enum WsMessageType {
    Subscriptions,
    Devices,
    DeviceProperties,
    Pipeline,
    SetFloodBrightness,
    SetDotBrightness,
    Error,
    Info,
    Warning,
}

impl Default for WsMessageType {
    fn default() -> Self {
        Self::Error
    }
}

#[derive(Serialize, fmt::Debug)]
pub struct BackWsMessage {
    #[serde(rename = "type")]
    pub kind: WsMessageType,
    pub data: WsMessageData,
    pub message: Option<String>,
}

impl<'de> Deserialize<'de> for BackWsMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        pub struct Message {
            #[serde(rename = "type")]
            pub kind: WsMessageType,
            pub data: serde_json::Value,
            pub message: Option<String>,
        }

        let message = Message::deserialize(deserializer)?;
        let data = match message.kind {
            WsMessageType::Subscriptions => WsMessageData::Subscriptions(
                serde_json::from_value(message.data).unwrap_or_default(),
            ),
            WsMessageType::Devices => {
                WsMessageData::Devices(serde_json::from_value(message.data).unwrap_or_default())
            }
            WsMessageType::DeviceProperties => {
                WsMessageData::DeviceProperties(serde_json::from_value(message.data).unwrap())
            }
            WsMessageType::Pipeline => {
                WsMessageData::Pipeline(serde_json::from_value(message.data).unwrap())
            }
            WsMessageType::Error => {
                WsMessageData::Error(serde_json::from_value(message.data).unwrap_or_default())
            }
            WsMessageType::Info => {
                WsMessageData::Info(serde_json::from_value(message.data).unwrap_or_default())
            }
            WsMessageType::Warning => {
                WsMessageData::Warning(serde_json::from_value(message.data).unwrap_or_default())
            }
            WsMessageType::SetDotBrightness => {
                WsMessageData::SetDotBrightness(serde_json::from_value(message.data).unwrap())
            }
            WsMessageType::SetFloodBrightness => {
                WsMessageData::SetFloodBrightness(serde_json::from_value(message.data).unwrap())
            }
        };

        Ok(Self {
            kind: message.kind,
            data,
            message: message.message,
        })
    }
}

impl Default for BackWsMessage {
    fn default() -> Self {
        Self {
            kind: WsMessageType::Error.into(),
            data: WsMessageData::Error(depthai::Error::default()),
            message: None,
        }
    }
}

struct WsInner {
    receiver: crossbeam_channel::Receiver<WsMessage>,
    sender: crossbeam_channel::Sender<WsMessage>,
    shutdown: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<()>,
    connected: Arc<AtomicBool>,
}

pub struct WebSocket {
    inner: Option<WsInner>,
}

impl Default for WebSocket {
    fn default() -> Self {
        Self {
            inner: None
        }
    }
}

impl WebSocket {

    pub fn is_initialized(&self) -> bool {self.inner.is_some()}

    pub fn connect(&mut self, port: u32) {
        re_log::debug!("Creating websocket client");
        let (recv_tx, recv_rx) = crossbeam_channel::unbounded();
        let (send_tx, send_rx) = crossbeam_channel::unbounded();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let connected = Arc::new(AtomicBool::new(false));
        let connected_clone = connected.clone();
        let task;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            re_log::debug!("Using current tokio runtime");
            task = handle.spawn(spawn_ws_client(
                port,
                recv_tx,
                send_rx,
                shutdown_clone,
                connected_clone,
            ));
        } else {
            re_log::debug!("Creating new tokio runtime");
            task = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .spawn(spawn_ws_client(
                    port,
                    recv_tx,
                    send_rx,
                    shutdown_clone,
                    connected_clone,
                ));
        }

        self.inner = Some(
            WsInner {
                receiver: recv_rx,
                sender: send_tx,
                shutdown,
                task,
                connected
            }
        );
    }

    pub fn is_connected(&self) -> bool {
        match &self.inner {
            Some(ws_state) => {
                ws_state.connected.load(std::sync::atomic::Ordering::SeqCst)
            },
            None => false
        }
    }

    pub fn shutdown(&mut self) {
        match &mut self.inner {
            Some(ws_state) => ws_state.shutdown.store(true, std::sync::atomic::Ordering::SeqCst),
            None => ()
        }
    }

    pub fn receive(&self) -> Option<BackWsMessage> {
        match &self.inner {
            Some(ws_state) => {
                if let Ok(message) = ws_state.receiver.try_recv() {
                    match message {
                        WsMessage::Text(text) => {
                            re_log::debug!("Received: {:?}", text);
                            match serde_json::from_str::<BackWsMessage>(&text.as_str()) {
                                Ok(back_message) => {
                                    return Some(back_message);
                                }
                                Err(err) => {
                                    re_log::error!("Error: {:}", err);
                                    return None;
                                }
                            }
                        }
                        _ => {
                            return None;
                        }
                    }
                } else {
                    None
                }
            },
            None => None
        }
    }

    pub fn send(&self, message: String) {
        match &self.inner {
            Some(ws_state) => {
                ws_state.sender.send(WsMessage::Text(message));
                // TODO(filip): This is a hotfix for the websocket not sending the message
                // This makes the websocket actually send the previous msg
                // It has to be something related to tokio::spawn, because it works fine when just running in the current thread
                ws_state.sender.send(WsMessage::Text("".to_string()));
            },
            None => ()
        }

    }
}
