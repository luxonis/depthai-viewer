use super::depthai;
use super::ws::{ BackWsMessage as WsMessage, WebSocket, WsMessageData, WsMessageType };

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiError {
    pub detail: String,
}

impl Default for ApiError {
    fn default() -> Self {
        Self {
            detail: "ApiError".to_string(),
        }
    }
}

#[derive(Default)]
pub struct BackendCommChannel {
    pub ws: WebSocket,
}

impl BackendCommChannel {
    pub fn shutdown(&mut self) {
        self.ws.shutdown();
    }

    pub fn set_subscriptions(&mut self, subscriptions: &Vec<depthai::ChannelId>) {
        self.ws.send(
            serde_json
                ::to_string(
                    &(WsMessage {
                        kind: WsMessageType::Subscriptions,
                        data: WsMessageData::Subscriptions(subscriptions.clone()),
                        ..Default::default()
                    })
                )
                .unwrap()
        );
    }

    pub fn set_pipeline(&mut self, config: &depthai::DeviceConfig, runtime_only: bool) {
        self.ws.send(
            serde_json
                ::to_string(
                    &(WsMessage {
                        kind: WsMessageType::Pipeline,
                        data: WsMessageData::Pipeline((config.clone(), runtime_only)),
                        ..Default::default()
                    })
                )
                .unwrap()
        );
    }

    pub fn set_dot_brightness(&mut self, brightness: u32) {
        self.ws.send(
            serde_json
                ::to_string(
                    &(WsMessage {
                        kind: WsMessageType::SetDotBrightness,
                        data: WsMessageData::SetDotBrightness(brightness),
                        ..Default::default()
                    })
                )
                .unwrap()
        );
    }

    pub fn set_flood_brightness(&mut self, brightness: u32) {
        self.ws.send(
            serde_json
                ::to_string(
                    &(WsMessage {
                        kind: WsMessageType::SetFloodBrightness,
                        data: WsMessageData::SetFloodBrightness(brightness),
                        ..Default::default()
                    })
                )
                .unwrap()
        );
    }

    pub fn receive(&mut self) -> Option<WsMessage> {
        self.ws.receive()
    }

    pub fn get_devices(&mut self) {
        self.ws.send(
            serde_json
                ::to_string(
                    &(WsMessage {
                        kind: WsMessageType::Devices,
                        data: WsMessageData::Devices(Vec::new()),
                        ..Default::default()
                    })
                )
                .unwrap()
        );
    }

    pub fn select_device(&mut self, device_id: depthai::DeviceId) {
        self.ws.send(
            serde_json
                ::to_string(
                    &(WsMessage {
                        kind: WsMessageType::DeviceProperties,
                        data: WsMessageData::DeviceProperties(depthai::DeviceProperties {
                            id: device_id,
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                )
                .unwrap()
        );
    }
}
