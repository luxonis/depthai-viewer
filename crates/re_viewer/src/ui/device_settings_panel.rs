use crate::{
    depthai::depthai::{self, CameraBoardSocket},
    misc::ViewerContext,
};

use strum::IntoEnumIterator; // Needed for enum::iter()

/// The "Selection View" side-bar.
#[derive(serde::Deserialize, serde::Serialize, Default)]
#[serde(default)]
pub(crate) struct DeviceSettingsPanel {}

const CONFIG_UI_WIDTH: f32 = 224.0;

impl DeviceSettingsPanel {
    #[allow(clippy::unused_self)]
    pub fn show_panel(&mut self, ctx: &mut ViewerContext<'_>, ui: &mut egui::Ui) {
        let mut available_devices = ctx.depthai_state.get_devices();
        let currently_selected_device = ctx.depthai_state.selected_device.clone();
        let mut combo_device: depthai::DeviceInfo = currently_selected_device.info.clone();
        if !combo_device.mxid.is_empty() && available_devices.is_empty() {
            available_devices.push(combo_device.clone());
        }

        let mut show_device_config = true;
        egui::CentralPanel::default()
            .frame(egui::Frame {
                inner_margin: egui::Margin::same(0.0),
                fill: egui::Color32::WHITE,
                ..Default::default()
            })
            .show_inside(ui, |ui| {
                (egui::Frame {
                    inner_margin: egui::Margin::same(re_ui::ReUi::view_padding()),
                    ..Default::default()
                })
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Use up all the horizontal space (color)
                        ui.add_sized(
                            [ui.available_width(), re_ui::ReUi::box_height() + 20.0],
                            |ui: &mut egui::Ui| {
                                ui.horizontal(|ui| {
                                    ui.add(|ui: &mut egui::Ui| {
                                        egui::ComboBox::from_id_source("device_select")
                                            .selected_text(if !combo_device.mxid.is_empty() {
                                                combo_device.display_text()
                                            } else {
                                                "No device selected".to_owned()
                                            })
                                            .width(re_ui::ReUi::box_width())
                                            .wrap(true)
                                            .show_ui(ui, |ui: &mut egui::Ui| {
                                                if ui
                                                    .selectable_value(
                                                        &mut combo_device,
                                                        depthai::DeviceInfo::default(),
                                                        "No device",
                                                    )
                                                    .changed()
                                                {
                                                    ctx.depthai_state
                                                        .select_device(combo_device.mxid.clone());
                                                }
                                                for device in available_devices {
                                                    if ui
                                                        .selectable_value(
                                                            &mut combo_device,
                                                            device.clone(),
                                                            device.display_text(),
                                                        )
                                                        .changed()
                                                    {
                                                        ctx.depthai_state.select_device(
                                                            combo_device.mxid.clone(),
                                                        );
                                                    }
                                                }
                                            })
                                            .response
                                    });
                                    if !currently_selected_device.id.is_empty()
                                        && !ctx.depthai_state.is_update_in_progress()
                                    {
                                        ui.add_sized(
                                            [
                                                re_ui::ReUi::box_width() / 2.0,
                                                re_ui::ReUi::box_height() + 1.0,
                                            ],
                                            |ui: &mut egui::Ui| {
                                                ui.scope(|ui| {
                                                    let mut style = ui.style_mut().clone();
                                                    // TODO(filip): Create a re_ui bound button with this style
                                                    let color =
                                                        ctx.re_ui.design_tokens.error_bg_color;
                                                    let hover_color = ctx
                                                        .re_ui
                                                        .design_tokens
                                                        .error_hover_bg_color;
                                                    style.visuals.widgets.hovered.bg_fill =
                                                        hover_color;
                                                    style.visuals.widgets.hovered.weak_bg_fill =
                                                        hover_color;
                                                    style.visuals.widgets.inactive.bg_fill = color;
                                                    style.visuals.widgets.inactive.weak_bg_fill =
                                                        color;
                                                    style
                                                        .visuals
                                                        .widgets
                                                        .inactive
                                                        .fg_stroke
                                                        .color = egui::Color32::WHITE;
                                                    style.visuals.widgets.hovered.fg_stroke.color =
                                                        egui::Color32::WHITE;

                                                    ui.set_style(style);
                                                    if ui.button("Disconnect").clicked() {
                                                        ctx.depthai_state
                                                            .select_device(String::new());
                                                    }
                                                })
                                                .response
                                            },
                                        );
                                    }
                                })
                                .response
                            },
                        );
                    });

                    let device_selected = !ctx.depthai_state.selected_device.id.is_empty();
                    let pipeline_update_in_progress = ctx.depthai_state.is_update_in_progress();
                    if pipeline_update_in_progress {
                        ui.add_sized([CONFIG_UI_WIDTH, 10.0], |ui: &mut egui::Ui| {
                            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                ui.label(if device_selected {
                                    "Updating Pipeline"
                                } else {
                                    "Selecting Device"
                                });
                                ui.add(egui::Spinner::new())
                            })
                            .response
                        });
                        show_device_config = false;
                    }
                    if !device_selected && !pipeline_update_in_progress {
                        ui.label("Select a device to continue...");
                        show_device_config = false;
                    }
                });

                if show_device_config {
                    Self::device_configuration_ui(ctx, ui);
                }
            });
    }

    fn camera_config_ui(
        ctx: &mut ViewerContext<'_>,
        ui: &mut egui::Ui,
        camera_features: &depthai::CameraFeatures,
        camera_config: &mut depthai::CameraConfig,
    ) {
        let primary_700 = ctx.re_ui.design_tokens.primary_700;
        egui::CollapsingHeader::new(
            egui::RichText::new(camera_features.board_socket.display_name(ctx)).color(primary_700),
        )
        .default_open(true)
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.set_width(CONFIG_UI_WIDTH);
                ctx.re_ui.labeled_combo_box(
                    ui,
                    "Resolution",
                    format!("{}", camera_config.resolution),
                    false,
                    true,
                    |ui| {
                        for res in camera_features.resolutions.clone() {
                            let disabled = false;
                            ui.add_enabled_ui(!disabled, |ui| {
                                ui.selectable_value(
                                    &mut camera_config.resolution,
                                    res,
                                    format!("{res}"),
                                )
                                .on_disabled_hover_ui(|ui| {
                                    ui.label(format!(
                                        "{res} will be available in a future release!"
                                    ));
                                });
                            });
                        }
                    },
                );
                ctx.re_ui.labeled_dragvalue(
                    ui,
                    egui::Id::from("fps"), // TODO(filip): using "fps" as id causes all fps sliders to be linked - This is a bug, but also kind of a feature
                    None,
                    "FPS",
                    &mut camera_config.fps,
                    0..=camera_features.max_fps,
                );
                ctx.re_ui
                    .labeled_checkbox(ui, "Stream", &mut camera_config.stream_enabled);
            });
        });
    }

    fn device_configuration_ui(ctx: &mut ViewerContext<'_>, ui: &mut egui::Ui) {
        let mut device_config = ctx.depthai_state.modified_device_config.clone();
        let primary_700 = ctx.re_ui.design_tokens.primary_700;
        let connected_cameras = ctx.depthai_state.get_connected_cameras().clone();

        ctx.re_ui.styled_scrollbar(
            ui,
            re_ui::ScrollAreaDirection::Vertical,
            [false; 2],
            false,
            |ui| {
                (egui::Frame {
                    fill: ctx.re_ui.design_tokens.gray_50,
                    inner_margin: egui::Margin::symmetric(30.0, 21.0),
                    ..Default::default()
                })
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            for cam in connected_cameras.clone() {
                                let Some(config) = device_config.cameras
                                .iter_mut()
                                .find(|conf| conf.board_socket == cam.board_socket) else {
                                continue;
                            };
                                Self::camera_config_ui(ctx, ui, &cam, config);
                            }

                            ui.collapsing(
                                egui::RichText::new("AI settings").color(primary_700),
                                |ui| {
                                    ui.vertical(|ui| {
                                        ui.set_width(CONFIG_UI_WIDTH);
                                        ctx.re_ui.labeled_combo_box(
                                            ui,
                                            "AI Model",
                                            device_config.ai_model.display_name.clone(),
                                            false,
                                            true,
                                            |ui| {
                                                for nn in &ctx.depthai_state.neural_networks {
                                                    ui.selectable_value(
                                                        &mut device_config.ai_model,
                                                        nn.clone(),
                                                        &nn.display_name,
                                                    );
                                                }
                                            },
                                        );
                                        ctx.re_ui.labeled_combo_box(
                                            ui,
                                            "Run on",
                                            device_config.ai_model.camera.display_name(ctx),
                                            false,
                                            true,
                                            |ui| {
                                                for cam in &connected_cameras {
                                                    ui.selectable_value(
                                                        &mut device_config.ai_model.camera,
                                                        cam.board_socket,
                                                        cam.board_socket.display_name(ctx),
                                                    );
                                                }
                                            },
                                        );
                                    });
                                },
                            );

                            let mut depth = device_config.depth.unwrap_or_default();
                            ui.add_enabled_ui(
                                ctx.depthai_state.selected_device.has_stereo_pairs(),
                                |ui| {
                                    egui::CollapsingHeader::new(
                                        egui::RichText::new("Depth Settings").color(primary_700),
                                    )
                                    .open(
                                        if !ctx.depthai_state.selected_device.has_stereo_pairs() {
                                            Some(false)
                                        } else {
                                            None
                                        },
                                    )
                                    .show(ui, |ui| {
                                        ui.vertical(|ui| {
                                            ui.set_width(CONFIG_UI_WIDTH);
                                            let (cam1, cam2) = depth.stereo_pair;
                                            ctx.re_ui.labeled_combo_box(
                                                ui,
                                                "Camera Pair",
                                                format!(
                                                    "{}, {}",
                                                    cam1.display_name(ctx),
                                                    cam2.display_name(ctx)
                                                ),
                                                false,
                                                true,
                                                |ui| {
                                                    for pair in &ctx
                                                        .depthai_state
                                                        .selected_device
                                                        .stereo_pairs
                                                    {
                                                        ui.selectable_value(
                                                            &mut depth.stereo_pair,
                                                            *pair,
                                                            format!(
                                                                "{} {}",
                                                                pair.0.display_name(ctx),
                                                                pair.1.display_name(ctx)
                                                            ),
                                                        );
                                                    }
                                                },
                                            );
                                            ctx.re_ui.labeled_checkbox(
                                                ui,
                                                "LR Check",
                                                &mut depth.lr_check,
                                            );
                                            ctx.re_ui.labeled_combo_box(
                                                ui,
                                                "Align to",
                                                depth.align.display_name(ctx),
                                                false,
                                                true,
                                                |ui| {
                                                    for align in &connected_cameras {
                                                        ui.selectable_value(
                                                            &mut depth.align,
                                                            align.board_socket,
                                                            align.board_socket.display_name(ctx),
                                                        );
                                                    }
                                                },
                                            );
                                            ctx.re_ui.labeled_combo_box(
                                                ui,
                                                "Median Filter",
                                                format!("{:?}", depth.median),
                                                false,
                                                true,
                                                |ui| {
                                                    for filter in depthai::DepthMedianFilter::iter()
                                                    {
                                                        ui.selectable_value(
                                                            &mut depth.median,
                                                            filter,
                                                            format!("{filter:?}"),
                                                        );
                                                    }
                                                },
                                            );
                                            ctx.re_ui.labeled_dragvalue(
                                                ui,
                                                egui::Id::from("LR Threshold"),
                                                Some(100.0),
                                                "LR Threshold",
                                                &mut depth.lrc_threshold,
                                                0..=10,
                                            );
                                            ctx.re_ui.labeled_checkbox(
                                                ui,
                                                "Extended Disparity",
                                                &mut depth.extended_disparity,
                                            );
                                            ctx.re_ui.labeled_checkbox(
                                                ui,
                                                "Subpixel Disparity",
                                                &mut depth.subpixel_disparity,
                                            );
                                            ctx.re_ui.labeled_dragvalue(
                                                ui,
                                                egui::Id::from("Sigma"),
                                                Some(100.0),
                                                "Sigma",
                                                &mut depth.sigma,
                                                0..=65535,
                                            );
                                            ctx.re_ui.labeled_dragvalue(
                                                ui,
                                                egui::Id::from("Confidence"),
                                                Some(100.0),
                                                "Confidence",
                                                &mut depth.confidence,
                                                0..=255,
                                            );
                                            ctx.re_ui.labeled_toggle_switch(
                                                ui,
                                                "Depth enabled",
                                                &mut device_config.depth_enabled,
                                            );
                                        });
                                    })
                                    .header_response
                                    .on_disabled_hover_ui(
                                        |ui| {
                                            ui.label(
                                                "Selected device doesn't have any stereo pairs!",
                                            );
                                        },
                                    );
                                },
                            );

                            device_config.depth = Some(depth);
                            ctx.depthai_state.modified_device_config = device_config.clone();
                            ui.vertical(|ui| {
                                ui.horizontal(|ui| {
                                    let apply_enabled = {
                                        if let Some(applied_config) =
                                            &ctx.depthai_state.applied_device_config.config
                                        {
                                            let only_runtime_configs_changed =
                                                depthai::State::only_runtime_configs_changed(
                                                    applied_config,
                                                    &device_config,
                                                );
                                            let apply_enabled = !only_runtime_configs_changed
                                                && ctx
                                                    .depthai_state
                                                    .applied_device_config
                                                    .config
                                                    .is_some()
                                                && device_config != applied_config.clone()
                                                && !ctx.depthai_state.selected_device.id.is_empty()
                                                && !ctx.depthai_state.is_update_in_progress();

                                            if !apply_enabled && only_runtime_configs_changed {
                                                ctx.depthai_state
                                                    .set_pipeline(&mut device_config, true);
                                            }
                                            apply_enabled
                                        } else {
                                            !ctx.depthai_state
                                                .applied_device_config
                                                .update_in_progress
                                        }
                                    };

                                    ui.add_enabled_ui(apply_enabled, |ui| {
                                        ui.scope(|ui| {
                                            let mut style = ui.style_mut().clone();
                                            if apply_enabled {
                                                let color =
                                                    ctx.re_ui.design_tokens.primary_bg_color;
                                                let hover_color =
                                                    ctx.re_ui.design_tokens.primary_hover_bg_color;
                                                style.visuals.widgets.hovered.bg_fill = hover_color;
                                                style.visuals.widgets.hovered.weak_bg_fill =
                                                    hover_color;
                                                style.visuals.widgets.inactive.bg_fill = color;
                                                style.visuals.widgets.inactive.weak_bg_fill = color;
                                                style.visuals.widgets.inactive.fg_stroke.color =
                                                    egui::Color32::WHITE;
                                                style.visuals.widgets.hovered.fg_stroke.color =
                                                    egui::Color32::WHITE;
                                            }
                                            style.spacing.button_padding =
                                                egui::Vec2::new(24.0, 4.0);
                                            ui.set_style(style);
                                            if ui
                                                .add_sized(
                                                    [CONFIG_UI_WIDTH, re_ui::ReUi::box_height()],
                                                    egui::Button::new("Apply"),
                                                )
                                                .clicked()
                                            {
                                                ctx.depthai_state
                                                    .set_pipeline(&mut device_config, false);
                                            }
                                        });
                                    });
                                });
                            });
                        });
                        ui.allocate_space(ui.available_size());
                    });
                });
            },
        );
        // Set a more visible scroll bar color
    }
}
