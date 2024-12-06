use crate::{
    depthai::depthai::{self},
    misc::ViewerContext,
};

use strum::IntoEnumIterator; // Needed for enum::iter()

/// The "Selection View" side-bar.
#[derive(serde::Deserialize, serde::Serialize, Default)]
#[serde(default)]
pub(crate) struct DeviceSettingsPanel {}

const CONFIG_UI_WIDTH: f32 = 280.0;

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
                fill: ui.visuals().panel_fill,
                // fill: egui::Color32::WHITE,
                ..Default::default()
            })
            .show_inside(ui, |ui| {
                (egui::Frame {
                    inner_margin: egui::Margin::same(re_ui::ReUi::view_padding()),
                    fill: ui.visuals().panel_fill,
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
        device_config: &mut depthai::DeviceConfig,
    ) {
        let mut connected_cameras_clone = ctx.depthai_state.get_connected_cameras().clone();
        for mut camera_features in ctx.depthai_state.get_connected_cameras_mut() {
            let Some(camera_config) = device_config
                .cameras
                .iter_mut()
                .find(|conf| conf.board_socket == camera_features.board_socket)
            else {
                continue;
            };
            // let text_color = ctx.re_ui.design_tokens.primary_700;
            let text_color = ui.style().visuals.strong_text_color();
            egui::CollapsingHeader::new(
                egui::RichText::new(
                    camera_features
                        .board_socket
                        .display_name(&connected_cameras_clone),
                )
                .color(text_color),
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
                                    .on_disabled_hover_ui(
                                        |ui| {
                                            ui.label(format!(
                                                "{res} will be available in a future release!"
                                            ));
                                        },
                                    );
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
                    if let Some(tof_config) = &mut camera_features.tof_config {
                        // Add a UI that is hidden by default Drop
                        ui.collapsing("ToF config", |mut ui| {
                            ui.vertical(|ui| {
                                ctx.re_ui.labeled_combo_box(
                                    ui,
                                    "Median Filter",
                                    format!("{:?}", tof_config.get_median_filter()),
                                    false,
                                    true,
                                    |ui| {
                                        let mut median_filter = tof_config.get_median_filter();
                                        for filter in depthai::MedianFilter::iter() {
                                            if ui
                                                .selectable_value(
                                                    &mut median_filter,
                                                    filter,
                                                    format!("{filter:?}"),
                                                )
                                                .changed()
                                            {
                                                tof_config.set_median_filter(median_filter);
                                            }
                                        }
                                    },
                                );
                                let mut phase_unwrapping_level =
                                    tof_config.get_phase_unwrapping_level();
                                if ctx
                                    .re_ui
                                    .labeled_dragvalue(
                                        ui,
                                        egui::Id::new("tof-phase-unwrap-level"),
                                        None,
                                        "Phase unwrap level",
                                        &mut phase_unwrapping_level,
                                        0..=6,
                                    )
                                    .changed()
                                {
                                    tof_config.set_phase_unwrapping_level(phase_unwrapping_level);
                                }
                                let mut phase_unwrap_error_threshold =
                                    tof_config.get_phase_unwrap_error_threshold();
                                if ctx
                                    .re_ui
                                    .labeled_dragvalue(
                                        ui,
                                        egui::Id::new("tof-phase-unwrap-error-threshold"),
                                        None,
                                        "Phase unwrap error threshold",
                                        &mut phase_unwrap_error_threshold,
                                        0..=u16::MAX,
                                    )
                                    .changed()
                                {
                                    tof_config.set_phase_unwrap_error_threshold(
                                        phase_unwrap_error_threshold,
                                    );
                                }
                                let mut enable_phase_unwrapping =
                                    tof_config.get_enable_phase_unwrapping().unwrap_or(false);
                                if ctx
                                    .re_ui
                                    .labeled_toggle_switch(
                                        ui,
                                        "Enable phase unwrapping",
                                        &mut enable_phase_unwrapping,
                                    )
                                    .changed()
                                {
                                    tof_config
                                        .set_enable_phase_unwrapping(Some(enable_phase_unwrapping));
                                }
                                let mut enable_fppn_correction =
                                    tof_config.get_enable_fppn_correction().unwrap_or(false);
                                if ctx
                                    .re_ui
                                    .labeled_toggle_switch(
                                        ui,
                                        "Enable FPPN correction",
                                        &mut enable_fppn_correction,
                                    )
                                    .changed()
                                {
                                    tof_config
                                        .set_enable_fppn_correction(Some(enable_fppn_correction));
                                }
                                let mut enable_optical_correction =
                                    tof_config.get_enable_optical_correction().unwrap_or(false);
                                if ctx
                                    .re_ui
                                    .labeled_toggle_switch(
                                        ui,
                                        "Enable optical correction",
                                        &mut enable_optical_correction,
                                    )
                                    .changed()
                                {
                                    tof_config.set_enable_optical_correction(Some(
                                        enable_optical_correction,
                                    ));
                                }
                                let mut enable_temperature_correction = tof_config
                                    .get_enable_temperature_correction()
                                    .unwrap_or(false);
                                if ctx
                                    .re_ui
                                    .labeled_toggle_switch(
                                        ui,
                                        "Enable temperature correction",
                                        &mut enable_temperature_correction,
                                    )
                                    .changed()
                                {
                                    tof_config.set_enable_temperature_correction(Some(
                                        enable_temperature_correction,
                                    ));
                                }
                                let mut enable_wiggle_correction =
                                    tof_config.get_enable_wiggle_correction().unwrap_or(false);
                                if ctx
                                    .re_ui
                                    .labeled_toggle_switch(
                                        ui,
                                        "Enable wiggle correction",
                                        &mut enable_wiggle_correction,
                                    )
                                    .changed()
                                {
                                    tof_config.set_enable_wiggle_correction(Some(
                                        enable_wiggle_correction,
                                    ));
                                }
                                let mut enable_phase_shuffle_temporal_filter = tof_config
                                    .get_enable_phase_shuffle_temporal_filter()
                                    .unwrap_or(false);
                                if ctx
                                    .re_ui
                                    .labeled_toggle_switch(
                                        ui,
                                        "Enable phase shuffle temporal filter",
                                        &mut enable_phase_shuffle_temporal_filter,
                                    )
                                    .changed()
                                {
                                    tof_config.set_enable_phase_shuffle_temporal_filter(Some(
                                        enable_phase_shuffle_temporal_filter,
                                    ));
                                }
                                let mut enable_burst_mode =
                                    tof_config.get_enable_burst_mode().unwrap_or(false);
                                if ctx
                                    .re_ui
                                    .labeled_toggle_switch(
                                        ui,
                                        "Enable burst mode",
                                        &mut enable_burst_mode,
                                    )
                                    .changed()
                                {
                                    tof_config.set_enable_burst_mode(Some(enable_burst_mode));
                                }
                            })
                        });
                    }
                    ctx.re_ui.labeled_toggle_switch(
                        ui,
                        "Stream",
                        &mut camera_config.stream_enabled,
                    );
                });
            });
        }
    }

    fn device_configuration_ui(ctx: &mut ViewerContext<'_>, ui: &mut egui::Ui) {
        let mut device_config = ctx.depthai_state.modified_device_config.clone();

        let text_color = ui.style().visuals.strong_text_color();
        // let text_color = ctx.re_ui.design_tokens.primary_700;
        ctx.re_ui.styled_scrollbar(
            ui,
            re_ui::ScrollAreaDirection::Vertical,
            [false; 2],
            false,
            |ui| {
                (egui::Frame {
                    fill: ui.style().visuals.panel_fill,
                    inner_margin: egui::Margin::symmetric(30.0, 21.0),
                    ..Default::default()
                })
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.collapsing(
                                egui::RichText::new("Recalibrate_device").color(text_color),
                                |ui| {
                                    ui.vertical(|ui| {
                                        ui.scope(|ui| {
                                            let mut style = ui.style_mut().clone();
                                            let color = style.visuals.selection.bg_fill;
                                            style.visuals.widgets.hovered.bg_fill = color;
                                            style.visuals.widgets.inactive.bg_fill = color;
                                            style.visuals.widgets.inactive.fg_stroke.color = egui::Color32::WHITE;
                                            style.visuals.widgets.hovered.fg_stroke.color = egui::Color32::WHITE;
                                            style.spacing.button_padding = egui::Vec2::new(24.0, 4.0);
                                            ui.set_style(style);

                                            if ui
                                                .add_sized(
                                                    [CONFIG_UI_WIDTH, re_ui::ReUi::box_height()],
                                                    egui::Button::new("Recalibration"),
                                                )
                                                .clicked()
                                            {
                                                ctx.depthai_state.start_recalibration(1);
                                            }

                                            if ui
                                                .add_sized(
                                                    [CONFIG_UI_WIDTH, re_ui::ReUi::box_height()],
                                                    egui::Button::new("Calibration Check"),
                                                )
                                                .clicked()
                                            {
                                                ctx.depthai_state.calibration_check(1);
                                            }

                                            if ui
                                                .add_sized(
                                                    [CONFIG_UI_WIDTH, re_ui::ReUi::box_height()],
                                                    egui::Button::new("Flash Calibration"),
                                                )
                                                .clicked()
                                            {
                                                ctx.depthai_state.flash_calibration(1);
                                            }

                                            if ui
                                                .add_sized(
                                                    [CONFIG_UI_WIDTH, re_ui::ReUi::box_height()],
                                                    egui::Button::new("Restore Factory Calibration"),
                                                )
                                                .clicked()
                                            {
                                                ctx.depthai_state.flash_factorycalibration(1);
                                            }

                                            let calibration_mode_display = ctx.depthai_state.modified_device_config.calibration_mode.as_ref().map(|mode| mode.to_string()).unwrap_or_else(|| depthai::CalibrationMode::Factory.to_string());
                                            ui.scope(|ui| {
                                                // Dropdown for Calibration Modes
                                                ctx.re_ui.labeled_combo_box(
                                                    ui,
                                                    "Calibration Mode",
                                                    calibration_mode_display,
                                                    true,
                                                    true,
                                                    |ui| {
                                                        for mode in depthai::CalibrationMode::all_modes() {
                                                            ui.selectable_value(
                                                                &mut ctx.depthai_state.modified_device_config.calibration_mode,
                                                                Some(mode.clone()), // Wrap in Some
                                                                mode.to_string(),
                                                            );
                                                        }
                                                    },
                                                );

                                                // "Apply" Button
                                                let mut style = ui.style_mut().clone();
                                                let color = style.visuals.selection.bg_fill;

                                                // Apply button customization
                                                style.visuals.widgets.hovered.bg_fill = color;
                                                style.visuals.widgets.hovered.weak_bg_fill = color;
                                                style.visuals.widgets.inactive.bg_fill = color;
                                                style.visuals.widgets.inactive.weak_bg_fill = color;
                                                style.visuals.widgets.inactive.fg_stroke.color = egui::Color32::WHITE;
                                                style.visuals.widgets.hovered.fg_stroke.color = egui::Color32::WHITE;
                                                style.spacing.button_padding = egui::Vec2::new(24.0, 4.0);
                                                ui.set_style(style);

                                                // Add the button
                                                if ui
                                                    .add_sized(
                                                        [CONFIG_UI_WIDTH, re_ui::ReUi::box_height()],
                                                        egui::Button::new("Apply"),
                                                    )
                                                    .clicked()
                                                {
                                                    // Only send the selected mode to the backend when the button is clicked
                                                    let selected_mode = ctx.depthai_state.modified_device_config.calibration_mode.clone().unwrap_or(depthai::CalibrationMode::Factory);
                                                    ctx.depthai_state.set_calibration_mode(selected_mode);
                                                }
                                            });
                                        });
                                    });
                                },
                            );
                            Self::camera_config_ui(ctx, ui, &mut device_config);
                            ui.collapsing(
                                egui::RichText::new("AI settings").color(text_color),
                                |ui| {
                                    ui.vertical(|ui| {
                                        ui.set_width(CONFIG_UI_WIDTH);
                                        ctx.re_ui.labeled_combo_box(
                                            ui,
                                            "AI Model",
                                            match &device_config.ai_model {
                                                Some(model) => model.display_name.clone(),
                                                None => "No Model".to_string(),
                                            },
                                            false,
                                            true,
                                            |ui| {
                                                for nn in &ctx.depthai_state.neural_networks {
                                                    ui.selectable_value(
                                                        &mut device_config.ai_model,
                                                        nn.clone(),
                                                        match &nn {
                                                            Some(model) => model.display_name.clone(),
                                                            None => "No Model".to_string(),
                                                        },
                                                    );
                                                }
                                            },
                                        );
                                        match &mut device_config.ai_model {
                                            Some(model) => {
                                                ctx.re_ui.labeled_combo_box(
                                                    ui,
                                                    "Run on",
                                                    model.camera.display_name(ctx.depthai_state.get_connected_cameras()),
                                                    false,
                                                    true,
                                                    |ui| {
                                                        let filtered_cameras: Vec<_> =  ctx.depthai_state.get_connected_cameras()
                                                            .iter() // iterates over references
                                                            .filter(|cam| {
                                                                !(cam.supported_types.contains(
                                                                    &depthai::CameraSensorKind::THERMAL,
                                                                ) || cam.supported_types.contains(
                                                                    &depthai::CameraSensorKind::TOF,
                                                                ))
                                                            })
                                                            .collect();
                                                        for cam in filtered_cameras {
                                                            ui.selectable_value(
                                                                &mut model.camera,
                                                                cam.board_socket,
                                                                cam.board_socket.display_name(ctx.depthai_state.get_connected_cameras()),
                                                            );
                                                        }
                                                    },
                                                );
                                            }
                                            None => {
                                            }
                                        };
                                    });
                            });

                            let mut stereo: depthai::StereoDepthConfig = device_config.stereo.unwrap_or_default();
                            ui.add_enabled_ui(
                                ctx.depthai_state.selected_device.has_stereo_pairs(),
                                |ui| {
                                    egui::CollapsingHeader::new(
                                        egui::RichText::new("Stereo depth settings").color(text_color),
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
                                            ctx.re_ui.labeled_combo_box(ui, "Preset", format!("{}", stereo.depth_preset), false, true, |ui| {
                                                for preset in depthai::DepthProfilePreset::iter() {
                                                    ui.selectable_value(&mut stereo.depth_preset, preset, format!("{preset}"));
                                                }
                                            });
                                            let (cam1, cam2) = stereo.stereo_pair;
                                            ctx.re_ui.labeled_combo_box(
                                                ui,
                                                "Camera Pair",
                                                format!(
                                                    "{}, {}",
                                                    cam1.display_name(ctx.depthai_state.get_connected_cameras()),
                                                    cam2.display_name(ctx.depthai_state.get_connected_cameras())
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
                                                            &mut stereo.stereo_pair,
                                                            *pair,
                                                            format!(
                                                                "{} {}",
                                                                pair.0.display_name(ctx.depthai_state.get_connected_cameras()),
                                                                pair.1.display_name(ctx.depthai_state.get_connected_cameras())
                                                            ),
                                                        );
                                                    }
                                                },
                                            );
                                            ctx.re_ui.labeled_combo_box(
                                                ui,
                                                "Align to",
                                                stereo.align.display_name(ctx.depthai_state.get_connected_cameras()),
                                                false,
                                                true,
                                                |ui| {
                                                    for align in ctx.depthai_state.get_connected_cameras() {
                                                        ui.selectable_value(
                                                            &mut stereo.align,
                                                            align.board_socket,
                                                            align.board_socket.display_name(ctx.depthai_state.get_connected_cameras()),
                                                        );
                                                    }
                                                },
                                            );
                                            let dot_drag = ctx.re_ui.labeled_dragvalue(
                                                ui,
                                                egui::Id::from("Dot brightness [mA]"),
                                                None,
                                                "Dot brightness [mA]",
                                                &mut device_config.dot_brightness,
                                                0..=1200);
                                                if dot_drag.drag_released() {
                                                    ctx.depthai_state.set_dot_brightness(device_config.dot_brightness);
                                                } else if dot_drag.changed() && !dot_drag.dragged() {
                                                    // Dragging isn't ongoing, but the value changed
                                                    ctx.depthai_state.set_dot_brightness(device_config.dot_brightness);
                                                }
                                            let flood_drag = ctx.re_ui.labeled_dragvalue(
                                                ui,
                                                egui::Id::from("Flood light brightness [mA]"),
                                                None,
                                                "Flood light brightness [mA]",
                                                &mut device_config.flood_brightness,
                                                0..=1500,
                                            );
                                            if flood_drag.drag_released() {
                                                ctx.depthai_state.set_flood_brightness(device_config.flood_brightness);
                                            } else if flood_drag.changed() && !flood_drag.dragged() {
                                                // Dragging isn't ongoing, but the value changed
                                                ctx.depthai_state.set_flood_brightness(device_config.flood_brightness);
                                            }

                                            if stereo.depth_preset == depthai::DepthProfilePreset::CUSTOM { // advanced settings. Leave it out unless no preset is enabled.
                                                ctx.re_ui.labeled_toggle_switch(
                                                    ui,
                                                    "LR Check",
                                                    &mut stereo.lr_check,
                                                );
                                                ctx.re_ui.labeled_combo_box(
                                                    ui,
                                                    "Median Filter",
                                                    format!("{:?}", stereo.median),
                                                    false,
                                                    true,
                                                    |ui| {
                                                        for filter in depthai::MedianFilter::iter()
                                                        {
                                                            ui.selectable_value(
                                                                &mut stereo.median,
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
                                                    &mut stereo.lrc_threshold,
                                                    0..=10,
                                                );
                                                ctx.re_ui.labeled_toggle_switch(
                                                    ui,
                                                    "Extended Disparity",
                                                    &mut stereo.extended_disparity,
                                                );
                                                ctx.re_ui.labeled_toggle_switch(
                                                    ui,
                                                    "Subpixel Disparity",
                                                    &mut stereo.subpixel_disparity,
                                                );
                                                ctx.re_ui.labeled_dragvalue(
                                                    ui,
                                                    egui::Id::from("Sigma"),
                                                    Some(100.0),
                                                    "Sigma",
                                                    &mut stereo.sigma,
                                                    0..=65535,
                                                );
                                                ctx.re_ui.labeled_dragvalue(
                                                    ui,
                                                    egui::Id::from("Confidence"),
                                                    Some(100.0),
                                                    "Confidence",
                                                    &mut stereo.confidence,
                                                    0..=255,
                                                );
                                            }
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

                            device_config.stereo = Some(stereo);
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
                                            let color = style.visuals.selection.bg_fill;
                                            //TODO(tomas):Investigate whether this could be default button style
                                            if apply_enabled {
                                                style.visuals.widgets.hovered.bg_fill = color;
                                                style.visuals.widgets.hovered.weak_bg_fill = color;
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
