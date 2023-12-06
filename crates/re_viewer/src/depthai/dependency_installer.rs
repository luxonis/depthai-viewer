use instant::Instant;
use serde::{Deserialize, Serialize};
use std::io::Read;
use subprocess::{ExitStatus, Popen, PopenConfig, PopenError, Redirection};
use tokio::task;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StatusDump {
    venv_site_packages: String,
}

use crate::app::BackendEnvironment;

/// The installer process which is run in a separate thread
/// Runs install_requirements.py and captures stdout and stderr
/// Sends stdout and stderr to the main thread via a channel
struct InstallerProcess {
    process: Result<Popen, PopenError>,
    log_output: String,
    update_task: Option<task::JoinHandle<()>>,
    stdio_tx: crossbeam_channel::Sender<String>,
}

#[derive(Debug)]
pub enum InstallerProcessError {
    Error(String),
}

impl InstallerProcess {
    pub fn spawn(
        environment: BackendEnvironment,
        stdio_tx: crossbeam_channel::Sender<String>,
    ) -> task::JoinHandle<Result<(), InstallerProcessError>> {
        task::spawn_blocking(move || {
            let mut installer_process = InstallerProcess {
                process: Popen::create(
                    &[
                        environment.python_path.clone(),
                        String::from("-m"),
                        String::from("depthai_viewer.install_requirements"),
                    ],
                    PopenConfig {
                        stdout: Redirection::Pipe,
                        ..Default::default()
                    },
                ),
                log_output: String::with_capacity(4096),
                update_task: None,
                stdio_tx,
            };
            installer_process.run()
        })
    }

    /// Runs the main loop of the installer process
    /// Read stdout and send it to the main thread
    /// On error, return the exit status, otherwise Ok(())
    pub fn run(&mut self) -> Result<(), InstallerProcessError> {
        loop {
            self.update();
            if let Some(exit_status) = self.get_exit_status() {
                println!("Got exit status: {:?}", exit_status);
                println!("Log output: {}", self.log_output);
                if exit_status.success() {
                    return Ok(());
                } else {
                    let mut stderr = String::with_capacity(4096);
                    if let Ok(process) = &mut self.process {
                        // process.stderr.as_ref().unwrap().read_to_string(&mut stderr);
                        return Err(InstallerProcessError::Error(stderr));
                    } else {
                        return Err(InstallerProcessError::Error(
                            "Failed to spawn installer process".to_string(),
                        ));
                    }
                }
            }
        }
    }

    pub fn get_exit_status(&mut self) -> Option<ExitStatus> {
        if let Ok(process) = &mut self.process {
            if let Some(exit_status) = process.poll() {
                re_log::debug!("Installer process exited with {:?}", exit_status);
                return Some(exit_status);
            }
        }
        None
    }

    pub fn update(&mut self) {
        if let Ok(process) = &mut self.process {
            if let Some(exit_status) = process.poll() {
                re_log::debug!("Installer process exited with {:?}", exit_status);
                return;
            }
            let mut tmp_buf = [0u8; 256];
            // Read stdout into buffer
            if let Some(out_f) = &mut process.stdout {
                match out_f.read(&mut tmp_buf) {
                    std::io::Result::Ok(n_read) => {
                        if let Ok(utf8) = std::str::from_utf8(&tmp_buf[..n_read]) {
                            self.log_output.push_str(utf8);
                            self.stdio_tx.send(utf8.to_string());
                        } else {
                            re_log::warn!("Failed to convert stdout to utf8");
                        }
                    }
                    std::io::Result::Err(err) => {
                        re_log::warn!("Failed to read stdout {err}");
                    }
                }
            }
        }
    }
}

pub struct PulsatingIcon {
    icon: re_ui::Icon,
    pulsating: bool,
    pulsating_timer: Instant,
}

impl PulsatingIcon {
    pub fn new(icon: re_ui::Icon) -> Self {
        Self {
            icon,
            pulsating: false,
            pulsating_timer: Instant::now(),
        }
    }

    pub fn start_pulsating(&mut self) {
        self.pulsating = true;
        self.pulsating_timer = Instant::now();
    }

    pub fn stop_pulsating(&mut self) {
        self.pulsating = false;
    }

    pub fn show(&mut self, re_ui: &re_ui::ReUi, ui: &mut egui::Ui) {
        let mut progress = 255;
        if self.pulsating {
            let elapsed = self.pulsating_timer.elapsed().as_secs_f32();
            let progress_f32 = if elapsed < 1.0 {
                elapsed
            } else {
                2.0 - elapsed
            } + 0.2; // Never go fully transparent
            progress = (progress_f32.clamp(0.0, 1.0) * 255.0) as u8;
        }
        if self.pulsating_timer.elapsed() > instant::Duration::from_secs(2) {
            self.pulsating_timer = Instant::now();
        }
        let tint = egui::Color32::from_rgba_premultiplied(progress, progress, progress, progress);
        let icon_image = re_ui.icon_image(&self.icon);
        ui.add(egui::Image::new(icon_image.texture_id(ui.ctx()), [124.0, 124.0]).tint(tint));
    }
}

pub struct DependencyInstaller {
    installed_environment: Option<BackendEnvironment>,
    process: task::JoinHandle<Result<(), InstallerProcessError>>,
    stdio_rx: crossbeam_channel::Receiver<String>,
    stdio_tx: crossbeam_channel::Sender<String>,
    stdio: String,
    backend_environment: BackendEnvironment,
    pulsating_dai_icon: PulsatingIcon,
    installation_acknowledged: bool,
}

impl DependencyInstaller {
    pub fn new(environment: BackendEnvironment) -> Self {
        let (stdio_tx, stdio_rx) = crossbeam_channel::unbounded();
        let process = InstallerProcess::spawn(environment.clone(), stdio_tx.clone());
        let mut pulsating_dai_icon = PulsatingIcon::new(re_ui::icons::DEPTHAI_ICON);
        pulsating_dai_icon.start_pulsating();
        Self {
            installed_environment: None,
            process,
            stdio_rx,
            stdio_tx,
            stdio: String::with_capacity(4096),
            backend_environment: environment,
            installation_acknowledged: false,
            pulsating_dai_icon,
        }
    }

    pub fn show(&mut self, re_ui: &re_ui::ReUi, ui: &mut egui::Ui) {
        let frame = egui::Frame::default()
            .fill(egui::Color32::WHITE)
            .stroke(egui::Stroke::new(1.0, egui::Color32::GRAY))
            .inner_margin(egui::Margin::symmetric(16.0, 0.0))
            .rounding(8.0);

        egui::Window::new("Dependency Installer")
            .title_bar(false)
            .frame(frame)
            .collapsible(false)
            .resizable(true)
            .default_height(600.0)
            .show(ui.ctx(), |ui| {
                let frame = egui::Frame::default().inner_margin(egui::Margin {
                    top: 24.0,
                    bottom: 4.0,
                    ..0.0.into()
                });
                egui::TopBottomPanel::top("header")
                    .frame(frame)
                    .show_separator_line(false)
                    .show_inside(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            self.pulsating_dai_icon.show(re_ui, ui);
                            if !self.dependencies_installed() {
                                if self.process.is_finished() {
                                    ui.label("Error installing dependencies");
                                    self.pulsating_dai_icon.stop_pulsating();
                                    if ui.button("Retry").clicked() {
                                        self.pulsating_dai_icon.start_pulsating();
                                        self.process = InstallerProcess::spawn(
                                            self.backend_environment.clone(),
                                            self.stdio_tx.clone(),
                                        );
                                    }
                                } else {
                                    ui.label("Installing dependencies...");
                                }
                            } else {
                                ui.label(
                                    egui::RichText::new("Dependencies installed successfully!")
                                        .color(re_ui.design_tokens.success_400),
                                );
                                self.pulsating_dai_icon.stop_pulsating();
                                if ui.button("Ok").clicked() {
                                    self.installation_acknowledged = true;
                                }
                            }
                        });
                    });

                let frame = egui::Frame::default().inner_margin(0.0);
                // Needed to contain the central panel within the window. Why? I don't know.
                egui::TopBottomPanel::bottom("invisible")
                    .frame(frame)
                    .show_separator_line(false)
                    .show_inside(ui, |_| ());

                let frame = egui::Frame::default()
                    .fill(egui::Color32::WHITE)
                    .rounding(4.0)
                    .stroke(egui::Stroke::new(1.0, re_ui.design_tokens.gray_400))
                    .inner_margin(egui::Margin::same(12.0));

                egui::CollapsingHeader::new("Details").default_open(true).show_unindented(ui, |ui| {
                    egui::CentralPanel::default()
                        .frame(frame)
                        .show_inside(ui, |ui| {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                                if re_ui.small_icon_button(ui, &re_ui::icons::COPY).clicked() {
                                    crate::misc::Clipboard::with(|clipboard| {
                                        clipboard.set_text(self.stdio.clone());
                                    });
                                }
                            });
                            re_ui.styled_scrollbar(
                                ui,
                                re_ui::ScrollAreaDirection::Both,
                                [false; 2],
                                true,
                                |ui| {
                                    ui.label(&self.stdio);
                                },
                            );
                        });
                });
            });
    }

    fn dependencies_installed(&self) -> bool {
        self.installed_environment.is_some()
    }

    pub fn update(&mut self) {
        // Receive stdout from the installer process
        while let Ok(stdout) = self.stdio_rx.try_recv() {
            self.stdio.push_str(&stdout);
        }

        if self.process.is_finished() {
            match self.stdio.find("Status Dump: ") {
                Some(mut status_dump_index) => {
                    status_dump_index += "Status Dump: ".len();
                    let status_dump: StatusDump =
                        serde_json::from_str(&self.stdio[status_dump_index..].trim()).unwrap();
                    self.installed_environment = Some(BackendEnvironment {
                        python_path: self.backend_environment.python_path.clone(),
                        venv_site_packages: Some(status_dump.venv_site_packages.clone()),
                    });
                }
                None => {}
            }
        }
    }

    /// Get the installed environment if it was successfully installed and the user pressed OK, otherwise always None
    pub fn try_get_installed_environment(&self) -> Option<BackendEnvironment> {
        if self.installation_acknowledged {
            return self.installed_environment.clone();
        }
        None
    }
}
