use egui::TextEdit;
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
                        re_log::warn!("Failed to read stdout {:?}", err);
                    }
                }
            }
        }
    }
}

pub struct DependencyInstaller {
    installed_environment: Option<BackendEnvironment>,
    process: task::JoinHandle<Result<(), InstallerProcessError>>,
    stdio_rx: crossbeam_channel::Receiver<String>,
    stdio_tx: crossbeam_channel::Sender<String>,
    stdio: String,
    backend_environment: BackendEnvironment,
}

impl DependencyInstaller {
    pub fn new(environment: BackendEnvironment) -> Self {
        let (stdio_tx, stdio_rx) = crossbeam_channel::unbounded();
        let process = InstallerProcess::spawn(environment.clone(), stdio_tx.clone());
        Self {
            installed_environment: None,
            process,
            stdio_rx,
            stdio_tx,
            stdio: String::with_capacity(4096),
            backend_environment: environment,
        }
    }

    pub fn show(&mut self, re_ui: &re_ui::ReUi, ui: &mut egui::Ui) {
        egui::Window::new("Dependency Installer").show(ui.ctx(), |ui| {
            ui.scope(|ui| {
                let mut style = ui.style_mut().clone();
                style.visuals.widgets.noninteractive.bg_fill = egui::Color32::WHITE;
                ui.set_style(style);
                if !self.process.is_finished() {
                    ui.label("Installing dependencies...");
                    egui::ScrollArea::vertical()
                        .max_height(200.0)
                        .max_width(400.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            ui.label(&self.stdio);
                        });
                } else {
                    // On successful install, a status dump is printed to stdout
                    match self.stdio.find("Status Dump: ") {
                        Some(mut status_dump_index) => {
                            status_dump_index += "Status Dump: ".len();
                            let status_dump: StatusDump =
                                serde_json::from_str(&self.stdio[status_dump_index..].trim())
                                    .unwrap();
                            self.installed_environment = Some(BackendEnvironment {
                                python_path: self.backend_environment.python_path.clone(),
                                venv_site_packages: Some(status_dump.venv_site_packages.clone()),
                            });
                        }
                        None => {
                            ui.label("Error installing dependencies");
                            if ui.button("Retry").clicked() {
                                self.process = InstallerProcess::spawn(
                                    self.backend_environment.clone(),
                                    self.stdio_tx.clone(),
                                );
                            }
                        }
                    }
                }
            });
        });
    }

    pub fn update(&mut self) {
        // Receive stdout from the installer process
        while let Ok(stdout) = self.stdio_rx.try_recv() {
            self.stdio.push_str(&stdout);
        }
    }

    /// Get the installed environment if it was successfully installed, otherwise always None
    pub fn try_get_installed_environment(&self) -> Option<BackendEnvironment> {
        self.installed_environment.clone()
    }
}
