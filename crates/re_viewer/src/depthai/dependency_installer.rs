use egui::TextEdit;
use std::{os::unix::prelude::FileExt, path::PathBuf};
use subprocess::{Popen, PopenConfig, PopenError, Redirection};
use tempdir::TempDir;

use crate::app::BackendEnvironment;

#[derive(Clone, Debug)]
pub struct DependencyInstallerResult {}

struct ProcessWrapper {
    process: Result<Popen, PopenError>,
    log_output: String,
}

impl ProcessWrapper {
    pub fn update(&mut self) {
        if let Ok(process) = &mut self.process {
            let tmp_buf = [0u8; 256];
            // Read stdout into buffer
            if let Some(out_f) = process.stdout {
                if let Ok(n_read) = out_f.read_at(&mut tmp_buf, self.log_output.len() - 1 as u64) {
                    if let Ok(utf8) = std::str::from_utf8(&tmp_buf[..n_read]) {
                        self.log_output.push_str(utf8);
                    } else {
                        re_log::warn!("Failed to convert stdout to utf8");
                    }
                } else {
                    re_log::warn!("Failed to read stdout");
                }
            }
        }
    }
}

pub struct DependencyInstaller {
    result: Option<DependencyInstallerResult>,
    environment: BackendEnvironment,
    process: ProcessWrapper,
}

impl DependencyInstaller {
    pub fn new(environment: BackendEnvironment) -> Self {
        let process = Self::create_installer_process(environment.python_path.clone());
        Self {
            result: None,
            environment,
            process,
        }
    }

    fn create_installer_process(python_path: String) -> ProcessWrapper {
        ProcessWrapper {
            process: Popen::create(
                &[
                    python_path,
                    String::from("-m"),
                    String::from("depthai_viewer.install_requirements"),
                ],
                PopenConfig {
                    stdout: Redirection::Pipe,
                    ..Default::default()
                },
            ),
            log_output: String::with_capacity(4096),
        }
    }

    pub fn update(&mut self, ctx: &egui::Context) {
        egui::Window::new("Dependency Installer").show(ctx, |ui| {
            self.process.update();
            if let Ok(process) = &mut self.process.process {
                ui.label("Installing dependencies...");
                ui.collapsing("Details", |ui| {
                    TextEdit::multiline(self.process.stdout_file)
                });
            } else {
                ui.label("Error installing dependencies");
                if ui.button("Retry").clicked() {
                    self.process =
                        Self::create_installer_process(self.environment.python_path.clone());
                }
            }
        });
    }

    pub fn get_result(&self) -> Option<DependencyInstallerResult> {
        self.result.clone()
    }
}
