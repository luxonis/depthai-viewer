use std::process::Command;
use std::path::Path;
fn main() {
    // Required for `cargo build` to work on mac: https://pyo3.rs/v0.14.2/building_and_distribution.html#macos
    pyo3_build_config::add_extension_module_link_args();

    re_build_build_info::export_env_vars();
    let script_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("packages/installer.py");
    if !script_path.exists() {
        panic!("Pre-build script not found: {:?}", script_path);
    }

    println!("Running pre-build script...");

    // Try `python` first
    let python_result = Command::new("python3")
        .arg(&script_path)
        .status();

    // If `python` fails, fall back to `python3`
    let status = match python_result {
        Ok(status) if status.success() => status,
        _ => Command::new("python")
            .arg(&script_path)
            .status()
            .expect("Failed to execute pre-build script with both `python` and `python3`"),
    };

    if !status.success() {
        panic!("Pre-build script failed with status: {:?}", status);
    }

    println!("Pre-build script completed successfully.");
    println!("cargo:rerun-if-changed=select_package.py");
}
