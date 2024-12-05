use std::process::Command;
use std::path::Path;
fn main() {
    // Required for `cargo build` to work on mac: https://pyo3.rs/v0.14.2/building_and_distribution.html#macos
    pyo3_build_config::add_extension_module_link_args();

    re_build_build_info::export_env_vars();
    let script_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("select_package.py");

    // Check if the script exists
    if !script_path.exists() {
        panic!("Pre-build script not found at: {:?}", script_path);
    }

    // Execute the script
    println!("Running pre-build script...");
    let status = Command::new("python")
        .arg(script_path)
        .status()
        .expect("Failed to execute pre-build script");

    if !status.success() {
        panic!("Pre-build script failed with status: {:?}", status);
    }

    // Rerun the build script if this script changes
    println!("cargo:rerun-if-changed=select_package.py");
}
