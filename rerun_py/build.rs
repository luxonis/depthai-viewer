use std::process::Command;
fn main() {
    // Required for `cargo build` to work on mac: https://pyo3.rs/v0.14.2/building_and_distribution.html#macos
    pyo3_build_config::add_extension_module_link_args();

    re_build_build_info::export_env_vars();
    println!("Running pre-build script...");
    let status = Command::new("python")
        .arg("select_package.py")
        .status()
        .expect("Failed to execute pre-build script");
    if !status.success() {
        panic!("Pre-build script failed.");
    }
}
