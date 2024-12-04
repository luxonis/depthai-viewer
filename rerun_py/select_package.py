import os
import platform
import shutil

# Define source directories for different OS types
PACKAGES = {
    "Linux": "packages/linux",
    "Windows": "packages/windows",
    "Darwin": "packages/macos",
}

# Target destination
TARGET_DIR = "depthai_viewer/_backend/obscured_utilities/utilities/pyarmor_runtime_007125"

def copy_package():
    # Detect operating system
    system = platform.system()
    source_dir = PACKAGES.get(system)

    if not source_dir:
        raise RuntimeError(f"Unsupported OS: {system}")

    # Clean and recreate target directory
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)

    # Copy the correct files
    shutil.copytree(source_dir, TARGET_DIR, dirs_exist_ok=True)
    print(f"Copied {source_dir} to {TARGET_DIR}")

if __name__ == "__main__":
    copy_package()
