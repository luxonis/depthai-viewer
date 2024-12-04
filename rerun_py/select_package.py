import os
import platform
import shutil

# Base directory (absolute path of the current script's directory)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Define source directories for different OS types
PACKAGES = {
    "Linux": os.path.join(BASE_DIR, "packages", "linux_x86_64"),
    "Windows": os.path.join(BASE_DIR, "packages", "windows_x86_64"),
    "Darwin": os.path.join(BASE_DIR, "packages", "darwin_x86_64"),
}

# Target destination
TARGET_DIR = os.path.join(BASE_DIR, "depthai_viewer", "_backend", "obscured_utilities", "utilities", "pyarmor_runtime_007125")

def copy_package():
    # Detect operating system
    system = platform.system()
    source_dir = PACKAGES.get(system)

    if not source_dir or not os.path.exists(source_dir):
        raise RuntimeError(f"Package directory not found for OS: {system}. Expected path: {source_dir}")

    print(f"[pre-build] Detected OS: {system}")

    # Clean and recreate target directory
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Copy common files (__pycache__ and __init__.py) from packages directory
    common_source_dir = os.path.join(BASE_DIR, "packages")
    for item in ["__init__.py"]:
        item_path = os.path.join(common_source_dir, item)
        target_path = os.path.join(TARGET_DIR, item)
        if os.path.exists(item_path):
            if os.path.isdir(item_path):
                shutil.copytree(item_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(item_path, target_path)
            print(f"[pre-build] Copied {item_path} to {target_path}")

    # Copy the platform-specific folder into the target directory
    target_subdir = os.path.join(TARGET_DIR, os.path.basename(source_dir))
    shutil.copytree(source_dir, target_subdir, dirs_exist_ok=True)
    print(f"[pre-build] Copied {source_dir} to {target_subdir}")

if __name__ == "__main__":
    copy_package()
