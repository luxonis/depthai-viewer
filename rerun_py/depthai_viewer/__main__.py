"""See `python3 -m depthai-viewer --help`."""

import os
import sys
from pathlib import Path
import shutil
import platform
from depthai_viewer import (
    bindings,
    unregister_shutdown,
)
from depthai_viewer import version as depthai_viewer_version  # type: ignore[attr-defined]
from depthai_viewer.install_requirements import get_site_packages

script_path = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(script_path, "venv-" + depthai_viewer_version())

def version_dynamic_recalibration(SCRIPT_DIR) -> str:
    def get_python_version():
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    def copy_folder_to_runtime(folder_path, destination_path):
        try:
            if destination_path.exists():
                shutil.rmtree(destination_path)
            shutil.copytree(folder_path, destination_path)
            print(f"Copied folder from {folder_path} to {destination_path}")
        except Exception as e:
            print(f"Error copying folder to runtime: {e}")
            raise
    COMPILERS_DIR = Path(SCRIPT_DIR + "/_backend/compilers")
    DESTINATION_DIR = Path(SCRIPT_DIR + "/_backend/obscured_utilities/utilities")
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)

    python_version = get_python_version()

    system = platform.system().lower()
    folder_mapping = {
        "darwin": "darwin_x86_64",
        "linux": "linux_x86_64",
        "windows": "windows_x86_64",
    }
    folder_to_copy = folder_mapping.get(system)
    if not folder_to_copy:
        raise ValueError(f"Unsupported OS: {system}")

    folder_source = COMPILERS_DIR / python_version / "pyarmor_runtime_007125" / folder_to_copy
    if not folder_source.exists():
        raise FileNotFoundError(f"Folder {folder_source} not found in compilers directory.")

    destination = DESTINATION_DIR / "pyarmor_runtime_007125" / folder_to_copy
    print(f"Copying {folder_to_copy} from {folder_source} to {destination}...")
    copy_folder_to_runtime(folder_source, destination)

    shutil.copy(COMPILERS_DIR / python_version / "pyarmor_runtime_007125" / "__init__.py", DESTINATION_DIR / "pyarmor_runtime_007125"/ "__init__.py")
    additional_files = ["__init__.py", "calibration_handler.py", "display_handler.py"]
    for file_name in additional_files:
        source_file = COMPILERS_DIR / python_version / file_name
        if not source_file.exists():
            raise FileNotFoundError(f"{file_name} not found in {source_file}.")
        shutil.copy(source_file, DESTINATION_DIR)
        print(f"Copied {file_name} to {DESTINATION_DIR}")

    print("Operation completed successfully.")

def main() -> None:
    python_exe = sys.executable
    # Call the bindings.main using the Python executable in the venv
    unregister_shutdown()
    version_dynamic_recalibration(script_path)
    # The viewer will take care of installing the requirements if site_packages_directory is None
    site_packages_directory = None
    try:
        site_packages_directory = get_site_packages()
    except Exception:
        pass
    sys.exit(bindings.main(sys.argv, python_exe, site_packages_directory))


if __name__ == "__main__":
    main()
