import os
import platform
import shutil
from pathlib import Path
import logging
import sys
# Configure logging
logging.basicConfig(
    filename="installer.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Directories in the GitHub Actions environment
SCRIPT_DIR = Path(os.getenv("GITHUB_WORKSPACE", ".")).resolve()
COMPILERS_DIR = SCRIPT_DIR / "packages/compilers"
DESTINATION_DIR = SCRIPT_DIR / "depthai_viewer/_backend/obscured_utilities/utilities"  # Output directory for processed files

# Ensure destination directory exists
DESTINATION_DIR.mkdir(parents=True, exist_ok=True)


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
        logging.error(f"Error copying folder to runtime: {e}")
        raise


def main():
    try:
        python_version = get_python_version()
        print(f"Python version: {python_version}")

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
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
