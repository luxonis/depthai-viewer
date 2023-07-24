import os
import shutil
import signal
import subprocess
import sys
import traceback
import json
import struct

from depthai_viewer import bindings, unregister_shutdown
from depthai_viewer import version as depthai_viewer_version  # type: ignore[attr-defined]

script_path = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(script_path, "venv-" + depthai_viewer_version())
venv_python = (
    os.path.join(venv_dir, "Scripts", "python")
    if sys.platform == "win32"
    else os.path.join(venv_dir, "bin", "python")
)

def delete_partially_created_venv(path: str) -> None:
    try:
        if os.path.exists(path):
            print(f"Deleting partially created virtual environment: {path}")
            shutil.rmtree(path)
    except Exception as e:
        print(f"Error occurred while attempting to delete the virtual environment: {e}")
        print(traceback.format_exc())


def sigint_mid_venv_install_handler(signum, frame) -> None:  # type: ignore[no-untyped-def]
    delete_partially_created_venv(venv_dir)


def get_site_packages() -> str:
    """Get the site packages directory of the virtual environment. Throws an exception if the site packages could not be fetched."""
    return subprocess.run(
            [venv_python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'], end='')"],
            capture_output=True,
            text=True,
            check=True,
    ).stdout.strip()

def dependencies_installed() -> bool:
    return os.path.exists(venv_dir)

def create_venv_and_install_dependencies() -> str:

    venv_packages_dir = ""
    try:
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        # Create venv if it doesn't exist
        if not os.path.exists(venv_dir):
            # In case of Ctrl+C during the venv creation, delete the partially created venv
            signal.signal(signal.SIGINT, sigint_mid_venv_install_handler)
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

            # Install dependencies
            subprocess.run([venv_python, "-m", "pip", "install", "-U", "pip"], check=True)
            # Install depthai_sdk first, then override depthai version with the one from requirements.txt
            subprocess.run(
                [
                    venv_python,
                    "-m",
                    "pip",
                    "install",
                    "depthai-sdk==1.11.0"
                    # "git+https://github.com/luxonis/depthai@refactor_xout#subdirectory=depthai_sdk",
                ],
                check=True,
            )
            subprocess.run(
                [venv_python, "-m", "pip", "install", "-r", f"{script_path}/requirements.txt"],
                check=True,
            )
        venv_packages_dir = get_site_packages()

        # Delete old requirements
        for item in os.listdir(os.path.join(venv_dir, "..")):
            if not item.startswith("venv-"):
                continue
            if item == os.path.basename(venv_dir):
                continue
            print(f"Removing old venv: {item}")
            shutil.rmtree(os.path.join(venv_dir, "..", item))

        # Restore original SIGINT handler
        signal.signal(signal.SIGINT, original_sigint_handler)

    except Exception as e:
        print(f"Error occurred during dependency installation: {e}")
        print(traceback.format_exc())
        delete_partially_created_venv(venv_dir)
    finally:
        status_dump = json.dumps({
            "venv_site_packages": venv_packages_dir,
        })
        if dependencies_installed():
            print(f"Status Dump: {status_dump}")

if __name__ == '__main__':
    create_venv_and_install_dependencies()
