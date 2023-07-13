import os
import shutil
import signal
import subprocess
import sys
import traceback

from depthai_viewer import bindings, unregister_shutdown
from depthai_viewer import version as depthai_viewer_version  # type: ignore[attr-defined]

script_path = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(script_path, "venv-" + depthai_viewer_version())


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


def get_site_packages(python_executable_path: str) -> str:
    return subprocess.run(
            [python_executable_path, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'], end='')"],
            capture_output=True,
            text=True,
            check=True,
    ).stdout.strip()


def create_venv_and_install_dependencies() -> str:
    py_executable = (
        os.path.join(venv_dir, "Scripts", "python")
        if sys.platform == "win32"
        else os.path.join(venv_dir, "bin", "python")
    )
    try:
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        # Create venv if it doesn't exist
        if not os.path.exists(venv_dir):
            # In case of Ctrl+C during the venv creation, delete the partially created venv
            signal.signal(signal.SIGINT, sigint_mid_venv_install_handler)
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

            # Install dependencies
            subprocess.run([py_executable, "-m", "pip", "install", "-U", "pip"], check=True)
            # Install depthai_sdk first, then override depthai version with the one from requirements.txt
            subprocess.run(
                [
                    py_executable,
                    "-m",
                    "pip",
                    "install",
                    "depthai-sdk==1.11.0"
                    # "git+https://github.com/luxonis/depthai@refactor_xout#subdirectory=depthai_sdk",
                ],
                check=True,
            )
            subprocess.run(
                [py_executable, "-m", "pip", "install", "-r", f"{script_path}/requirements.txt"],
                check=True,
            )
        venv_packages_dir = get_site_packages(py_executable)

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
        return os.path.normpath(venv_packages_dir)

    except Exception as e:
        print(f"Error occurred during the creation of the virtual environment or installation of dependencies: {e}")
        print(traceback.format_exc())
        delete_partially_created_venv(venv_dir)
        exit(1)



if __name__ == '__main__':
    venv_site_packages = create_venv_and_install_dependencies()
    python_exe = sys.executable
    # Call the bindings.main using the Python executable in the venv
    unregister_shutdown()
    sys.exit(bindings.main(sys.argv, python_exe, venv_site_packages))
