import json
import os
import shutil
import signal
import subprocess
import sys
import traceback
from typing import Any, Dict

from depthai_viewer import version as depthai_viewer_version  # type: ignore[attr-defined]

script_path = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(script_path, "venv-" + depthai_viewer_version())
venv_python = (
    os.path.join(venv_dir, "Scripts", "python") if sys.platform == "win32" else os.path.join(venv_dir, "bin", "python")
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
    """Gets site packages dir of the virtual environment. Throws an exception if site packages could not be fetched."""
    return subprocess.run(
        [venv_python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'], end='')"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def download_blobs() -> None:
    import blobconverter
    from depthai_sdk.components.nn_helper import getSupportedModels

    models = [
        "yolov8n_coco_640x352",
        "mobilenet-ssd",
        "face-detection-retail-0004",
        "age-gender-recognition-retail-0013",
    ]
    sdk_models = getSupportedModels(printModels=False)
    for model in models:
        zoo_type = None
        if model in sdk_models:
            model_config_file = sdk_models[model] / "config.json"
            config = json.load(open(model_config_file))
            if "model" in config:
                model_config: Dict[str, Any] = config["model"]
                if "model_name" in model_config:
                    zoo_type = model_config.get("zoo", "intel")
        blobconverter.from_zoo(model, zoo_type=zoo_type, shaves=6)


def dependencies_installed() -> bool:
    return os.path.exists(venv_dir)


def create_venv_and_install_dependencies() -> None:
    venv_packages_dir = ""
    try:
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        # Create venv if it doesn't exist
        if not dependencies_installed():
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
                    "depthai-sdk==1.13.1.dev0+b0340e0c4ad869711d7d5fff48e41c46fe41f475" "--extra-index-url",
                    "https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/",
                    # "git+https://github.com/luxonis/depthai@develop#subdirectory=depthai_sdk",
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

        env = os.environ.copy()
        env["PYTHONPATH"] = venv_packages_dir
        # Download blobs
        subprocess.run(
            [sys.executable, "-c", "from depthai_viewer.install_requirements import download_blobs; download_blobs()"],
            check=True,
            env=env,
        )

        # Restore original SIGINT handler
        signal.signal(signal.SIGINT, original_sigint_handler)

    except Exception as e:
        print(f"Error occurred during dependency installation: {e}")
        print(traceback.format_exc())
        delete_partially_created_venv(venv_dir)
    finally:
        status_dump = json.dumps(
            {
                "venv_site_packages": venv_packages_dir,
            }
        )
        if dependencies_installed():
            print(f"Status Dump: {status_dump}")


if __name__ == "__main__":
    create_venv_and_install_dependencies()
