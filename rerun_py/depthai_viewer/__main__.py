"""See `python3 -m depthai-viewer --help`."""

import os
import sys

from depthai_viewer import (
    bindings,
    unregister_shutdown,
)
from depthai_viewer import version as depthai_viewer_version  # type: ignore[attr-defined]
from depthai_viewer.install_requirements import get_site_packages

script_path = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(script_path, "venv-" + depthai_viewer_version())


def main() -> None:
    python_exe = sys.executable
    # Call the bindings.main using the Python executable in the venv
    unregister_shutdown()
    # The viewer will take care of installing the requirements if site_packages_directory is None
    site_packages_directory = None
    try:
        site_packages_directory = get_site_packages()
    except Exception:
        pass
    sys.exit(bindings.main(sys.argv, python_exe, site_packages_directory))


if __name__ == "__main__":
    main()
