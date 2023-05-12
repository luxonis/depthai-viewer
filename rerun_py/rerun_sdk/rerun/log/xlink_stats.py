from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt

from rerun import bindings
from rerun.components.xlink_stats import XLinkStats
from rerun.log.log_decorator import log_decorator


@log_decorator
def log_xlink_stats(
    total_bytes_written: int, total_bytes_read: int
) -> None:
    """
    Log an XLink throughput statistic.

    Parameters
    ----------
    total_bytes_written:
        Total bytes written by the device to the XLink.
    total_bytes_read:
        Total bytes read by the device from the XLink.
    """
    instanced: Dict[str, Any] = {}
    instanced["rerun.xlink_stats"] = XLinkStats.create(total_bytes_written, total_bytes_read)  # type: ignore[arg-type]
    bindings.log_arrow_msg("xlink_stats", components=instanced, timeless=False)
