"""
Experimental features for Rerun.

These features are not yet stable and may change in future releases without
going through the normal deprecation cycle.
"""

from depthai_viewer.log.experimental.blueprint import add_space_view, new_blueprint, set_auto_space_views, set_panels
from depthai_viewer.log.experimental.text import log_text_box

__all__ = ["log_text_box", "add_space_view", "set_panels", "set_auto_space_views", "new_blueprint"]
