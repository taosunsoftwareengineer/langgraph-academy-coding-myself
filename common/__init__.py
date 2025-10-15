"""Common utilities package.

This file makes the `common` folder a Python package. It re-exports
useful symbols (like `display_image`) so other modules can import them
as `from common.image_display import display_image` or `from common import display_image`.
"""
from .image_display import display_image

__all__ = ["display_image"]
