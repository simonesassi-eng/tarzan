"""Export layer: HTML newsletter generation (Tarzan's primary artifact)."""

from tarzan.export.newsletter import (
    build_context,
    generate_newsletter,
    render_newsletter,
)

__all__ = [
    "build_context",
    "generate_newsletter",
    "render_newsletter",
]
