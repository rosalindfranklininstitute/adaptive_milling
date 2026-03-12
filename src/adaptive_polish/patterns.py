from __future__ import annotations
import math
import typing
from dataclasses import dataclass, field

from fibsem.milling.patterning import BasePattern
from fibsem.structures import FibsemRectangleSettings, CrossSectionPattern
from fibsem.milling.properties import (
    DEFAULT_DISTANCE_METADATA,
    DEFAULT_ANGLE_METADATA,
    DEFAULT_CROSS_SECTION_METADATA,
)


@dataclass
class AsymmetricFiducialPattern(BasePattern[FibsemRectangleSettings]):
    width: float = field(
        default=0.1e-6,
        metadata={
            "label": "Width",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the asymmetric fiducial pattern.",
        },
    )
    height: float = field(
        default=10.0e-6,
        metadata={
            "label": "Height",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Width of the asymmetric fiducial pattern.",
        },
    )
    depth: float = field(
        default=0.5e-6,
        metadata={
            "label": "Depth",
            **DEFAULT_DISTANCE_METADATA,
            "tooltip": "Depth of the asymmetric fiducial pattern.",
        },
    )
    rotation: float = field(
        default=45.0,
        metadata={
            **DEFAULT_ANGLE_METADATA,
            "tooltip": "Rotation of the asymmetric fiducial in degrees.",
        },
    )
    cross_section: CrossSectionPattern = field(
        default=CrossSectionPattern.Rectangle,
        metadata=DEFAULT_CROSS_SECTION_METADATA,
    )

    name: typing.ClassVar[str] = "AsymmetricFiducial"

    def define(self) -> list[FibsemRectangleSettings]:
        """Draw a fiducial milling pattern (asymmetrical asterisk shape)"""
        width = self.width
        height = self.height
        depth = self.depth
        rotation = math.radians(self.rotation)
        cross_section = self.cross_section

        vertical_pattern = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation,
        )
        horizontal_pattern = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation + math.pi / 2,  # 90 degrees
        )
        short_offset_pattern = FibsemRectangleSettings(
            width=width,
            height=height * 0.5,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation + 0.64577,  # ~37 degrees
        )
        medium_offest_pattern = FibsemRectangleSettings(
            width=width,
            height=height * 0.75,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation + 2.56563,  # ~ 147 degrees
        )

        self.shapes = [
            vertical_pattern,
            horizontal_pattern,
            short_offset_pattern,
            medium_offest_pattern,
        ]
        return self.shapes
