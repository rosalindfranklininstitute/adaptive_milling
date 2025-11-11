from __future__ import annotations
import math
import typing
from dataclasses import dataclass

from fibsem.milling.patterning import BasePattern
from fibsem.structures import FibsemRectangleSettings, CrossSectionPattern


@dataclass
class AsymmetricFiducialPattern(BasePattern[FibsemRectangleSettings]):
    width: float = 0.1e-6
    height: float = 10.0e-6
    depth: float = 0.5e-6
    rotation: float = 45.0
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle

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
