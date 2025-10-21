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
        rotation = math.degrees(self.rotation)
        cross_section = self.cross_section

        left_pattern = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation,
        )
        right_pattern = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation + math.degrees(90),
        )
        left_pattern_2 = FibsemRectangleSettings(
            width=width,
            height=height * 0.5,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation + math.degrees(37),
        )
        right_pattern_2 = FibsemRectangleSettings(
            width=width,
            height=height * 0.75,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation + math.degrees(147),
        )

        self.shapes = [left_pattern, right_pattern, left_pattern_2, right_pattern_2]
        return self.shapes
