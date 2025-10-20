from __future__ import annotations
import math
import typing
from dataclasses import dataclass

from fibsem.milling.patterning import FiducialPattern
from fibsem.structures import FibsemRectangleSettings


@dataclass
class AsymmetricFiducialPattern(FiducialPattern):
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
