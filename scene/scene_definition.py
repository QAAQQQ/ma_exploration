from dataclasses import dataclass, field
from typing import Tuple, List
import math


@dataclass
class WallSegment:
    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]

    thickness: float = 0.2
    height: float = 2.0

    material: str = "concrete"

    # movable: bool = False
    # mass: float = 0.0

    @property
    def length(self):
        return math.hypot(
            self.end[0]-self.start[0],
            self.end[1]-self.start[1])

    @property
    def center(self):
        return (
            (self.start[0]+self.end[0])/2,
            (self.start[1]+self.end[1])/2)

    @property
    def angle(self):
        return math.atan2(
            self.end[1]-self.start[1],
            self.end[0]-self.start[0]
        )


@dataclass
class BoxObject:
    name: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    category: str


@dataclass
class StairObject:
    name: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]


@dataclass
class Scene:

    walls: List[WallSegment] = field(default_factory=list) 
    floors: List[BoxObject] = field(default_factory=list)
    stairs: List[StairObject] = field(default_factory=list)
    platforms: List[BoxObject] = field(default_factory=list) #backup place


