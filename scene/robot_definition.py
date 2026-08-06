from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class RobotConfig:
    name:str
    position:Tuple[float,float,float]
    color:Tuple[float,float,float,float]

    model_type: str  # "box" or "xml"
    # box robot
    size: Optional[Tuple[float,float,float]] = None
    # xml robot
    xml_path: Optional[str] = None

    lidar_pos:Tuple[float,float,float] = (0.0, 0.0, 0.2)

    @property
    def lidar_site_name(self):
        return f"{self.name}_lidar_site"




