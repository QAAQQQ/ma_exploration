# 用于生成map+机器人+lidar

import numpy as np
import random
from enum import Enum




class LevelConnectionType(Enum): #构建2.5D的基础
    STAIRS = 0
    RAMP = 1
    ELEVATOR = 2

class MapGrid(Enum):
    EMPTY = 0
    WALL = 1
    AGNET = 2

class MapGenerator：
    def __init__(self, map_size=(20, 20), grid_resolution=0.5):
        self.map_size = map_size
        self.grid_resolution = grid_resolution


