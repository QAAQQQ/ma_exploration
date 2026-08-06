import numpy as np
from scene.scene_definition import Scene, WallSegment, BoxObject, StairObject
from scene.map_generator import CellType
import random

# class LevelConnectionType(Enum): #构建2.5D的基础
#     STAIRS = 0
#     RAMP = 1
#     ELEVATOR = 2

# class MapGrid(Enum):
#     EMPTY = 0
#     WALL = 1
#     AGNET = 2


class MapToScene:

    def __init__(self, cell_size=0.5, wall_height=2.0, wall_thickness=0.2):
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness  

    def convert(self, semantic_map):
        scene = Scene()
        self.create_floor(semantic_map, scene)
        self.extract_walls(semantic_map, scene)
        self.extract_stairs(semantic_map, scene)
        return scene

    def sample_spawn_position(self,semantic_map,n_agents=1,robot_z=0.25,seed=None):
        rng = random.Random(seed)

        empty_cells = semantic_map.empty_cells

        if n_agents > len(empty_cells):
            raise ValueError(
                f"Not enough empty cells for {n_agents} agents. "
                f"Only {len(empty_cells)} empty cells available."
            )

        spawn_cells = rng.sample(empty_cells, n_agents)

        positions = []

        for row, col in spawn_cells:
            world_x = (col + 0.5) * self.cell_size
            world_y = (row + 0.5) * self.cell_size

            positions.append((
                world_x,
                world_y,
                robot_z,
            ))

        return positions
        
    def create_floor(self, semantic_map, scene):
        h, w = semantic_map.grid.shape
        scene.floors.append(BoxObject(
            name="floor",
            position=(w*self.cell_size/2, h*self.cell_size/2, -0.05),
            size=(w*self.cell_size/2, h*self.cell_size/2, 0.05),
            category="floor"))
    

    def extract_stairs(self, semantic_map, scene):
        cells = np.argwhere(semantic_map.grid == CellType.STAIR)
        for i, (y, x) in enumerate(cells):
            scene.stairs.append(StairObject(
                name=f"stair_{i}",
                position=(x*self.cell_size, y*self.cell_size, 0.1),
                size=(self.cell_size/2, self.cell_size/2, 0.1)))

    
    def extract_walls(self, semantic_map, scene):
        grid = semantic_map.grid
        mask = (grid == CellType.WALL).copy()
        wall_id = 0
        while np.any(mask):
            rect = self._find_max_rectangle(mask)
            if rect is None:
                break
            x, y, w, h = rect
            segment = self._rectangle_to_wallsegment(x,y,w,h)
            scene.walls.append(
                WallSegment(
                    name=f"wall_{wall_id}",
                    start=segment["start"],
                    end=segment["end"],
                    thickness=segment["thickness"],
                    height=self.wall_height
                )
            )
            wall_id += 1
            
            # remove rectangle
            mask[
                y:y+h,
                x:x+w
            ] = False
        
    def _find_max_rectangle(self, mask):
        h, w = mask.shape

        # 1. 找左上角第一个wall
        start = None
        for y in range(h):
            for x in range(w):
                if mask[y,x]:
                    start = (x,y)
                    break
            if start is not None:
                break
        if start is None:
            return None
        x0, y0 = start

        # 2. 向右扩展最大width
        width = 0
        while (
            x0 + width < w
            and mask[y0, x0+width]
        ):
            width += 1

        # 3. 向下扩展
        height = 1
        while y0 + height < h:
            valid = True
            for x in range(x0,x0 + width):
                if not mask[y0+height,x]:
                    valid = False
                    break
            if not valid:
                break
            height += 1

        return (x0,y0,width,height)

    def _rectangle_to_wallsegment(self,x,y,w,h):
        x0 = x * self.cell_size
        y0 = y * self.cell_size
        width = w * self.cell_size
        height = h * self.cell_size

        # horizontal wall
        if width >= height:
            cy = y0 + height / 2
            start = (x0,cy)
            end = (x0 + width,cy)
            thickness = height

        # vertical wall
        else:
            cx = x0 + width / 2
            start = (cx,y0)
            end = (cx,y0 + height)
            thickness = width

        return {"start": start,"end": end,"thickness": thickness}
