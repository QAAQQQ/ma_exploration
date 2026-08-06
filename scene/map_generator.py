import numpy as np
import random
import matplotlib.pyplot as plt
from enum import IntEnum


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    STAIR = 2
    PLATFORM = 3


class SemanticMap:
    def __init__(self, grid):
        self.grid = grid
        self.height, self.width = grid.shape

    @property
    def empty_cells(self):
        y, x = np.where(self.grid == CellType.EMPTY)
        return list(zip(y, x))


class MapGenerator:

    def __init__(
        self,
        width=40,
        height=40,
        seed=None
    ):
        self.width = width
        self.height = height

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)


    def generate(self, map_type="maze"):

        if map_type == "random":
            grid = self.random_map()

        elif map_type == "maze":
            grid = self.maze()

        elif map_type == "bsp":
            grid = self.bsp()

        else:
            raise ValueError(
                f"Unknown map type {map_type}"
            )

        return SemanticMap(grid)


    # -----------------------
    # Random obstacle map
    # -----------------------

    def random_map(self):

        grid = np.zeros(
            (self.height,self.width),
            dtype=np.int32
        )

        # border
        grid[0,:] = CellType.WALL
        grid[-1,:] = CellType.WALL
        grid[:,0] = CellType.WALL
        grid[:,-1] = CellType.WALL


        for y in range(1,self.height-1):
            for x in range(1,self.width-1):

                if random.random()<0.02:
                    grid[y,x]=CellType.WALL


        return self.ensure_connectivity(grid) # 这个可能有问题，因为random很容易全都不connect就会有很多黑色


    # -----------------------
    # Maze
    # -----------------------

    def maze(self):

        grid=np.ones(
            (self.height,self.width),
            dtype=np.int32
        )


        def carve(y,x):

            dirs=[
                (0,2),
                (0,-2),
                (2,0),
                (-2,0)
            ]

            random.shuffle(dirs)


            for dy,dx in dirs:

                ny=y+dy
                nx=x+dx

                if (
                    1<=ny<self.height-1
                    and
                    1<=nx<self.width-1
                    and
                    grid[ny,nx]==CellType.WALL
                ):

                    grid[y+dy//2,
                         x+dx//2]=CellType.EMPTY

                    grid[ny,nx]=CellType.EMPTY

                    carve(ny,nx)


        grid[1,1]=CellType.EMPTY

        carve(1,1)


        return grid



    # -----------------------
    # BSP
    # -----------------------

    def bsp(self):

        grid=np.ones(
            (self.height,self.width),
            dtype=np.int32
        )


        def split(
            y1,x1,
            y2,x2,
            depth=0
        ):

            h=y2-y1
            w=x2-x1


            if (
                h<8
                or w<8
                or depth>5
            ):

                grid[
                    y1+1:y2-1,
                    x1+1:x2-1
                ] = CellType.EMPTY

                return (
                    (y1+y2)//2,
                    (x1+x2)//2
                )


            horizontal = (
                h>w
            )


            if h==w:
                horizontal=random.random()<0.5


            if horizontal:

                cut=random.randint(
                    y1+3,
                    y2-3
                )


                c1=split(
                    y1,x1,
                    cut,x2,
                    depth+1
                )

                c2=split(
                    cut,x1,
                    y2,x2,
                    depth+1
                )


                x=random.randint(
                    min(c1[1],c2[1]),
                    max(c1[1],c2[1])
                )


                grid[
                    min(c1[0],c2[0]):
                    max(c1[0],c2[0])+1,
                    x
                ]=CellType.EMPTY


            else:

                cut=random.randint(
                    x1+3,
                    x2-3
                )


                c1=split(
                    y1,x1,
                    y2,cut,
                    depth+1
                )

                c2=split(
                    y1,cut,
                    y2,x2,
                    depth+1
                )


                y=random.randint(
                    min(c1[0],c2[0]),
                    max(c1[0],c2[0])
                )


                grid[
                    y,
                    min(c1[1],c2[1]):
                    max(c1[1],c2[1])+1
                ]=CellType.EMPTY


            return (
                (c1[0]+c2[0])//2,
                (c1[1]+c2[1])//2
            )


        split(
            0,0,
            self.height,
            self.width
        )


        return grid



    # -----------------------
    # Connectivity check
    # -----------------------

    def ensure_connectivity(self,grid):

        # 找第一个空点
        empty=np.argwhere(
            grid==CellType.EMPTY
        )

        if len(empty)==0:
            return grid


        start=tuple(empty[0])


        visited=set([start])

        stack=[start]


        while stack:

            y,x=stack.pop()

            for dy,dx in [
                (1,0),
                (-1,0),
                (0,1),
                (0,-1)
            ]:

                ny=y+dy
                nx=x+dx

                if (
                    0<=ny<self.height
                    and
                    0<=nx<self.width
                    and
                    grid[ny,nx]==CellType.EMPTY
                    and
                    (ny,nx) not in visited
                ):
                    visited.add((ny,nx))
                    stack.append((ny,nx))


        # 不连通区域变墙
        for y,x in empty:

            if (y,x) not in visited:
                grid[y,x]=CellType.WALL


        return grid



    def visualise(self,semantic_map,title="map"):

        plt.figure(figsize=(6,6))

        plt.imshow(
            semantic_map.grid,
            cmap="gray_r"
        )

        plt.title(title)

        plt.axis("off")

        plt.show()



if __name__=="__main__":


    generator=MapGenerator(
        width=50,
        height=50,
        seed=10
    )


    for t in [
        "random",
        "maze",
        "bsp"
    ]:

        m=generator.generate(t)

        visualise(
            m,
            t
        )