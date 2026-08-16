import math
import mujoco
import mujoco.viewer


class MujocoBuilder:

    def __init__(self):

        self.spec = mujoco.MjSpec()

        # MuJoCo 3.10.0
        self.world = self.spec.worldbody

    # ==================================================
    # build
    # ==================================================

    def build(self, scene, robots):
        # 每次 build 都从干净的 MuJoCo scene 开始
        self.spec = mujoco.MjSpec()
        self.world = self.spec.worldbody

        # floor
        for floor in scene.floors:
            self.add_box(floor)

        # walls
        for wall in scene.walls:
            self.add_wall(wall)

        # stairs
        for stair in scene.stairs:
            self.add_stair(stair)

        # robot placeholder
        for robot in robots:
            self.add_robot(robot)

        model = self.spec.compile()

        return model

    # ==================================================
    # WallSegment -> MuJoCo box
    # ==================================================

    def add_wall(self, wall):
        body = self.world.add_body(name=wall.name)
        cx, cy = wall.center
        body.pos = [cx,cy,wall.height / 2]

        geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[wall.length / 2, wall.thickness / 2,wall.height / 2],
            rgba=[0.65,0.65,0.65,1])

        # rotation around z
        geom.quat = self.angle_to_quat(wall.angle)

    # ==================================================
    # BoxObject
    # ==================================================
    def add_box(self, obj):
        body = self.world.add_body(name=obj.name)
        body.pos = obj.position
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=obj.size,
            rgba=self.category_color(obj.category))

    # ==================================================
    # StairObject
    # ==================================================

    def add_stair(self, stair):
        body = self.world.add_body(name=stair.name)
        body.pos = stair.position
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=stair.size,
            rgba=[0.5,0.5,0.5,1])

    # ==================================================
    # Robots+Lidar
    # ==================================================

    def add_robot(self,robot):
        body = self.world.add_body(name=robot.name)
        body.pos = robot.position

        if robot.model_type == "box":
            if robot.size is None:
                raise ValueError("Box robot requires size")
            
            # Planar motion joints
            body.add_joint(
                name=f"{robot.name}_x_joint",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=(1.0, 0.0, 0.0),
                damping=1.0,
            )

            body.add_joint(
                name=f"{robot.name}_y_joint",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=(0.0, 1.0, 0.0),
                damping=1.0,
            )

            body.add_joint(
                name=f"{robot.name}_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=(0.0, 0.0, 1.0),
                damping=1.0,
            )

            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(robot.size[0] / 2,robot.size[1] / 2,robot.size[2] / 2),
                rgba=robot.color)
        
        elif robot.model_type == "xml":
            if robot.xml_path is None:
                raise ValueError("XML robot requires path")
            raise ValueError("XML robot not implmented")           
            
        else:
            raise ValueError(f"Unknown robot type {robot.model_type}")
        
        body.add_site(
            name=robot.lidar_site_name,
            pos=robot.lidar_pos,
            type=mujoco.mjtGeom.mjGEOM_SPHERE, # 后三行可以不写，这样site就不可见
            size=(0.03,),
            rgba=(1,0,0,1))

        body.add_site( #加个指向标
            name="marker_"+robot.name,
            pos=(0.0,0.0,2),
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(0.05,0.5),
            rgba=(0.0,1.0,0.0,0.6))
        
        body.add_site( #方向标
            name="heading_" + robot.name,
            pos=(0.15, 0.0, 2.5),      # 向局部 +x 偏一点，并放在顶部
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.04, 0.02, 0.02),   # 很小
            rgba=(0.0, 1.0, 0.0, 0.8),
        )

    # ==================================================
    # helpers 
    # ==================================================

    def angle_to_quat(self, angle):

        half = angle / 2
        return [math.cos(half),0,0,math.sin(half)]

    def category_color(self, category):

        if category == "floor":
            return [0.8,0.8,0.8,1]
        return [0.4,0.4,0.8,1]
    
    def save_xml(self, filename):
        self.spec.to_file(filename)

