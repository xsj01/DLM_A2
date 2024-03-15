"""Sweeping task."""

import numpy as np
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils

import copy

object_dist_range = 0.2

def sample_zone_pose(zone_size, bounds, margin=.025, gap = 0.15):
    r = np.linalg.norm(np.array(zone_size[:2]))/2
    margin = r+margin
    if np.random.rand() < 0.5: sign=1.
    else: sign=-1.
    bounds = copy.deepcopy(bounds)
    bounds[0] = [bounds[0][0]+margin, bounds[0][1]-margin]

    

    if np.random.rand() < 0.5:
        bounds[1] = [object_dist_range/2+r+gap, bounds[1][1]-margin]
    else:
        bounds[1] = [bounds[1][0]+margin, -object_dist_range/2-r-gap]

    rx = bounds[0][0] + np.random.rand() * (bounds[0][1] - bounds[0][0])
    ry = bounds[1][0] + np.random.rand() * (bounds[1][1] - bounds[1][0])
    theta = np.random.rand() * 2 * np.pi

    pos = (rx, ry, 0.)
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
    return pos, rot

def check_collision(pose, exist_pose, dist):
    for pose1 in exist_pose:
        rx, ry, _ = pose[0]
        rx1, ry1, _ = pose1[0]

        dd = np.sqrt((rx-rx1)**2+(ry-ry1)**2)
        if dd < dist:
            return True
    return False

class SweepingPiles(Task):
    """Sweeping task."""

    def __init__(self):
        super().__init__()
        self.ee = Spatula
        self.max_steps = 30#000#30
        # self.primitive = primitives.push
        self.primitive = primitives.move_to
        self.lang_template = "push the pile of blocks into the green square"
        self.task_completed_desc = "done sweeping."

    def reset(self, env, object_poses=None):
        def case_obstacle(args=None,  n_object=50, n_obstacle=1, obstacle_in_way=True):
            # if 'goal_case' in env.cfg and env.cfg.goal_case in ['spread', 'split']:
            #     obstacle_in_way = False
            obstacle_size = (0.1, 0.1, 0)
            cx = (self.bounds[0, 0] + self.bounds[0, 1])/2 
            cy = (self.bounds[1, 0] + self.bounds[1, 1])/2 
            
            if obstacle_in_way:
                pos = (np.array([zone_pose[0][0], zone_pose[0][1]])+np.array([cx, cy]))/2. + np.random.rand(2) *0.05
            else:
                rx = (self.bounds[0, 0] + self.bounds[0, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                ry = (self.bounds[1, 0] + self.bounds[1, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                pos = [rx, ry]
            obstacle_pose = (pos[0], pos[1], 0.), (0,0,0,1.)
            
            # obstacle_pose = self.get_random_pose(env, obstacle_size)
            env.add_object('zone/obstacle.urdf', obstacle_pose, 'fixed')

            while len(obj_ids)<n_object:
                # rx = self.bounds[0, 0] + 0.05 + np.random.rand() * 0.4
                # ry = self.bounds[1, 0] + 0.3 + np.random.rand() * 0.4
                rx = (self.bounds[0, 0] + self.bounds[0, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                ry = (self.bounds[1, 0] + self.bounds[1, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                # if not env.cfg.get("obstacle"):
                xyz = (rx, ry, 0.01)
                if np.sqrt((rx-obstacle_pose[0][0])**2+(ry-obstacle_pose[0][1])**2) < obstacle_size[0]/2+0.01+0.03:
                    continue
                # else:
                #     xyz = (rx, ry, 0.05)
                theta = np.random.rand() * 2 * np.pi
                xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
                obj_id = env.add_object('block/small.urdf', (xyz, xyzw))
                obj_pts[obj_id] = self.get_box_object_points(obj_id)
                obj_ids.append((obj_id, (0, None)))
            # if env.cfg.get("obstacle"):
            #     env.step_simulation()
            #     while not env.is_static:
            #         env.step_simulation()

            # Goal: all small blocks must be in zone.
            # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
            # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
            # self.goals.append((goal, metric))
            self.goals.append((obj_ids, np.ones((50, 1)), [zone_pose], True, False,
                            'zone', (obj_pts, [(zone_pose, zone_size)]), 1))
            self.lang_goals.append(self.lang_template)

        def case_big_object(args=False):
            if args is False:
                num=1
            else:
                num = args
            pose_list = []
            while len(pose_list)<num:
                rx = (self.bounds[0, 0] + self.bounds[0, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                ry = (self.bounds[1, 0] + self.bounds[1, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                xyz = (rx, ry, 0.01)
                theta = np.random.rand() * 2 * np.pi
                xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
                if not check_collision([xyz], pose_list, 0.07):
                    pose_list.append([xyz])
                else:
                    continue
                obj_id = env.add_object('block/large.urdf', (xyz, xyzw))
                obj_pts[obj_id] = self.get_mesh_object_points(obj_id)
                obj_ids.append((obj_id, (0, None)))

            # Goal: all small blocks must be in zone.
            # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
            # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
            # self.goals.append((goal, metric))
            self.goals.append((obj_ids, np.ones((1, 1)), [zone_pose], True, False,
                            'zone', (obj_pts, [(zone_pose, zone_size)]), 1))
            self.lang_goals.append(self.lang_template)

        def case_toy(args=False):
            if args is False:
                num=1
            else:
                num = args
            pose_list = []
            while len(pose_list)<num:
                rx = (self.bounds[0, 0] + self.bounds[0, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                ry = (self.bounds[1, 0] + self.bounds[1, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                xyz = (rx, ry, 0.01)
                theta = np.random.rand() * 2 * np.pi
                xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
                if not check_collision([xyz], pose_list, 0.07):
                    pose_list.append([xyz])
                else:
                    continue
                obj_id = env.add_object('block/disk.urdf', (xyz, xyzw))
                obj_pts[obj_id] = self.get_mesh_object_points(obj_id)
                obj_ids.append((obj_id, (0, None)))

            # Goal: all small blocks must be in zone.
            # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
            # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
            # self.goals.append((goal, metric))
            self.goals.append((obj_ids, np.ones((1, 1)), [zone_pose], True, False,
                            'zone', (obj_pts, [(zone_pose, zone_size)]), 1))
            self.lang_goals.append(self.lang_template)
        
        def case_tiny_object(args=False):
            if args is False:
                num=200
            else:
                num = args
            for _ in range(num):
                # rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
                # ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
                rx = (self.bounds[0, 0] + self.bounds[0, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                ry = (self.bounds[1, 0] + self.bounds[1, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                xyz = (rx, ry, 0.01)
                theta = np.random.rand() * 2 * np.pi
                xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
                obj_id = env.add_object('block/tiny.urdf', (xyz, xyzw))
                obj_pts[obj_id] = self.get_mesh_object_points(obj_id)
                obj_ids.append((obj_id, (0, None)))
            # print(len(obj_pts))
            env.step_simulation()
            while not env.is_static:
                env.step_simulation()
            # Goal: all small blocks must be in zone.
            # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
            # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
            # self.goals.append((goal, metric))
            self.goals.append((obj_ids, np.ones((num, 1)), [zone_pose], True, False,
                            'zone', (obj_pts, [(zone_pose, zone_size)]), 1))
            self.lang_goals.append(self.lang_template)
        
        def case_tiny_block(args=False):
            if args is False:
                num=200
            else:
                num = args
            for _ in range(num):
                # rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
                # ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
                rx = (self.bounds[0, 0] + self.bounds[0, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                ry = (self.bounds[1, 0] + self.bounds[1, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                xyz = (rx, ry, 0.01)
                theta = np.random.rand() * 2 * np.pi
                xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
                obj_id = env.add_object('block/tinyblock.urdf', (xyz, xyzw))
                obj_pts[obj_id] = self.get_box_object_points(obj_id)
                obj_ids.append((obj_id, (0, None)))
            # print(len(obj_pts))
            env.step_simulation()
            while not env.is_static:
                env.step_simulation()
            # Goal: all small blocks must be in zone.
            # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
            # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
            # self.goals.append((goal, metric))
            self.goals.append((obj_ids, np.ones((num, 1)), [zone_pose], True, False,
                            'zone', (obj_pts, [(zone_pose, zone_size)]), 1))
            self.lang_goals.append(self.lang_template)

        def case_default(args=False):
            if args is False:
                num=50
            else:
                num = args
            for _ in range(num):
                # rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
                # ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
                rx = (self.bounds[0, 0] + self.bounds[0, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                ry = (self.bounds[1, 0] + self.bounds[1, 1])/2 + (np.random.rand()-0.5) * object_dist_range
                xyz = (rx, ry, 0.01)
                theta = np.random.rand() * 2 * np.pi
                xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
                obj_id = env.add_object('block/small.urdf', (xyz, xyzw))
                obj_pts[obj_id] = self.get_box_object_points(obj_id)
                obj_ids.append((obj_id, (0, None)))

            # Goal: all small blocks must be in zone.
            # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
            # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
            # self.goals.append((goal, metric))
            self.goals.append((obj_ids, np.ones((num, 1)), [zone_pose], True, False,
                            'zone', (obj_pts, [(zone_pose, zone_size)]), 1))
            self.lang_goals.append(self.lang_template)
        
        def case_fixed_pose():
            num = object_poses.shape[0]
            for i in range(num):
                xyz = tuple(object_poses[i, -3:].tolist())
                xyzw = tuple(object_poses[i, :4].tolist())
                # print(xyz, xyzw)
                obj_id = env.add_object('block/small.urdf', (xyz, xyzw))
                obj_pts[obj_id] = self.get_box_object_points(obj_id)
                obj_ids.append((obj_id, (0, None)))
            
            self.goals.append((obj_ids, np.ones((num, 1)), [zone_pose], True, False,
                            'zone', (obj_pts, [(zone_pose, zone_size)]), 1))
            self.lang_goals.append(self.lang_template)

        def case_split(args=False):
            # print(zone_pose)
            
            if args is False:
                target_num=2
            else:
                target_num = args
            zone_size = (0.12, 0.12, 0)
            # zone_pose = self.get_random_pose(env, zone_size)
            zone_pose = sample_zone_pose(zone_size, self.bounds)
            if not env.cfg.get('data_gen'):
                env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            
            zone_pose_list = [zone_pose]
            count=0
            while len(zone_pose_list)<target_num:
                if count>100:
                    break
                zone_pose = sample_zone_pose(zone_size, self.bounds)
                if not check_collision(zone_pose, zone_pose_list, np.sqrt(2)*zone_size[0]+0.06):
                    
                    zone_pose_list.append(zone_pose)
                    env.add_object('zone/zone.urdf', zone_pose, 'fixed')
                count+=1
           
            self.zone = (zone_pose_list, zone_size)

            return zone_size, zone_pose

        def case_spread(args=False):
            zone_size = (0.12, 0.12, 0)
            # zone_pose = self.get_random_pose(env, zone_size)
            zone_pose = sample_zone_pose(zone_size, self.bounds)
            self.zone = (zone_pose, zone_size)
            return zone_size, zone_pose
        
        def case_ring(args=False):
            zone_size = (0.12, 0.12, 0)
            # zone_pose = self.get_random_pose(env, zone_size)
            zone_pose = sample_zone_pose(zone_size, self.bounds)
            
            env.add_object('zone/zone_ring.urdf', zone_pose, 'fixed')
            return zone_size, zone_pose

        super().reset(env)

        if "transporter" in env.cfg and env.cfg["transporter"]:
            self.max_steps = 20
            self.primitive = primitives.push
        else:
            self.max_steps = 30#000#30
            # self.primitive = primitives.push
            self.primitive = primitives.move_to

        if 'goal_case' not in env.cfg or not env.cfg.goal_case:
            # Add goal zone.
            zone_size = (0.12, 0.12, 0)
            # zone_pose = self.get_random_pose(env, zone_size)
            zone_pose = sample_zone_pose(zone_size, self.bounds)
            if not env.cfg.get('data_gen'):
                env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            self.zone = (zone_pose, zone_size)
        else:
            func = eval("case_"+env.cfg.goal_case)

            zone_size, zone_pose = func(env.cfg.get("goal_case_args"))
            self.zone = (zone_pose, zone_size)

        # Add pile of small blocks.
        obj_pts = {}
        obj_ids = []
        if object_poses is not None:
            case_fixed_pose()
        elif env.cfg.get('object_case'):
            func = eval("case_"+env.cfg.object_case)
            func(env.cfg.get("object_case_args"))
        else:
            case_default(env.cfg.get("object_case_args"))
        
        
