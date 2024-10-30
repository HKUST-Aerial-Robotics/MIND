import copy
import numpy as np
from common.semantic_map import LocalSemanticMap
from planners.mind.planner import MINDPlanner
from common.bbox import PedestrianBBox, CyclistBBox, VehicleBBox, BusBBox, UnknownBBox
from common.kinematics import VehicleParam, kine_propagate
from common.geometry import project_point_on_polyline, remove_close_points
from av2.datasets.motion_forecasting.data_schema import ObjectType


class AgentColor:
    def exo(self):
        return ['lightcoral', 'indianred']  # facecolor, edgecolor

    def ego_disable(self):
        return ['lightskyblue', 'deepskyblue']

    def ego_enable(self):
        return ['lime', 'blue']

    def interest(self):
        return ['yellow', 'orange']

    def get_color(self, clr_name):
        if clr_name == "yellow":
            return ['yellow', 'orange']
        return


class AgentObservation:
    def __init__(self):
        self.id = None
        self.type = None
        self.clr = None
        self.bbox = None
        self.state = None
        self.timestep = None


class PlainAgent:
    def __init__(self):
        self.id = None
        self.type = None
        self.clr = None
        self.state = None
        self.ctrl = None
        self.bbox = None
        self.timestep = None

    def observe(self):
        obs = AgentObservation()
        obs.id = self.id
        obs.type = self.type
        obs.clr = self.clr
        obs.state = self.state
        # noise = np.random.normal(0, 0.05, self.state.shape)
        # noise[-1] = 0.0
        # obs.state = self.state + noise
        obs.bbox = self.bbox
        obs.timestep = self.timestep
        return obs

    def observe_no_noise(self):
        obs = AgentObservation()
        obs.id = self.id
        obs.type = self.type
        obs.clr = self.clr
        obs.bbox = self.bbox
        obs.state = self.state
        obs.timestep = self.timestep

        return obs


class NonReactiveAgent(PlainAgent):
    def __init__(self):
        super(NonReactiveAgent, self).__init__()
        self.traj_info = None
        self.traj_type = None
        self.traj_cat = None
        self.rec_step = 0
        self.max_step = 0
        self.lcl_smp = None

    def init(self, agt_id, traj_type, traj_cat, traj_info, smp, clr):
        self.id = agt_id
        self.clr = clr
        self.traj_type = traj_type
        self.traj_cat = traj_cat
        self.traj_info = traj_info
        self.type = self.traj_type[self.rec_step]
        if self.type == ObjectType.VEHICLE:
            self.bbox = VehicleBBox()
        elif self.type == ObjectType.PEDESTRIAN:
            self.bbox = PedestrianBBox()
        elif self.type == ObjectType.MOTORCYCLIST:
            self.bbox = CyclistBBox()
        elif self.type == ObjectType.CYCLIST:
            self.bbox = CyclistBBox()
        elif self.type == ObjectType.BUS:
            self.bbox = BusBBox()
        elif self.type == ObjectType.UNKNOWN:
            self.bbox = UnknownBBox()
        else:
            self.bbox = UnknownBBox()  # for all static objects

        traj_pos, traj_ang, traj_vel = self.traj_info[:3]
        self.state = np.array([traj_pos[self.rec_step][0], traj_pos[self.rec_step][1],
                               traj_vel[self.rec_step], traj_ang[self.rec_step]])
        self.ctrl = np.array([0.0, 0.0])
        self.max_step = len(self.traj_info[0]) - 1
        self.lcl_smp = LocalSemanticMap(self.id, smp)  # not used
        self.timestep = 0.0
        # print("[Agent]: id: {} Initialized with traj_len:{}.".format(self.id, len(self.traj_info[0])))

    def check_trigger(self, sim_time):
        return True

    def step(self):
        if self.rec_step >= self.max_step:
            print("[Agent]: No.{} replay finished.".format(self.id))
            return
        self.rec_step += 1

    def update_state(self, dt):
        self.type = self.traj_type[self.rec_step]
        if self.type == ObjectType.VEHICLE:
            self.bbox = VehicleBBox()
        elif self.type == ObjectType.PEDESTRIAN:
            self.bbox = PedestrianBBox()
        elif self.type == ObjectType.MOTORCYCLIST:
            self.bbox = CyclistBBox()
        elif self.type == ObjectType.CYCLIST:
            self.bbox = CyclistBBox()
        elif self.type == ObjectType.BUS:
            self.bbox = BusBBox()
        elif self.type == ObjectType.UNKNOWN:
            self.bbox = UnknownBBox()
        else:
            self.bbox = UnknownBBox()  # for all static objects

        traj_pos, traj_ang, traj_vel = self.traj_info[:3]
        self.state = np.array([traj_pos[self.rec_step][0], traj_pos[self.rec_step][1],
                               traj_vel[self.rec_step], traj_ang[self.rec_step]])
        self.ctrl = np.array([0.0, 0.0])
        self.timestep += dt

    def is_valid(self):
        return self.traj_info[-1][self.rec_step]


class CustomizedAgent(NonReactiveAgent):
    def __init__(self):
        super(CustomizedAgent, self).__init__()
        self.last_pl_tri = None
        self.plan_rate = 10
        self.plan_step = 1.0 / self.plan_rate - 1e-4
        self.planner = None
        self.veh_param = VehicleParam()
        self.enable_timestep = 1e8
        self.is_enable = False

    def init(self, agt_id, traj_type, traj_cat, traj_info, smp, clr, use_traj=True, semantic_lane_id=None,
             target_velocity=None):
        super(CustomizedAgent, self).init(agt_id, traj_type, traj_cat, traj_info, smp, clr)

        # compute target lane by extending the recorded trajectory to the semantic lane
        virtual_semantic_lane, virtual_semantic_lane_info = self.get_target_lane(smp, use_traj, semantic_lane_id)

        #  compute target velocity
        if target_velocity is None:
            target_velocity = np.mean(self.traj_info[2], axis=0)

        self.lcl_smp = LocalSemanticMap(self.id, smp)
        self.lcl_smp.update_target_lane(virtual_semantic_lane)
        if virtual_semantic_lane_info is not None:
            self.lcl_smp.update_target_lane_info(virtual_semantic_lane_info)
        self.lcl_smp.update_target_velocity(target_velocity)
        self.timestep = 0.0
        self.init_state_ctrl()


    def get_target_lane(self, smp, use_traj, semantic_lane_id):
        traj_pos, traj_ang = self.traj_info[:2]

        if semantic_lane_id is None:  # get the closest semantic lane
            semantic_lane_id = self.get_closest_semantic_lane(smp, traj_pos, traj_ang)
            if semantic_lane_id is None:  # use the historical trajectory as the target lane
                virtual_target_lane = self.get_virtual_target_lane(traj_pos)
                # extending the historical trajectory as the semantic lane
                extend_pos = virtual_target_lane[-1] + (virtual_target_lane[-1] - virtual_target_lane[-2]) * 10.0
                virtual_target_lane = np.vstack([virtual_target_lane, extend_pos])
                return virtual_target_lane, None
            if use_traj:
                virtual_target_lane = self.get_virtual_target_lane(traj_pos)

                # find the closest point on the semantic lane to the last pos of the historical trajectory
                closest_idx = np.argmin(np.linalg.norm(smp.semantic_lanes[semantic_lane_id] - traj_pos[-1], axis=1))

                virtual_target_lane = np.vstack([virtual_target_lane, smp.semantic_lanes[semantic_lane_id][closest_idx:]])

                return virtual_target_lane, None
            else:
                return smp.semantic_lanes[semantic_lane_id], smp.semantic_lanes_infos[semantic_lane_id]
        else:
            if semantic_lane_id not in smp.semantic_lanes:
                raise ValueError("Semantic lane id {} not in the semantic map.".format(semantic_lane_id))
            if use_traj:
                virtual_target_lane = self.get_virtual_target_lane(traj_pos)
                # merge the virtual target lane with the semantic lane from the pos that is closest to the semantic lane
                diff = virtual_target_lane[:, np.newaxis, :] - smp.semantic_lanes[semantic_lane_id][np.newaxis, :, :]
                # compute the squared distance for each pair of points
                squared_distances = np.sum(diff ** 2, axis=2)
                # find the index of the minimum squared distance
                min_distance_index = np.argmin(squared_distances)
                # convert the index into two-dimensional indices corresponding to the positions in virtual_target_lane and semantic lane
                vir_idx, sml_idx = np.unravel_index(min_distance_index, squared_distances.shape)
                # truncate the virtual target lane to the closest point
                virtual_target_lane = virtual_target_lane[:vir_idx + 1]
                # merge the virtual target lane with the semantic lane from the closest point
                virtual_target_lane = np.vstack([virtual_target_lane, smp.semantic_lanes[semantic_lane_id][sml_idx:]])
                return virtual_target_lane, None
            else:
                return smp.semantic_lanes[semantic_lane_id], smp.semantic_lanes_infos[semantic_lane_id]

    def get_closest_semantic_lane(self, smp, traj_pos, traj_ang):
        # compute target lane by extending the historical lane to the semantic lane
        closest_lane_id = None
        # projection filtering
        min_dis_diff = 1e9
        ang_thres = np.pi / 4.0
        dis_thres = 5.0
        for lane_idx, lane in smp.semantic_lanes.items():
            start_proj_pt, start_proj_heading, _ = project_point_on_polyline(traj_pos[0], lane)
            start_ang_diff = np.abs(start_proj_heading - traj_ang[0])
            start_ang_diff = np.arctan2(np.sin(start_ang_diff), np.cos(start_ang_diff))
            start_dis_diff = np.linalg.norm(traj_pos[0] - start_proj_pt)
            if start_dis_diff > dis_thres or start_ang_diff > ang_thres:
                continue
            end_proj_pt, end_proj_heading, _ = project_point_on_polyline(traj_pos[-1], lane)
            # cal angle diff with normalization to [-pi, pi]
            end_ang_diff = np.abs(end_proj_heading - traj_ang[-1])
            end_ang_diff = np.arctan2(np.sin(end_ang_diff), np.cos(end_ang_diff))
            end_dis_diff = np.linalg.norm(traj_pos[-1] - end_proj_pt)
            if end_ang_diff < ang_thres and end_dis_diff < dis_thres:
                if end_dis_diff < min_dis_diff:
                    min_dis_diff = end_dis_diff
                    closest_lane_id = lane_idx
        return closest_lane_id

    def get_virtual_target_lane(self, traj_pos):
        # compute target lane by extending the historical lane to the semantic lane
        simplify_thres = 0.1
        traj_pos = remove_close_points(traj_pos, simplify_thres)
        virtual_semantic_lane = copy.deepcopy(traj_pos)
        return virtual_semantic_lane

    def set_enable_timestep(self, timestep):
        self.enable_timestep = timestep

    def check_enable(self, timestep):
        if timestep >= self.enable_timestep and not self.is_enable:
            self.is_enable = True
            self.init_state_ctrl()
            # self.clr = AgentColor().ego_enable()  # change the color to enable color

    def init_state_ctrl(self):
        #  get initial state from the cfg
        traj_pos, traj_ang, traj_vel = self.traj_info[:3]
        self.state = np.array([traj_pos[self.rec_step][0], traj_pos[self.rec_step][1],
                               traj_vel[self.rec_step], traj_ang[self.rec_step]])
        self.ctrl = np.array([0.0, 0.0])

    def init_planner(self, cfg_dir):
        pass

    def check_trigger(self, sim_time):
        record_trigger = False
        planner_trigger = False
        if not self.is_enable:
            record_trigger = super().check_trigger(sim_time)
        if self.last_pl_tri is None or (sim_time - self.last_pl_tri) >= self.plan_step:
            planner_trigger = True
            self.last_pl_tri = sim_time

        return record_trigger, planner_trigger

    def plan(self):
        return True, None

    def update_state(self, dt):
        if not self.is_enable:
            super().update_state(dt)
        else:
            self._update_state(dt)

    def _update_state(self, dt):
        self.state = kine_propagate(self.state, self.ctrl, dt, self.veh_param.wb, self.veh_param.max_spd,
                                    self.veh_param.max_str)
        self.timestep += dt

    def update_observation(self, agents):
        self.lcl_smp.update_observation(agents)


class MINDAgent(CustomizedAgent):
    def __init__(self):
        super(MINDAgent, self).__init__()
        self.gt_tgt_lane = None

    def init(self, agt_id, traj_type, traj_cat, traj_info, smp, clr, use_traj=False, semantic_lane_id=None,
             target_velocity=None):
        #  only use the semantic lane as the target lane
        super().init(agt_id, traj_type, traj_cat, traj_info, smp, clr, use_traj, semantic_lane_id, target_velocity)

    def init_planner(self, cfg_dir):
        self.planner = MINDPlanner(cfg_dir)

    def update_target_lane(self, smp, semantic_lane_id):
        self.gt_tgt_lane, _ = self.get_target_lane(smp, True, semantic_lane_id)
        self.gt_tgt_lane = remove_close_points(self.gt_tgt_lane, 4.0)
        self.planner.update_target_lane(self.gt_tgt_lane)

    def plan(self):
        self.planner.update_state_ctrl(self.lcl_smp.ego_agent.state, self.ctrl)
        is_success, self.ctrl, best_tree_set = self.planner.plan(self.lcl_smp)
        return is_success, best_tree_set

    def update_observation(self, agents):
        self.lcl_smp.update_observation(agents)
        self.planner.update_observation(self.lcl_smp)

