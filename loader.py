import numpy as np
from importlib import import_module
from common.data import padding_traj_nn
from common.geometry import project_point_on_polyline
from agent import AgentColor, MINDAgent, NonReactiveAgent
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import TrackCategory


class ArgoAgentLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_agents(self, smp, cl_agt_cfg=None):
        cl_agts = self.get_closed_loop_agents(cl_agt_cfg)
        trajs_info = self.get_trajs_info(smp)
        agents = []
        for traj_pos, traj_ang, traj_vel, traj_type, traj_tid, traj_cat, has_flag in zip(*trajs_info):
            traj_info = [traj_pos, traj_ang, traj_vel, has_flag]
            if traj_tid in cl_agts:
                agent_file, agent_name = cl_agts[traj_tid]["agent"].split(':')
                planner_cfg = cl_agts[traj_tid]["planner_config"]
                # get planner type
                agent = getattr(import_module(agent_file), agent_name)()

                if isinstance(agent, MINDAgent):
                    agt_clr = AgentColor().ego_disable()

                agent.init(traj_tid, traj_type, traj_cat, traj_info, smp, agt_clr,
                           semantic_lane_id=cl_agts[traj_tid]["semantic_lane"],
                           target_velocity=cl_agts[traj_tid]["target_velocity"])

                agent.set_enable_timestep(cl_agts[traj_tid]["enable_timestep"])
                agent.init_planner(planner_cfg)

                if isinstance(agent, MINDAgent):
                    agent.update_target_lane(smp, cl_agts[traj_tid]["semantic_lane"])

            else:
                agent = NonReactiveAgent()
                agt_clr = AgentColor().exo()
                agent.init(traj_tid, traj_type, traj_cat, traj_info, smp, agt_clr)
            agents.append(agent)
        return agents


    def get_closed_loop_agents(self, cl_agt_cfg):
        closed_loop_agents = dict()
        if cl_agt_cfg is None:
            return closed_loop_agents
        for c in cl_agt_cfg:
            agt_id = c["id"]
            if agt_id in closed_loop_agents.keys():
                continue
            closed_loop_agents[agt_id] = dict()
            closed_loop_agents[agt_id]["enable_timestep"] = c["enable_timestep"]
            if c["target_velocity"] == -1:
                closed_loop_agents[agt_id]["target_velocity"] = None
            else:
                closed_loop_agents[agt_id]["target_velocity"] = c["target_velocity"]
            if c["semantic_lane"] == -1:
                closed_loop_agents[agt_id]["semantic_lane"] = None
            else:
                closed_loop_agents[agt_id]["semantic_lane"] = c["semantic_lane"]
            closed_loop_agents[agt_id]["agent"] = c["agent"]
            closed_loop_agents[agt_id]["planner_config"] = c["planner_config"]
        return closed_loop_agents

    def get_trajs_info(self, smp):
        scenario = scenario_serialization.load_argoverse_scenario_parquet(self.data_path)

        obs_len = 50
        scored_idcs, unscored_idcs, fragment_idcs = list(), list(), list()  # exclude AV
        for idx, x in enumerate(scenario.tracks):
            if x.track_id == scenario.focal_track_id and x.category == TrackCategory.FOCAL_TRACK:
                focal_idx = idx
            elif x.track_id == 'AV':
                av_idx = idx
            elif x.category == TrackCategory.SCORED_TRACK:
                scored_idcs.append(idx)
            elif x.category == TrackCategory.UNSCORED_TRACK:
                unscored_idcs.append(idx)
            elif x.category == TrackCategory.TRACK_FRAGMENT:
                fragment_idcs.append(idx)

        assert av_idx is not None, '[ERROR] Wrong av_idx'
        assert focal_idx is not None, '[ERROR] Wrong focal_idx'
        assert av_idx not in unscored_idcs, '[ERROR] Duplicated av_idx'

        sorted_idcs = [focal_idx, av_idx] + scored_idcs + unscored_idcs + fragment_idcs
        sorted_cat = ["focal", "av"] + ["score"] * \
                     len(scored_idcs) + ["unscore"] * len(unscored_idcs) + ["frag"] * len(fragment_idcs)
        sorted_tid = [scenario.tracks[idx].track_id for idx in sorted_idcs]

        # * must follows the pre-defined order
        trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags = list(), list(), list(), list(), list()
        trajs_tid, trajs_cat = list(), list()  # track id and category
        for k, ind in enumerate(sorted_idcs):
            track = scenario.tracks[ind]

            traj_ts = np.array([x.timestep for x in track.object_states], dtype=np.int16)  # [N_{frames}]
            traj_pos = np.array([list(x.position) for x in track.object_states])  # [N_{frames}, 2]
            traj_ang = np.array([x.heading for x in track.object_states])  # [N_{frames}]
            traj_vel = np.array([list(x.velocity) for x in track.object_states])  # [N_{frames}, 2]
            # cal scalar velocity
            traj_vel = np.linalg.norm(traj_vel, axis=1)  # [N_{frames}]

            ts = np.arange(0, 110)  # [0, 1,..., 109]
            ts_obs = ts[obs_len - 1]  # always 49

            # # * only contains future part
            if traj_ts[0] > ts_obs:
                continue
            # # * not observed at ts_obs
            if ts_obs not in traj_ts:
                continue

            # * far away from map (only for observed part)
            traj_obs_pts = traj_pos[:obs_len]  # [N_{frames}, 2]
            on_lanes = []
            on_lane_thres = 5.0
            for traj_pt in traj_obs_pts:
                on_lane = False
                for semantic_lane in smp.semantic_lanes.values():
                    proj_pt, _, _ = project_point_on_polyline(traj_pt, semantic_lane)
                    if np.linalg.norm(proj_pt - traj_pt) < on_lane_thres:
                        on_lane = True
                        break
                on_lanes.append(on_lane)

            # if any of the observed points is not on the lane, then skip
            if not np.all(on_lanes):
                continue

            has_flag = np.zeros_like(ts)
            # # print(has_flag.shape, traj_ts.shape, traj_ts)
            has_flag[traj_ts] = 1

            # object type
            traj_type = [track.object_type for _ in range(len(ts))]

            # pad pos, nearest neighbor
            traj_pos_pad = np.full((len(ts), 2), None)
            traj_pos_pad[traj_ts] = traj_pos
            traj_pos_pad = padding_traj_nn(traj_pos_pad)
            # pad ang, nearest neighbor
            traj_ang_pad = np.full(len(ts), None)
            traj_ang_pad[traj_ts] = traj_ang
            traj_ang_pad = padding_traj_nn(traj_ang_pad)
            traj_vel_pad = np.full((len(ts),), 0.0)
            traj_vel_pad[traj_ts] = traj_vel

            trajs_pos.append(traj_pos_pad)
            trajs_ang.append(traj_ang_pad)
            trajs_vel.append(traj_vel_pad)
            has_flags.append(has_flag)
            trajs_type.append(traj_type)
            trajs_tid.append(sorted_tid[k])
            trajs_cat.append(sorted_cat[k])

        res_traj_infos = self.resample_trajs_info(
            [trajs_pos, trajs_ang, trajs_vel, trajs_type, trajs_tid, trajs_cat, has_flags])

        trajs_pos, trajs_ang, trajs_vel, trajs_type, trajs_tid, trajs_cat, has_flags = res_traj_infos

        trajs_pos = np.array(trajs_pos).astype(np.float32)  # [N, 110(50), 2]
        trajs_ang = np.array(trajs_ang).astype(np.float32)  # [N, 110(50)]
        trajs_vel = np.array(trajs_vel).astype(np.float32)  # [N, 110(50), 2]
        has_flags = np.array(has_flags).astype(np.int16)  # [N, 110(50)]

        return trajs_pos, trajs_ang, trajs_vel, trajs_type, trajs_tid, trajs_cat, has_flags

    def resample_trajs_info(self, trajs_info):
        # traj_info = traj_pos, traj_ang, traj_vel, traj_type, traj_tid, traj_cat, has_flag
        ori_sim_step = 0.1
        sim_step = 0.02
        res_trajs_pos, res_trajs_ang, res_trajs_vel, res_trajs_type, res_trajs_tid, res_trajs_cat, res_has_flags = [], [], [], [], [], [], []
        interp_len = int(ori_sim_step / sim_step)

        trajs_pos, trajs_ang, trajs_vel, trajs_type, trajs_tid, trajs_cat, has_flags = trajs_info
        for a_idx in range(len(trajs_pos)):
            res_traj_pos, res_traj_ang, res_traj_vel, res_traj_type, res_traj_tid, res_traj_cat, res_has_flag = [], [], [], [], [], [], []
            for t_idx in range(len(trajs_pos[a_idx])):
                if t_idx == len(trajs_pos[a_idx]) - 1:
                    res_traj_pos.append(trajs_pos[a_idx][t_idx])
                    res_traj_ang.append(trajs_ang[a_idx][t_idx])
                    res_traj_vel.append(trajs_vel[a_idx][t_idx])
                    res_has_flag.append(has_flags[a_idx][t_idx])
                    res_traj_type.append(trajs_type[a_idx][t_idx])
                else:
                    for iidx in range(interp_len):
                        r = iidx / interp_len
                        res_traj_pos.append(trajs_pos[a_idx][t_idx] * (1 - r) + trajs_pos[a_idx][t_idx + 1] * r)
                        angle_diff = trajs_ang[a_idx][t_idx + 1] - trajs_ang[a_idx][t_idx]
                        # normalize to [-pi, pi]
                        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                        interp_ang = trajs_ang[a_idx][t_idx] + angle_diff * r
                        # normalize to [-pi, pi]
                        interp_ang = np.arctan2(np.sin(interp_ang), np.cos(interp_ang))
                        res_traj_ang.append(interp_ang)
                        res_traj_vel.append(trajs_vel[a_idx][t_idx] * (1 - r) + trajs_vel[a_idx][t_idx + 1] * r)
                        res_has_flag.append(has_flags[a_idx][t_idx] * (1 - r) + has_flags[a_idx][t_idx + 1] * r > 0.5)
                        res_traj_type.append(trajs_type[a_idx][t_idx])

            res_trajs_pos.append(np.array(res_traj_pos))
            res_trajs_ang.append(np.array(res_traj_ang))
            res_trajs_vel.append(np.array(res_traj_vel))
            res_trajs_type.append(res_traj_type)
            res_has_flags.append(np.array(res_has_flag))
            res_trajs_tid.append(trajs_tid[a_idx])
            res_trajs_cat.append(trajs_cat[a_idx])

        res_traj_info = [res_trajs_pos, res_trajs_ang, res_trajs_vel, res_trajs_type, res_trajs_tid, res_trajs_cat,
                         res_has_flags]
        return res_traj_info
