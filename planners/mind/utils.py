import torch
import copy
import numpy as np
from typing import List, Any, Dict
from shapely.geometry import LineString
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.datasets.motion_forecasting.data_schema import ObjectType

def gpu(data, device):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x, device=device) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data, device=device) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().to(device, non_blocking=True)
    return data



def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def padding_traj_nn(traj):
    n = len(traj)
    # forward
    buff = None
    for i in range(n):
        if np.all(buff == None) and np.all(traj[i] != None):
            buff = traj[i]
        if np.all(buff != None) and np.all(traj[i] == None):
            traj[i] = buff
        if np.all(buff != None) and np.all(traj[i] != None):
            buff = traj[i]
    # backward
    buff = None
    for i in reversed(range(n)):
        if np.all(buff == None) and np.all(traj[i] != None):
            buff = traj[i]
        if np.all(buff != None) and np.all(traj[i] == None):
            traj[i] = buff
        if np.all(buff != None) and np.all(traj[i] != None):
            buff = traj[i]
    return traj


def tgt_gather(batch_size, tgt_nodes_list, tgt_rpe_list):
    tgt_nodes_feat = []
    tgt_rpe_feat = []
    # ~ calc tgt feat
    for tgt_nodes, tgt_rpe in zip(tgt_nodes_list, tgt_rpe_list):
        tgt_nodes_feat.append(tgt_nodes)
        tgt_rpe_feat.append(tgt_rpe)

    tgt_nodes_feat = torch.stack(tgt_nodes_feat, dim=0)
    tgt_rpe_feat = torch.stack(tgt_rpe_feat, dim=0).reshape(batch_size, -1)
    return tgt_nodes_feat, tgt_rpe_feat


def graph_gather(batch_size, graphs):
    '''
        graphs[i]
            node_ctrs           torch.Size([116, N_{pt}, 2])
            node_vecs           torch.Size([116, N_{pt}, 2])
            intersect           torch.Size([116, N_{pt}])
            lane_type           torch.Size([116, N_{pt}, 3])
            cross_left          torch.Size([116, N_{pt}, 3])
            cross_right         torch.Size([116, N_{pt}, 3])
            left                torch.Size([116, N_{pt}])
            right               torch.Size([116, N_{pt}])
            lane_ctrs           torch.Size([116, 2])
            lane_vecs           torch.Size([116, 2])
            num_nodes           1160
            num_lanes           116
    '''
    lane_idcs = list()
    lane_count = 0
    for i in range(batch_size):
        l_idcs = torch.arange(lane_count, lane_count + graphs[i]["num_lanes"])
        lane_idcs.append(l_idcs)
        lane_count = lane_count + graphs[i]["num_lanes"]

    graph = dict()
    for key in ["node_ctrs", "node_vecs", "intersect", "lane_type", "cross_left", "cross_right", "left", "right"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)
    for key in ["lane_ctrs", "lane_vecs"]:
        graph[key] = [x[key] for x in graphs]

    lanes = torch.cat([graph['node_ctrs'],
                       graph['node_vecs'],
                       graph['intersect'].unsqueeze(2),
                       graph['lane_type'],
                       graph['cross_left'],
                       graph['cross_right'],
                       graph['left'].unsqueeze(2),
                       graph['right'].unsqueeze(2)], dim=-1)  # [N_{lane}, 9, F]
    return lanes, lane_idcs


def actor_gather(batch_size, trajs):
    num_actors = [len(x['TRAJS_CTRS']) for x in trajs]

    act_feats = []
    for i in range(batch_size):
        traj_pos = trajs[i]['TRAJS_POS_OBS']
        traj_disp = torch.zeros_like(traj_pos)
        traj_disp[:, 1:, :] = traj_pos[:, 1:, :] - traj_pos[:, :-1, :]

        act_feat = torch.cat([traj_disp,
                              trajs[i]['TRAJS_ANG_OBS'],
                              trajs[i]['TRAJS_VEL_OBS'],
                              trajs[i]['TRAJS_TYPE'],
                              trajs[i]['PAD_OBS'].unsqueeze(-1)], dim=-1)
        act_feats.append(act_feat)

    act_feats = [x.transpose(1, 2) for x in act_feats]
    actors = torch.cat(act_feats, 0)  # [N_a, feat_len, 50], N_a is agent number in a batch
    actors = actors[..., 2:]  # ! tmp solution
    actor_idcs = []  # e.g. [tensor([0, 1, 2, 3]), tensor([ 4,  5,  6,  7,  8,  9, 10])]
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i])
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs


def collate_fn(batch: List[Any]) -> Dict[str, Any]:
    if len(batch) == 0:
        return None
    batch = from_numpy(batch)
    data = dict()
    data['BATCH_SIZE'] = len(batch)
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        data[key] = [x[key] for x in batch]
    '''
        Keys:
        'BATCH_SIZE',
        'ORIG', 'ROT',
        'TRAJS', 'LANE_GRAPH', 'RPE'
    '''

    actors, actor_idcs = actor_gather(data['BATCH_SIZE'], data['TRAJS'])
    lanes, lane_idcs = graph_gather(data['BATCH_SIZE'], data["LANE_GRAPH"])
    tgt_nodes, tgt_rpe = tgt_gather(data['BATCH_SIZE'], data['TGT_NODES'], data['TGT_RPE'])

    data['ACTORS'] = actors
    data['ACTOR_IDCS'] = actor_idcs
    data['LANES'] = lanes
    data['LANE_IDCS'] = lane_idcs
    data['TGT_NODES'] = tgt_nodes
    data['TGT_RPE'] = tgt_rpe
    return data


def get_new_lane_graph(lane_graph, orig, rot, device):
    ret_lane_graph = gpu(copy.deepcopy(lane_graph), device=device)
    # transform the lane_ctrs and lane_vecs
    ret_lane_graph['lane_ctrs'] = torch.matmul(ret_lane_graph['lane_ctrs'] - orig, rot)
    ret_lane_graph['lane_vecs'] = torch.matmul(ret_lane_graph['lane_vecs'], rot)

    return ret_lane_graph


def get_origin_rotation(traj_pos, traj_ang, device):
    obs_len = 50
    orig = traj_pos[obs_len - 1]
    theta = traj_ang[obs_len - 1]
    if isinstance(orig, torch.Tensor):
        rot = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta), torch.cos(theta)]]).to(device)
    elif isinstance(orig, np.ndarray):
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    return orig, rot, theta


def get_rpe(ctrs, vecs, radius=100.0):
    # distance encoding
    d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
    mask = None
    d_pos = d_pos * 2 / radius  # scale [0, radius] to [0, 2]
    pos_rpe = d_pos.unsqueeze(0)

    # angle diff
    cos_a1 = get_cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
    sin_a1 = get_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
    # print('cos_a1: ', cos_a1.shape, 'sin_a1: ', sin_a1.shape)

    v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
    cos_a2 = get_cos(vecs.unsqueeze(0), v_pos)
    sin_a2 = get_sin(vecs.unsqueeze(0), v_pos)
    # print('cos_a2: ', cos_a2.shape, 'sin_a2: ', sin_a2.shape)

    ang_rpe = torch.stack([cos_a1, sin_a1, cos_a2, sin_a2])
    rpe = torch.cat([ang_rpe, pos_rpe], dim=0)
    return rpe, mask


def get_angle(vel):
    return torch.atan2(vel[..., 1], vel[..., 0])


def get_cos(v1, v2):
    ''' input: [M, N, 2], [M, N, 2]
        output: [M, N]
        cos(<a,b>) = (a dot b) / |a||b|
    '''
    v1_norm = v1.norm(dim=-1)
    v2_norm = v2.norm(dim=-1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]
    cos_dang = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
    return cos_dang


def get_sin(v1, v2):
    ''' input: [M, N, 2], [M, N, 2]
        output: [M, N]
        sin(<a,b>) = (a x b) / |a||b|
    '''
    v1_norm = v1.norm(dim=-1)
    v2_norm = v2.norm(dim=-1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]
    sin_dang = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
    return sin_dang


def get_agent_trajectories(agent_obs, device):
    obs_len = 50

    # * find idcs
    av_idx = None
    exo_idcs = list()  # exclude AV
    key_list = []
    for idx, key in enumerate(agent_obs.keys()):
        if key == 'AV':
            av_idx = idx
        else:
            exo_idcs.append(idx)
        key_list.append(key)

    sorted_idcs = [av_idx] + exo_idcs
    sorted_cat = ["av"] + ["exo"] * len(exo_idcs)
    sorted_tid = [key_list[idx] for idx in sorted_idcs]

    # * get timesteps and timesteps
    ts = np.arange(0, obs_len)  # [0, 1,..., 49]
    ts_obs = ts[obs_len - 1]  # always 49

    # * must follows the pre-defined order
    trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags = list(), list(), list(), list(), list()
    trajs_tid, trajs_cat = list(), list()  # track id and category
    for k, ind in enumerate(sorted_idcs):
        key = key_list[ind]
        track = agent_obs[key]

        # * pass if no observation at the last timestep
        if track.object_states[-1].observed is False:
            continue

        # * get traj
        observed_flag = np.array([1 if s.observed else 0 for s in track.object_states])

        traj_ts = np.arange(obs_len - len(track.object_states), obs_len)
        traj_ts = traj_ts[observed_flag == 1]

        traj_pos = np.array(
            [list(x.position) if x.observed else [0.0, 0.0] for x in track.object_states])  # [N_{frames}, 2]
        traj_pos = traj_pos[observed_flag == 1]
        traj_ang = np.array([x.heading if x.observed else 0.0 for x in track.object_states])  # [N_{frames}]
        traj_ang = traj_ang[observed_flag == 1]
        traj_vel = np.array(
            [list(x.velocity) if x.observed else [0.0, 0.0] for x in track.object_states])  # [N_{frames}, 2]
        traj_vel = traj_vel[observed_flag == 1]

        # print(has_flag.shape, traj_ts.shape, traj_ts)
        has_flag = np.zeros_like(ts)
        has_flag[traj_ts] = 1
        # object type
        obj_type = np.zeros(7)  # 7 types
        if track.object_type == ObjectType.VEHICLE:
            obj_type[0] = 1
        elif track.object_type == ObjectType.PEDESTRIAN:
            obj_type[1] = 1
        elif track.object_type == ObjectType.MOTORCYCLIST:
            obj_type[2] = 1
        elif track.object_type == ObjectType.CYCLIST:
            obj_type[3] = 1
        elif track.object_type == ObjectType.BUS:
            obj_type[4] = 1
        elif track.object_type == ObjectType.UNKNOWN:
            obj_type[5] = 1
        else:
            obj_type[6] = 1  # for all static objects
        traj_type = np.zeros((len(ts), 7))
        traj_type[traj_ts] = obj_type

        # pad pos, nearest neighbor
        traj_pos_pad = np.full((len(ts), 2), None)
        traj_pos_pad[traj_ts] = traj_pos
        traj_pos_pad = padding_traj_nn(traj_pos_pad)
        # pad ang, nearest neighbor
        traj_ang_pad = np.full(len(ts), None)
        traj_ang_pad[traj_ts] = traj_ang
        traj_ang_pad = padding_traj_nn(traj_ang_pad)
        # pad vel, fill zeros
        traj_vel_pad = np.full((len(ts), 2), 0.0)
        traj_vel_pad[traj_ts] = traj_vel

        trajs_pos.append(traj_pos_pad)
        trajs_ang.append(traj_ang_pad)
        trajs_vel.append(traj_vel_pad)
        trajs_type.append(traj_type)
        has_flags.append(has_flag)
        trajs_tid.append(sorted_tid[k])
        trajs_cat.append(sorted_cat[k])

    trajs_pos = np.array(trajs_pos).astype(np.float32)  # [N, 110(50), 2]
    trajs_ang = np.array(trajs_ang).astype(np.float32)  # [N, 110(50)]
    trajs_vel = np.array(trajs_vel).astype(np.float32)  # [N, 110(50), 2]
    trajs_type = np.array(trajs_type).astype(np.int16)  # [N, 110(50), 7]
    has_flags = np.array(has_flags).astype(np.int16)  # [N, 110(50)]
    return (torch.from_numpy(trajs_pos).to(device), torch.from_numpy(trajs_ang).to(device),
            torch.from_numpy(trajs_vel).to(device), torch.from_numpy(trajs_type).to(device),
            torch.from_numpy(has_flags).to(device), trajs_tid, trajs_cat)


def update_lane_graph_from_argo(static_map, orig, rot):
    node_ctrs, node_vecs, lane_type, intersect, cross_left, cross_right, left, right = [], [], [], [], [], [], [], []
    lane_ctrs, lane_vecs = [], []
    NUM_SEG_POINTS = 10
    SEG_LENGTH = 15.0

    for lane_id, lane in static_map.vector_lane_segments.items():
        # get lane centerline
        cl_raw = static_map.get_lane_segment_centerline(lane_id)[:, 0:2]  # use xy
        assert cl_raw.shape[0] == NUM_SEG_POINTS, "[Error] Wrong num of points in lane - {}:{}".format(
            lane_id, cl_raw.shape[0])

        cl_ls = LineString(cl_raw)
        num_segs = np.max([int(np.floor(cl_ls.length / SEG_LENGTH)), 1])
        ds = cl_ls.length / num_segs

        for i in range(num_segs):
            s_lb = i * ds
            s_ub = (i + 1) * ds
            num_sub_segs = NUM_SEG_POINTS

            cl_pts = []
            for s in np.linspace(s_lb, s_ub, num_sub_segs + 1):
                cl_pts.append(cl_ls.interpolate(s))
            ctrln = np.array(LineString(cl_pts).coords)  # [num_sub_segs + 1, 2]
            ctrln = (ctrln - orig).dot(rot)  # to local frame

            anch_pos = np.mean(ctrln, axis=0)
            anch_vec = (ctrln[-1] - ctrln[0]) / np.linalg.norm(ctrln[-1] - ctrln[0])
            anch_rot = np.array([[anch_vec[0], -anch_vec[1]],
                                 [anch_vec[1], anch_vec[0]]])

            lane_ctrs.append(anch_pos)
            lane_vecs.append(anch_vec)

            ctrln = (ctrln - anch_pos).dot(anch_rot)  # to instance frame

            ctrs = np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32)
            vecs = np.asarray(ctrln[1:] - ctrln[:-1], np.float32)
            node_ctrs.append(ctrs)  # middle point
            node_vecs.append(vecs)

            # ~ lane type
            lane_type_tmp = np.zeros(3)
            if lane.lane_type == LaneType.VEHICLE:
                lane_type_tmp[0] = 1
            elif lane.lane_type == LaneType.BIKE:
                lane_type_tmp[1] = 1
            elif lane.lane_type == LaneType.BUS:
                lane_type_tmp[2] = 1
            else:
                assert False, "[Error] Wrong lane type"
            lane_type.append(np.expand_dims(lane_type_tmp, axis=0).repeat(num_sub_segs, axis=0))

            # ~ intersection
            if lane.is_intersection:
                intersect.append(np.ones(num_sub_segs, np.float32))
            else:
                intersect.append(np.zeros(num_sub_segs, np.float32))

            # ~ lane marker type
            cross_left_tmp = np.zeros(3)
            if lane.left_mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                                       LaneMarkType.DASH_SOLID_WHITE,
                                       LaneMarkType.DASHED_WHITE,
                                       LaneMarkType.DASHED_YELLOW,
                                       LaneMarkType.DOUBLE_DASH_YELLOW,
                                       LaneMarkType.DOUBLE_DASH_WHITE]:
                cross_left_tmp[0] = 1  # crossable
            elif lane.left_mark_type in [LaneMarkType.DOUBLE_SOLID_YELLOW,
                                         LaneMarkType.DOUBLE_SOLID_WHITE,
                                         LaneMarkType.SOLID_YELLOW,
                                         LaneMarkType.SOLID_WHITE,
                                         LaneMarkType.SOLID_DASH_WHITE,
                                         LaneMarkType.SOLID_DASH_YELLOW,
                                         LaneMarkType.SOLID_BLUE]:
                cross_left_tmp[1] = 1  # not crossable
            else:
                cross_left_tmp[2] = 1  # unknown/none

            cross_right_tmp = np.zeros(3)
            if lane.right_mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                                        LaneMarkType.DASH_SOLID_WHITE,
                                        LaneMarkType.DASHED_WHITE,
                                        LaneMarkType.DASHED_YELLOW,
                                        LaneMarkType.DOUBLE_DASH_YELLOW,
                                        LaneMarkType.DOUBLE_DASH_WHITE]:
                cross_right_tmp[0] = 1  # crossable
            elif lane.right_mark_type in [LaneMarkType.DOUBLE_SOLID_YELLOW,
                                          LaneMarkType.DOUBLE_SOLID_WHITE,
                                          LaneMarkType.SOLID_YELLOW,
                                          LaneMarkType.SOLID_WHITE,
                                          LaneMarkType.SOLID_DASH_WHITE,
                                          LaneMarkType.SOLID_DASH_YELLOW,
                                          LaneMarkType.SOLID_BLUE]:
                cross_right_tmp[1] = 1  # not crossable
            else:
                cross_right_tmp[2] = 1  # unknown/none

            cross_left.append(np.expand_dims(cross_left_tmp, axis=0).repeat(num_sub_segs, axis=0))
            cross_right.append(np.expand_dims(cross_right_tmp, axis=0).repeat(num_sub_segs, axis=0))

            # ~ has left/right neighbor
            if lane.left_neighbor_id is None:
                left.append(np.zeros(num_sub_segs, np.float32))  # w/o left neighbor
            else:
                left.append(np.ones(num_sub_segs, np.float32))
            if lane.right_neighbor_id is None:
                right.append(np.zeros(num_sub_segs, np.float32))  # w/o right neighbor
            else:
                right.append(np.ones(num_sub_segs, np.float32))

    node_idcs = []  # List of range
    count = 0
    for i, ctr in enumerate(node_ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        count += len(ctr)

    lane_idcs = []  # node belongs to which lane, e.g. [0   0   0 ... 122 122 122]
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int16))
    # print("lane_idcs: ", lane_idcs.shape, lane_idcs)

    graph = dict()
    # geometry
    graph['node_ctrs'] = np.stack(node_ctrs, axis=0).astype(np.float32)
    graph['node_vecs'] = np.stack(node_vecs, axis=0).astype(np.float32)
    graph['lane_ctrs'] = np.array(lane_ctrs).astype(np.float32)
    graph['lane_vecs'] = np.array(lane_vecs).astype(np.float32)
    # node features
    graph['lane_type'] = np.stack(lane_type, axis=0).astype(np.int16)
    graph['intersect'] = np.stack(intersect, axis=0).astype(np.int16)
    graph['cross_left'] = np.stack(cross_left, axis=0).astype(np.int16)
    graph['cross_right'] = np.stack(cross_right, axis=0).astype(np.int16)
    graph['left'] = np.stack(left, axis=0).astype(np.int16)
    graph['right'] = np.stack(right, axis=0).astype(np.int16)
    graph['num_nodes'] = graph['node_ctrs'].shape[0] * graph['node_ctrs'].shape[1]
    graph['num_lanes'] = graph['lane_ctrs'].shape[0]
    return graph


def get_closest_point_on_segment(segment, point):
    p1, p2 = segment
    # Vector from p1 to p2
    segment_vector = p2 - p1

    # Projected vector from p1 to p
    projected_vector = torch.dot(point - p1, segment_vector) / torch.dot(segment_vector, segment_vector)

    # Clamp the projection to the segment
    t = torch.clamp(projected_vector, 0, 1)

    # Find the closest point on the segment
    closest = p1 + t * segment_vector
    return closest


def get_distance_to_polyline(polyline, point):
    min_distance = None

    for i in range(len(polyline) - 1):
        segment = (polyline[i], polyline[i + 1])
        closest = get_closest_point_on_segment(segment, point)
        distance = torch.norm(closest - point)

        if min_distance is None or distance < min_distance:
            min_distance = distance

    return min_distance


def get_covariance_matrix(data):
    # check is torch or numpy
    if isinstance(data, torch.Tensor):
        ret_shape = data.shape[:-1] + (2, 2)
        sigma_x = data[..., 0]
        sigma_y = data[..., 1]
        rho = data[..., 2]
        sigma_xy = rho * sigma_x * sigma_y
        return torch.stack([sigma_x ** 2, sigma_xy, sigma_xy, sigma_y ** 2], dim=-1).view(ret_shape)
    elif isinstance(data, np.ndarray):
        ret_shape = data.shape[:-1] + (2, 2)
        sigma_x = data[..., 0]
        sigma_y = data[..., 1]
        rho = data[..., 2]
        sigma_xy = rho * sigma_x * sigma_y
        return np.stack([sigma_x ** 2, sigma_xy, sigma_xy, sigma_y ** 2], axis=-1).reshape(ret_shape)
    else:
        raise ValueError("data should be torch.Tensor or numpy.ndarray")


def get_max_covariance(data):
    # check is torch or numpy
    if isinstance(data, torch.Tensor):
        ret_shape = data.shape[:-1] + (1,)
        sigma_x = data[..., 0]
        sigma_y = data[..., 1]
        # only return the maximum sigma
        return torch.maximum(sigma_x, sigma_y).view(ret_shape)
    elif isinstance(data, np.ndarray):
        ret_shape = data.shape[:-1] + (1,)
        sigma_x = data[..., 0]
        sigma_y = data[..., 1]
        # only return the maximum sigma
        return np.maximum(sigma_x, sigma_y).reshape(ret_shape)
    else:
        raise ValueError("data should be torch.Tensor or numpy.ndarray")
