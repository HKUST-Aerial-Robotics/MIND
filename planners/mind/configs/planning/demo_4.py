import numpy as np

class ScenTreeCfg:
    def __init__(self):
        self.max_depth = 5
        self.tar_dist_thres = 10.0
        self.tar_time_ahead = 5.0
        self.seg_length = 15.0  # approximated lane segment length
        self.seg_n_node = 10
        self.far_dist_thres = 10.0


class TrajTreeCfg:
    def __init__(self):
        self.dt = 0.2  # Discrete time step.
        self.state_size = 6
        self.action_size = 2
        self.w_opt_cfg = dict()
        self.opt_cfg = dict()
        # ========================== warm start opt config =======================
        w_des_state = 0.0 * np.eye(self.state_size)
        w_des_state[2, 2] = .1  # weight on velocity
        w_des_state[4, 4] = 1.  # weight on acceleration
        w_des_state[5, 5] = 10.0  # weight on steering angle
        
        self.w_opt_cfg['w_des_state'] = w_des_state
        
        w_state_con = np.zeros((self.state_size, self.state_size))
        w_state_con[2, 2] = 50.0  # weight on velocity
        w_state_con[4, 4] = 50.0  # weight on acceleration
        w_state_con[5, 5] = 500.0  # weight on steering angle

        self.w_opt_cfg['w_state_con'] = w_state_con
        
        state_upper_bound = np.array([100000.0, 100000.0, 8.0, 10.0, 4.0, 0.2])
        state_lower_bound = np.array([-100000.0, -100000.0, 0.0, -10.0, -6.0, -0.2])
        
        self.w_opt_cfg['state_upper_bound'] = state_upper_bound
        self.w_opt_cfg['state_lower_bound'] = state_lower_bound
        
        self.w_opt_cfg['w_ctrl'] = 5.0 * np.eye(self.action_size)

        self.w_opt_cfg['w_tgt'] = 1.0
        
        self.w_opt_cfg['smooth_grid_res'] = 0.4
        self.w_opt_cfg['smooth_grid_size'] = (256, 256)

        # ========================== opt config =======================
        w_des_state = 0.0 * np.eye(self.state_size)
        w_des_state[2, 2] = .1  # weight on velocity
        w_des_state[4, 4] = 1.  # weight on acceleration
        w_des_state[5, 5] = 10.0  # weight on steering angle

        self.opt_cfg['w_des_state'] = w_des_state

        w_state_con = np.zeros((self.state_size, self.state_size))
        w_state_con[2, 2] = 50.0  # weight on velocity
        w_state_con[4, 4] = 50.0  # weight on acceleration
        w_state_con[5, 5] = 500.0  # weight on steering angle

        self.opt_cfg['w_state_con'] = w_state_con

        state_upper_bound = np.array([100000.0, 100000.0, 8.0, 10.0, 4.0, 0.2])
        state_lower_bound = np.array([-100000.0, -100000.0, 0.0, -10.0, -6.0, -0.2])

        self.opt_cfg['state_upper_bound'] = state_upper_bound
        self.opt_cfg['state_lower_bound'] = state_lower_bound

        self.opt_cfg['w_ctrl'] = 5.0 * np.eye(self.action_size)

        self.opt_cfg['w_tgt'] = 1.0

        self.opt_cfg['smooth_grid_res'] = 0.4
        self.opt_cfg['smooth_grid_size'] = (256, 256)

        self.opt_cfg['w_ego'] = 1.0
        self.opt_cfg['w_ego_cov_offset'] = 1.0

        self.opt_cfg['w_exo'] = 10.0
        self.opt_cfg['w_exo_cov_offset'] = 2.5
        self.opt_cfg['w_exo_cost_offset'] = 10.0

        




