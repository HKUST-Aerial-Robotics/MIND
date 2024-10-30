class NetCfg():
    def __init__(self):
        self.g_cfg = dict()
        self.g_cfg['g_num_modes'] = 6
        self.g_cfg['g_obs_len'] = 50
        self.g_cfg['g_pred_len'] = 60

    def get_net_cfg(self):
        net_cfg = dict()
        net_cfg["network"] = "planners.mind.networks.network:ScenePredNet"
        net_cfg["in_actor"] = 14
        net_cfg["d_actor"] = 128
        net_cfg["n_fpn_scale"] = 4
        net_cfg["in_lane"] = 16
        net_cfg["d_lane"] = 128

        net_cfg["d_rpe_in"] = 5
        net_cfg["d_rpe"] = 128
        net_cfg["d_embed"] = 128
        net_cfg["n_scene_layer"] = 6
        net_cfg["n_scene_head"] = 8
        net_cfg["dropout"] = 0.1
        net_cfg["update_edge"] = True

        net_cfg["param_out"] = 'bezier'

        net_cfg.update(self.g_cfg)  # append global config
        return net_cfg
