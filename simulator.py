import os
import json
import shutil
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from common.visualization import draw_map, draw_agent, draw_scen_trees, reset_ax, draw_traj_trees, draw_traj

from agent import CustomizedAgent, NonReactiveAgent
from loader import ArgoAgentLoader
from common.semantic_map import SemanticMap
matplotlib.use('Agg')


class Simulator:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        self.sim_name = self.config['sim_name']
        self.seq_id = self.config['seq_id']
        self.output_dir = self.config['output_dir']
        self.num_threads = self.config['num_threads']
        self.seq_path = os.path.join('data/', self.seq_id)

        self.smp = SemanticMap()
        self.smp.load_from_argo2(Path(self.seq_path + f"/log_map_archive_{self.seq_id}.json"))

        self.render = self.config['render']
        self.cl_agents = self.config['cl_agents']

        self.sim_time = 0.0
        self.sim_step = 0.02
        self.sim_horizon = 500
        self.agents = []
        self.frames = []

    def run(self):
        self.init_sim()
        self.run_sim()
        self.render_video()

    def init_sim(self):
        self.agents = []
        scenario_path = Path(self.seq_path + f"/scenario_{self.seq_id}.parquet")
        replay_agent_loader = ArgoAgentLoader(scenario_path)
        self.agents += replay_agent_loader.load_agents(self.smp, self.cl_agents)

    def run_sim(self):
        print("Running simulation...")
        # reset sim time and frames
        self.frames = []
        self.sim_time = 0.0
        terminated = False

        for _ in tqdm(range(self.sim_horizon)):
            frame = {}
            # Update agent observations
            agent_obs = []
            for agent in self.agents:
                if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                    agent_obs.append(agent.observe())

            # Record ground truth
            agent_gt = []
            for agent in self.agents:
                if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                    agent_gt.append(agent.observe_no_noise())

            frame['agents'] = agent_gt

            # Update local semantic map and plan
            for agent in self.agents:
                if isinstance(agent, CustomizedAgent):
                    agent.check_enable(self.sim_time)
                    rec_tri, pl_tri = agent.check_trigger(self.sim_time)

                    if rec_tri:
                        agent.step()
                    if pl_tri:
                        agent.update_observation(agent_obs)
                        if agent.is_enable:  # if enable then plan to get control
                            is_success, res = agent.plan()
                            if not is_success:
                                print("Agent {} plan failed!".format(agent.id))
                                terminated = True
                                break

                            # hack for recording the planning result
                            if agent.id == 'AV':
                                frame['scen_tree'] = res[0]
                                frame['traj_tree'] = res[1]

                elif isinstance(agent, NonReactiveAgent):
                    agent.step()
                else:
                    raise ValueError("Unknown agent type")
                agent.update_state(self.sim_step)

            self.frames.append(frame)
            self.sim_time += self.sim_step

            if terminated:
                print("Simulation terminated!")
                break

    def render_video(self):
        if not self.render:
            return
        print("Rendering video...")
        # check directory exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        img_dir = self.output_dir + '/imgs'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # output frame png multiprocessing use the spawn method to create a new process
        ctx = torch.multiprocessing.get_context('spawn')
        pool = ctx.Pool(self.num_threads)
        pool.starmap(self.render_png, [(frame_idx, img_dir) for frame_idx in range(len(self.frames))])
        pool.close()

        # call ffmpeg to combine images into a video
        video_name = f'{self.seq_id}_{self.sim_name}.mov'
        output_command = "ffmpeg -r 25 -i " + img_dir + f'/frame_%03d.png' + " -vcodec mpeg4 -y " + \
                         self.output_dir + video_name
        os.system(output_command)
        shutil.rmtree(img_dir)

    ########################################
    # Visualization functions
    ########################################
    def render_png(self, frame_idx, img_dir):
        fig = plt.figure(figsize=(48, 48))
        ax = fig.add_subplot(111, projection='3d')
        plt.tight_layout()
        self.render_frame(frame_idx, ax)
        # Save the frame with directory path
        frame_filename = img_dir + f'/frame_{frame_idx:03d}.png'
        plt.tight_layout()
        plt.savefig(frame_filename)
        plt.close(fig)

    def render_frame(self, frame_idx, ax):
        scen_tree_vis = None
        traj_tree_vis = None

        # retrieve the vis data from the previous frame to avoid the empty visualization
        if 'scen_tree' in self.frames[frame_idx]:
            scen_tree_vis = self.frames[frame_idx]['scen_tree']
        else:
            pre_frame_idx = frame_idx - 1
            while pre_frame_idx >= 0 and 'scen_tree' not in self.frames[pre_frame_idx]:
                pre_frame_idx -= 1
            if pre_frame_idx >= 0 and 'scen_tree' in self.frames[pre_frame_idx]:
                scen_tree_vis = self.frames[pre_frame_idx]['scen_tree']

        if 'traj_tree' in self.frames[frame_idx]:
            traj_tree_vis = self.frames[frame_idx]['traj_tree']
        else:
            pre_frame_idx = frame_idx - 1
            while pre_frame_idx >= 0 and 'traj_tree' not in self.frames[pre_frame_idx]:
                pre_frame_idx -= 1
            if pre_frame_idx >= 0 and 'traj_tree' in self.frames[pre_frame_idx]:
                traj_tree_vis = self.frames[pre_frame_idx]['traj_tree']

        # Clear the previous cube and draw a new one
        range_3d = 15.0
        font_size = 35
        reset_ax(ax)

        # Process the frame
        center = np.array([0, 0])
        center[0] = self.config['render_config']['camera_position']['x']
        center[1] = self.config['render_config']['camera_position']['y']
        cam_yaw = self.config['render_config']['camera_position']['yaw']
        elev = self.config['render_config']['camera_position']['elev']
        ax.set_xlim([center[0] - range_3d, center[0] + range_3d])
        ax.set_ylim([center[1] - range_3d, center[1] + range_3d])
        ax.set_zlim([0, 2 * range_3d])
        ax.view_init(elev=elev, azim=180 + np.rad2deg(cam_yaw))

        draw_map(ax, self.smp.map_data)
        if scen_tree_vis is not None:
            draw_scen_trees(ax, scen_tree_vis)
        if traj_tree_vis is not None:
            draw_traj_trees(ax, traj_tree_vis)

        #  plot agents
        for agent in self.frames[frame_idx]['agents']:
            draw_agent(ax, agent, vis_bbox=False)
            if np.linalg.norm(agent.state[:2] - center) < 2 * range_3d:
                ax.text(agent.state[0], agent.state[1], 1.0, 'No.{}:{:.2f}m/s'.format(agent.id, agent.state[2]),
                           fontsize=font_size)

        # try to retrieve the history of the agent in current frame
        agent_history = dict()
        for agent in self.frames[frame_idx]['agents']:
            agent_history[agent.id] = [agent.state[:2]]

        back_step = 100
        for i in range(1, back_step):
            if frame_idx - i < 0:
                break
            for agent in self.frames[frame_idx - i]['agents']:
                if agent.id in agent_history:
                    agent_history[agent.id].append(agent.state[:2])

        # plot the history of the agent
        for agent_id, history in agent_history.items():
            history.reverse()
            # check length of history
            if np.linalg.norm(history[0] - history[-1]) < 0.1:
                continue
            draw_traj(ax, history)
