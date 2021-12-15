import argparse
import numpy as np
from numpy.core.fromnumeric import reshape
import open3d as o3d
import os.path as osp
import pyflex
import random
import matplotlib.pyplot as plt
import math
import copy
import time
import os

from collections import deque
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif

from mpl_toolkits.mplot3d import Axes3D

class MakePickPlace:
    def __init__(self, num):
        parser = argparse.ArgumentParser(description='Process some integers.')
        # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
        parser.add_argument('--env_name', type=str, default='ClothFlatten')
        parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
        parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
        parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
        parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')
        parser.add_argument('--num_depth', type=int, default=3, help='Number of depth of bfs')
        parser.add_argument('--threshold_r', type=int, default=0.045, help='R threshold')
        parser.add_argument('--grid_param', type=int, default=100, help='change grid size')

        args = parser.parse_args()

        
        env_kwargs = env_arg_dict[args.env_name]

        # Generate and save the initial states for running this environment for the first time
        env_kwargs['use_cached_states'] = False
        env_kwargs['save_cached_states'] = False
        env_kwargs['num_variations'] = args.num_variations
        env_kwargs['render'] = False#True
        env_kwargs['headless'] = True #args.headless
        env_kwargs['num_picker'] = 1


        if not env_kwargs['use_cached_states']:
            print('Waiting to generate environment variations. May take 1 minute for each variation...')
        self.env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
        # self.env.reset(set_picker_highest_point=True)
        self.env.reset()
        self.args = args
        self.env_kwargs = env_kwargs

        self.threshold_r = args.threshold_r
        self.max_depth = args.num_depth
        self.img_size = args.img_size
        self.grid_param = args.grid_param
        self.num = num

        # ground truth data
        self.nodes = pyflex.get_n_particles()
        self.edges = pyflex.get_spring_indices()
        self.positions = None
        self.velocities = None
        self.graph = [[] for i in range(self.nodes)]
        self.near_graph = [set() for i in range(self.nodes)]

        # observated data
        self.obs = None
        self.reshaped_obs = None
        self.downpcd = None
        self.downpcd_np = None
        self.num_downpcd = None
        self.sorted_obs = None

        # matching data
        self.matching_index = None

        print('finish initialization')

    def process(self):
        # self.env.reset()


        self.make_graph()
        print("make undirected graph done")

        # get the near graph
        for i in range(self.nodes):
            is_visited = [False for i in range(self.nodes)]
            self.bfs(i, is_visited)
        print("get near graph done")

        # proceed one step for stabilize simulation(maybe)
        action = self.env.action_space.sample()
        action[0] = 0
        action[1] = 0
        action[2] = 0
        action[3] = 0
        self.obs, _, _, _ = self.env.step(action)#, record_continuous_video=True, img_size=self.img_size)

        pos = pyflex.get_positions()
        self.positions = np.delete(np.reshape(pos, [-1, 4]), 3, 1)
        self.past_positions = copy.deepcopy(self.positions)

        # prepare downsampled point cloud from obs
        self.downsample_point_cloud()
        print("downsampling point cloud done")

        # do bipartite matching
        self.bipartite_matching()
        print("bipartite matching done")



        



        num_edges = 0
        edge_from = []
        edge_to = []
        edge_connect = []

        min_dist = 100000
        picked_index = None

        for i in range(self.num_downpcd):
            if min_dist > np.linalg.norm(self.downpcd_np[i] - self.sorted_obs[0]):
                picked_index = i
                min_dist = np.linalg.norm(self.downpcd_np[i] - self.sorted_obs[0])

        for i in range(self.num_downpcd):
            for j in range(self.num_downpcd):
                if i == j:
                    continue
                if self.get_dist(i, j) < self.threshold_r:
                    num_edges += 1
                    edge_from.append(i)
                    edge_to.append(j)
                    if self.is_connected(i, j):
                        edge_connect.append(1)
                    else:
                        edge_connect.append(0)
        if not os.path.exists('examples/pickplace_data'):
            os.makedirs('examples/pickplace_data')
        pass_w = 'examples/pickplace_data/ppdata' + str(self.num) + '.txt'



        # frames = [self.env.get_image(self.img_size, self.img_size)]
        with open(pass_w, 'w') as f:
            f.write(str(self.num_downpcd) + ' ' + str(num_edges) + '\n')
            for i in range(num_edges):
                f.write(str(edge_from[i]) + ' ')
            f.write('\n')
            for i in range(num_edges):
                f.write(str(edge_to[i]) + ' ')
            f.write('\n')
            for i in range(num_edges):
                f.write(str(edge_connect[i]) + ' ')
            f.write('\n')
            step_size_x = random.uniform(-0.5, 0.5)
            step_size_y = random.uniform(0, 0.5)
            step_size_z = random.uniform(-0.5, 0.5)
            step_size_dist = random.uniform(0.15, 0.4)

            for i in range(1):
                action = self.env.action_space.sample()
                action[0] = 0
                action[1] = 0
                action[2] = 0
                action[3] = 0
                self.obs, _, _, info = self.env.step(action)#, record_continuous_video=True, img_size=self.img_size)

                # frames.extend(info['flex_env_recorded_frames'])
                pos = pyflex.get_positions()
                self.positions = np.delete(np.reshape(pos, [-1, 4]), 3, 1)
                for j in range(self.num_downpcd):
                    is_picked = [0, 0]
                    if j == picked_index:
                        is_picked[0] = 1
                    else:
                        is_picked[1] = 1
                    # f.write(str(self.positions[self.matching_index[j]][0]*1000)[:14]+' '
                    #         +str(self.positions[self.matching_index[j]][1]*1000)[:14]+' '
                    #         +str(self.positions[self.matching_index[j]][2]*1000)[:14]+' '
                    #         +str(is_picked[0])+' '+str(is_picked[1])+'\n')



            self.env.set_action_tool_random_high(self.downpcd_np, self.positions, self.matching_index)


            for i in range(60):
                action = self.env.action_space.sample()
                action[0] = step_size_x
                action[1] = step_size_y
                action[2] = step_size_z
                move_dist = step_size_dist
                alpha = move_dist/math.sqrt(action[0]**2 + action[1]**2 + action[2]**2)

                action[0] *= alpha
                action[1] *= alpha
                action[2] *= alpha 
                action[3] = 1
                obs, _, _, info = self.env.step(action)#, record_continuous_video=True, img_size=self.img_size)
                # frames.extend(info['flex_env_recorded_frames'])
                pos = pyflex.get_positions()
                vel = pyflex.get_velocities()
                self.positions = np.delete(np.reshape(pos, [-1, 4]), 3, 1)
                self.velocities = np.reshape(vel, [-1, 3])
                for j in range(self.num_downpcd):
                    is_picked = [0, 0]
                    if j == picked_index:
                        is_picked[0] = 1
                    else:
                        is_picked[1] = 1
                    f.write(str(self.positions[self.matching_index[j]][0])[:14] + ' '
                            + str(self.positions[self.matching_index[j]][1])[:14] + ' '
                            + str(self.positions[self.matching_index[j]][2])[:14] + ' '
                            + str(self.velocities[self.matching_index[j]][0])[:14] + ' '
                            + str(self.velocities[self.matching_index[j]][1])[:14] + ' '
                            + str(self.velocities[self.matching_index[j]][2])[:14] + ' '
                            + str(is_picked[0]) + ' ' + str(is_picked[1]) + '\n')
            for i in range(40):
                action = self.env.action_space.sample()
                action[0] = 0
                action[1] = 0
                action[2] = 0
                move_dist = 0
                alpha = 0 #move_dist/math.sqrt(action[0]**2 + action[1]**2 + action[2]**2)
                action[0] *= alpha
                action[1] *= alpha
                action[2] *= alpha
                action[3] = 0
                obs, _, _, info = self.env.step(action)#, record_continuous_video=True, img_size=self.img_size)
                # frames.extend(info['flex_env_recorded_frames'])
                pos = pyflex.get_positions()
                vel = pyflex.get_velocities()
                self.positions = np.delete(np.reshape(pos, [-1, 4]), 3, 1)
                self.velocities = np.reshape(vel, [-1, 3])
                for j in range(self.num_downpcd):
                    is_picked = [0, 1]
                    f.write(str(self.positions[self.matching_index[j]][0])[:14] + ' '
                            + str(self.positions[self.matching_index[j]][1])[:14] + ' '
                            + str(self.positions[self.matching_index[j]][2])[:14] + ' '
                            + str(self.velocities[self.matching_index[j]][0])[:14] + ' '
                            + str(self.velocities[self.matching_index[j]][1])[:14] + ' '
                            + str(self.velocities[self.matching_index[j]][2])[:14] + ' '
                            + str(is_picked[0]) + ' ' + str(is_picked[1]) + '\n')

        # self.save_video_dir = "./data/"
        # if not osp.exists(self.save_video_dir):
        #     os.makedirs('data')
        # if self.save_video_dir is not None:
        #     save_name = osp.join(self.save_video_dir, 'make_data_random-highest.gif')
        #     save_numpy_as_gif(np.array(frames), save_name, fps=20)
        #     print('Video generated and save to {}'.format(save_name))


    def make_graph(self):
        # make undirected graph
        for i in range(self.edges.shape[0]//2):
            self.graph[self.edges[i*2]].append(self.edges[i*2+1])
            self.graph[self.edges[i*2+1]].append(self.edges[i*2])
        for i in range(self.nodes):
            self.graph[i] = list(set(self.graph[i]))
        return

    def bfs(self, start_node, is_visited):
        que = deque()
        is_visited[start_node] = True
        que.append((start_node, 0))
        while len(que):
            tup_q = que.pop()
            current_node = tup_q[0]
            current_depth = tup_q[1]
            if current_depth >= self.max_depth:
                continue
            for next_node in self.graph[current_node]:
                if not is_visited[next_node]:
                    is_visited[next_node] = True
                    self.near_graph[start_node].add(next_node)
                    que.append((next_node, current_depth+1))
        return
    
    def downsample_point_cloud(self):
        # prepare downsampled point cloud from obs
        self.reshaped_obs = np.reshape(self.obs, [-1, 3])

        min_x = 1000*self.grid_param
        max_x = -1000*self.grid_param
        min_y = 1000*self.grid_param
        max_y = -1000*self.grid_param
        for r in self.reshaped_obs:
            min_x = min(min_x, r[0]*self.grid_param)
            max_x = max(max_x, r[0]*self.grid_param)
            min_y = min(min_y, r[2]*self.grid_param)
            max_y = max(max_y, r[2]*self.grid_param)

        min_x = math.floor(min_x)
        max_x = math.ceil(max_x)
        min_y = math.floor(min_y)
        max_y = math.ceil(max_y)

        data = [[np.zeros(3) for i in range(max_y-min_y)] for j in range(max_x-min_x)]
        used = np.zeros((max_x-min_x, max_y-min_y), dtype=bool)

        ord = np.argsort(self.reshaped_obs[:,1])[::-1]
        sorted_obs = self.reshaped_obs[ord]
        self.sorted_obs = sorted_obs
        
        for r in sorted_obs:
            x = int(r[0]*self.grid_param) - min_x
            y = int(r[2]*self.grid_param) - min_y
            if not used[x][y]:
                data[x][y] = r
                used[x][y] = True
        
        self.num_pcd = np.sum(used)
        self.pcd_np = np.zeros((self.num_pcd, 3))
        cnt = 0
        for i in range(used.shape[0]):
            for j in range(used.shape[1]):
                if used[i][j]:
                    self.pcd_np[cnt] = data[i][j]
                    cnt += 1

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcd_np)
        v_size = 0.0216#*1.44375
        self.downpcd = pcd.voxel_down_sample(voxel_size=v_size)#(voxel_size=0.035)
        self.downpcd_np = np.asarray(self.downpcd.points)
        self.num_downpcd = self.downpcd_np.shape[0]
        # o3d.visualization.draw_geometries([self.downpcd])


    def bipartite_matching(self):
        # do bipartite matching?
        # maybe this is enough

        self.matching_index = [None for i in range(self.num_downpcd)]
        for i in range(self.num_downpcd):
            lis = []
            
            for j in range(self.nodes):
                lis.append((np.linalg.norm(self.downpcd_np[i] - self.positions[j]), j))

            lis.sort(key=lambda x: x[0])
            self.matching_index[i] = lis[0][1]

        return



    def is_connected(self, x, y):
        if self.matching_index[x] == self.matching_index[y]:
            return True
        if self.matching_index[x] in self.near_graph[self.matching_index[y]]:
            return True
        return False

    def get_dist(self, x, y):
        return np.linalg.norm(self.downpcd_np[x] - self.downpcd_np[y])

    def plot_obs(self):
        fig = plt.figure()
        ax = Axes3D(fig)

        scax = []
        scay = []
        scaz = []

        for r in self.downpcd_np:
            scax.append(r[0])
            scay.append(r[2])
            scaz.append(r[1])

        ax.scatter(scax, scay, scaz, s=3)
        plt.show()

    def visualize_graph(self, edge=True):
        fig = plt.figure()
        ax = Axes3D(fig)

        pos = pyflex.get_positions()
        pos = np.reshape(pos, [-1, 4])

        point_list = [self.matching_index[i] for i in range(self.num_downpcd)]


        scax = []
        scay = []
        scaz = []

        for r in point_list:
            scax.append(pos[r][0])
            scay.append(pos[r][1])
            scaz.append(pos[r][2])
            if not edge:
                continue
            for x in self.near_graph[r]:
                if x in point_list:
                    ax.plot([pos[r][0], pos[x][0]], [pos[r][1], pos[x][1]], [pos[r][2], pos[x][2]], linestyle="solid")

        ax.scatter(scax, scay, scaz, s=3)
        plt.show()

    def calc_reward(self, downsampled_pc):
        return null

if __name__ == '__main__':
    for i in range(1052, 2000):
        mp = MakePickPlace(i)
        mp.process()