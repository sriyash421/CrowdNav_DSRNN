import os
import pickle
import pandas as pd
import numpy as np
import scipy.spatial.distance
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.state import *


class WrapperEnv(CrowdSim):
    def __init__(self, seed, num):
        super().__init__()
        self.thisSeed = seed+num
        self.nenv = num

    def generate_full_ob(self):
        obs = [human.get_full_state() for human in self.humans]
        obs.append(self.robot.get_full_state())

        return obs


def get_metrics(observations, actions, reward):
    dists = get_distances(observations)
    return dict(returns=sum(reward),
                avg_dist=np.mean(dists),
                min_dist=np.min(dists),
                avg_min_dist=np.mean(np.min(dists, axis=1))
                )


def get_distances(observations):
    dists = list()
    for obs in observations:
        obs = np.array([[h.px, h.py] for h in obs])
        dist = scipy.spatial.distance.pdist(obs, 'euclidean')
        dists.append(dist.flatten())
    return np.array(dists)


class Trajectory(object):
    def __init__(self, config):
        self.config = config
        self.observations = list()
        self.actions = list()
        self.rewards = list()

    def add_step(self, obs, act, reward):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(reward)


class RolloutStorage(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.episode_num = 0

    def add_traj(self, traj):
        data = dict(
            observations=traj.observations,
            actions=traj.actions,
            rewards=traj.rewards,
            length=len(traj.observations),
            **get_metrics(traj.observations, traj.actions, traj.rewards),
            **traj.config.__dict__()
        )
        with open(os.path.join(self.output_dir, "trajectories", f"{self.episode_num}.pkl"), "wb") as fout:
            pickle.dump(data, fout)
        data.pop('observations')
        data.pop('actions')
        data.pop('rewards')
        data['episode_num'] = self.episode_num

        df = pd.DataFrame([data])
        df.to_csv(os.path.join(self.output_dir, "data.csv"),
                  mode='w' if self.episode_num == 0 else 'a', header=(self.episode_num == 0))

        self.episode_num += 1
