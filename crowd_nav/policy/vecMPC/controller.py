#!/usr/bin/env python
from time import time
import numpy as np
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.vecMPC.logger import BaseLogger
from .predictors import *

class vecMPC(Policy):

    def __init__(self, config):
        super(vecMPC, self).__init__(config)

        self.model_predictor = eval(config.MPC["model_predictor"])()
        self.logger = BaseLogger(config.save_path, config.model, config.exp_name)
        self.sim_heading = None
        self.trajectory = None
        self.model_predictor.vpref = config.robot.v_pref
        self.model_predictor.dt = config.env.time_step
        self.action_params = config.MPC['params']['action']
        self.span = np.deg2rad(self.action_params['span'])
        self.n_actions = self.action_params["n_actions"]
        if self.span == 2 * np.pi:
            self.span = 2 * np.pi - ((2 * np.pi) / self.n_actions)
        self.model_predictor.set_params(config.MPC['params'])
        self.logger.add_params(config.MPC['params'])

        self.name = 'vecmpc'
    
    def reset(self):
        self.trajectory = None
        self.sim_heading = None
        self.logger.reset()
        self.model_predictor.reset()
    
    @staticmethod
    def flatten_state(state):
        px, py = state.get_position()
        vx, vy = state.get_velocity()
        return np.array([px, py, vx, vy, state.radius])

    def update_trajectory(self, state):
        arr = [self.flatten_state(state.self_state)]
        for state in state.human_states:
            arr.append(self.flatten_state(state))
        arr = np.stack(arr, axis=0)
        self.trajectory = np.concatenate((self.trajectory, arr[None]), axis=0) if self.trajectory is not None else arr[None]
        return arr

    # noinspection PyAttributeOutsideInit
    def predict(self, state):
        start_t = time()
        self.pathbot_state = state.self_state
        self.ego_state = state.self_state

        # Initialize sim heading for MPC if first iteration
        if self.sim_heading is None:
            self.sim_heading = self.pathbot_state.get_heading()
        else:
            # Set the joint state sim_heading for rollouts
            self.pathbot_state.set_sim_heading(self.sim_heading)

        array_state = self.update_trajectory(state)
        self.logger.update_trajectory(self.trajectory, 0, 0, time())

        if self.reach_destination(state):
            return self.action_post_processing([0,0], start_t)

        goal = np.array(self.pathbot_state.get_goal())
        action_set = self.generate_action_set(self.pathbot_state, self.trajectory, goal)
        # (N x S x T' x H x 2), (N x S), (2 x 1)
        predictions, costs, action_set, best_action = self.model_predictor.predict(self.trajectory, array_state, action_set, goal)

        self.logger.add_predictions(array_state, action_set, predictions, costs, goal)
        return self.action_post_processing(best_action[2:], start_t)
        
    def action_post_processing(self, action, start_t):
        global_action_xy = ActionXY(action[0], action[1])
        # Keep in global frame ActionXY
        action_xy = global_action_xy

        # Only update heading if we take an action
        if np.linalg.norm(action) > 0:
            self.sim_heading = np.arctan2(action[1], action[0])
        else:
            # Randomly set heading to get us unstuck if not moving
            if np.linalg.norm(self.pathbot_state.get_velocity()) < 0.1:
                self.sim_heading = np.random.rand() * 2 * np.pi
        self.logger.add_action(action)
        self.logger.add_time(time()-start_t)
        return action_xy

    def generate_action_set(self, state, trajectory, goal):
        """To get actions"""
        # self.action_params["type"] # TODO: to use multiple actions
        # Get current (global) state info for ego agent
        pos = np.array(state.get_position())

        sim_heading = state.get_sim_heading()
        thetas = [sim_heading-(self.span / 2.0) + i * self.span / (self.n_actions - 1) for i in range(self.n_actions)]
        thetas = thetas if len(thetas) > 1 else np.arctan2(goal-pos)
        vpref = self.pathbot_state.get_vpref()
        goals = pos[:, None] + (vpref * self.model_predictor.prediction_horizon *10) * np.stack((np.cos(thetas), np.sin(thetas)), axis=0) # (2, N)

        return self.generate_cv_rollout(pos, goals, vpref, self.model_predictor.rollout_steps)

    def generate_cv_rollout(self, position, goals, vpref, length):
        """To get particular rollout"""
        rollouts = []
        position = np.repeat(position[:, None], goals.shape[1], axis=1)  # (2, N)
        for _ in range(length):
            ref_velocities = self.generate_cv_action(position, goals, vpref) # (2, N)
            position, _ = self.step_dynamics(position, ref_velocities)
            rollouts.append( np.concatenate((position, ref_velocities), axis=0).transpose((1, 0)))
        rollouts = np.stack(rollouts, axis=1) # N x T x 4
        return rollouts
    
    def generate_cv_action(self, position, goal, vpref):
        """To get particular action"""
        dxdy = goal-position # (2, N)
        thetas = np.arctan2(dxdy[1], dxdy[0]) # (N, )
        return np.stack((np.cos(thetas), np.sin(thetas)), axis=0) * vpref # (2, N)

    def step_dynamics(self, position, action): # (2, N), (2, N)
        position = position +  action * self.model_predictor.dt # (2, N)
        return position, action