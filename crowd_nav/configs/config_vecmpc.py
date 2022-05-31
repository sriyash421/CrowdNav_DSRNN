from genericpath import exists
import os
import yaml
import numpy as np

class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    # environment settings
    env = BaseConfig()
    env.env_name = 'CrowdSimDict-v0'  # name of the environment
    env.time_limit = 50 # time limit of each episode (second)
    env.time_step = 0.1 # length of each timestep/control frequency (second)
    env.val_size = 100
    env.test_size = 100 # number of episodes for test.py
    env.randomize_attributes = True # randomize the preferred velocity and radius of humans or not
    env.seed = 0  # random seed for environment

    # reward function
    reward = BaseConfig()
    reward.success_reward = 10
    reward.collision_penalty = -20
    # discomfort distance for the front half of the robot
    reward.discomfort_dist_front = 0.25
    # discomfort distance for the back half of the robot
    reward.discomfort_dist_back = 0.25
    reward.discomfort_penalty_factor = 10
    reward.gamma = 0.99  # discount factor for rewards

    # environment settings
    sim = BaseConfig()
    sim.render = False # show GUI for visualization
    sim.circle_radius = 6 # radius of the circle where all humans start on
    sim.human_num = 3 # total number of humans
    # Group environment: set to true; FoV environment: false
    sim.group_human = False

    # human settings
    humans = BaseConfig()
    humans.visible = True # a human is visible to other humans and the robot
    # policy to control the humans: orca or social_force
    humans.policy = "orca"
    humans.radius = 0.2 # radius of each human
    humans.v_pref = 0.8 # max velocity of each human
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    humans.random_goal_changing = False
    humans.goal_change_chance = 0.25

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = False
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # robot settings
    robot = BaseConfig()
    robot.visible = True  # the robot is visible to humans
    # robot policy: srnn for now
    robot.policy = 'vecmpc'
    robot.radius = 0.3  # radius of the robot
    robot.v_pref = 0.8  # max velocity of the robot
    # robot FOV = this values * PI
    robot.FOV = 2.

    # add noise to observation or not
    noise = BaseConfig()
    noise.add_noise = True
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    # robot action type
    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"


    model = "sgan"
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params", f"{model}.yaml")
    if not exists(path):
        raise FileNotFoundError
    with open(path, "r") as fin:
        MPC = yaml.safe_load(fin)
    print(MPC)
    MPC['params']['dt'] = 0.2
    MPC['params']['prediction_length'] = 1.2

    save_path = "test"
    exp_name = "test"

    test_setting = "cooperative"