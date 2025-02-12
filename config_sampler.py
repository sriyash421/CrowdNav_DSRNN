import numpy as np
from crowd_nav.configs.config import Config, BaseConfig


class SimpleConfigSampler():
    def __init__(self):
        self.random = np.random.RandomState()

    def sample(self, seed):
        self.random = np.random.RandomState(seed=seed)
        
        config = Config()
        config.humans.policy = "mix"
        config.robot.policy = config.humans.policy
        config.robot.radius = config.humans.radius
        config.robot.v_pref = config.humans.v_pref
        config.robot.visible = True

        # config for ORCA
        config.orca.neighbor_dist = self.random.uniform(0, 10)
        config.orca.safety_space = self.random.uniform(0, 1)
        config.orca.time_horizon = self.random.uniform(0, 10)
        config.orca.time_horizon_obst = self.random.uniform(0, 10)

        # social force
        config.sf.A = self.random.uniform(0, 10)  # Strenght of interaction
        config.sf.B = self.random.uniform(0, 10) + 1e-8  # Range of interaction
        config.sf.KI = self.random.uniform(0, 10) + 1e-8  # Goal Strength

        config.noise.add_noise = True
        # uniform, gaussian
        config.noise.type = "uniform"
        config.noise.magnitude = self.random.uniform(0, 0.2)
        return config


class GaussianConfigSampler():
    def __init__(self, scale=0.5, p=0.5):
        self.sc = scale
        self.p = p
        self.random = np.random.RandomState()

    def sample(self, seed):
        self.random = np.random.RandomState(seed=seed)

        config = Config()
        config.humans.policy = "mix"
        config.robot.policy = config.humans.policy
        config.robot.radius = config.humans.radius
        config.robot.v_pref = config.humans.v_pref
        config.robot.visible = True

        # config for ORCA
        config.orca = BaseConfig()
        config.orca.neighbor_dist = 10 * \
            (1 + self.random.uniform(-self.sc, self.sc))
        config.orca.safety_space = 0.15 * \
            (1 + self.random.uniform(-self.sc, self.sc))
        config.orca.time_horizon = 5 * \
            (1 + self.random.uniform(-self.sc, self.sc))
        config.orca.time_horizon_obst = 5 * \
            (1 + self.random.uniform(-self.sc, self.sc))

        # social force
        config.sf = BaseConfig()
        # Strenght of interaction
        config.sf.A = 2. * (1 + self.random.uniform(-self.sc, self.sc))
        # Range of interaction
        config.sf.B = 1 * (1 + self.random.uniform(-self.sc, self.sc))
        # Goal Strength
        config.sf.KI = 1 * (1 + self.random.uniform(-self.sc, self.sc))

        config.noise = BaseConfig()
        config.noise.add_noise = self.random.uniform(0, 1) < self.p
        # uniform, gaussian
        config.noise.type = "uniform"
        config.noise.magnitude = 0.1 * \
            (1 + self.random.uniform(-self.sc, self.sc))

        return config


class FullConfigSampler():
    def __init__(self, scale=0.5, p=0.5):
        self.sc = scale
        self.p = p
        self.random = np.random.RandomState()

    def sample(self, seed):
        self.random = np.random.RandomState(seed=seed)

        config = Config()

        config.sim.train_val_sim = "circle_crossing" if self.random.uniform(
            0, 1) < self.p else 'square_crossing'
        config.sim.test_sim = "circle_crossing" if self.random.uniform(
            0, 1) < self.p else 'square_crossing'
        config.sim.square_width = self.random.randint(5, 15)
        config.sim.circle_radius = self.random.randint(2, 10)

        config.humans.policy = self.random.choice("orca", "social_force", "mix")
        config.humans.radius = 0.3 * (1 + self.random.uniform(-self.sc, self.sc))
        config.humans.v_pref = 1 * (1 + self.random.uniform(-self.sc, self.sc))
        config.robot.policy = self.humans.policy
        config.robot.radius = self.humans.radius
        config.robot.v_pref = self.humans.v_pref
        config.robot.visible = True

        # a human may change its goal before it reaches its old goal
        config.humans.random_goal_changing = self.random.uniform(0, 1) < self.p
        config.humans.goal_change_chance = 0.25 * \
            (1 + self.random.uniform(-self.sc, self.sc))

        # a human may change its goal after it reaches its old goal
        config.humans.end_goal_changing = self.random.uniform(0, 1) < self.p
        config.humans.end_goal_change_chance = 1.0 * \
            (1 + self.random.uniform(-self.sc, self.sc))

        # a human may change its radius and/or v_pref after it reaches its current goal
        config.humans.random_radii = self.random.uniform(0, 1) < self.p
        config.humans.random_v_pref = self.random.uniform(0, 1) < self.p

        # one human may have a random chance to be blind to other agents at every time step
        config.humans.random_unobservability = self.random.uniform(0, 1) < self.p
        config.humans.unobservable_chance = 0.3 * \
            (1 + self.random.uniform(-self.sc, self.sc))

        config.humans.random_policy_changing = self.random.uniform(0, 1) < self.p

        # config for ORCA
        config.orca = BaseConfig()
        config.orca.neighbor_dist = 10 * \
            (1 + self.random.uniform(-self.sc, self.sc))
        config.orca.safety_space = 0.15 * \
            (1 + self.random.uniform(-self.sc, self.sc))
        config.orca.time_horizon = 5 * \
            (1 + self.random.uniform(-self.sc, self.sc))
        config.orca.time_horizon_obst = 5 * \
            (1 + self.random.uniform(-self.sc, self.sc))

        # social force
        config.sf = BaseConfig()
        config.sf.A = 2. * (1 + self.random.uniform(-self.sc, self.sc))
        config.sf.B = 1 * (1 + self.random.uniform(-self.sc, self.sc))
        config.sf.KI = 1 * (1 + self.random.uniform(-self.sc, self.sc))

        config.noise = BaseConfig()
        config.noise.add_noise = self.random.uniform(0, 1) < self.p
        # uniform, gaussian
        config.noise.type = "uniform"
        config.noise.magnitude = 0.1 * \
            (1 + self.random.uniform(-self.sc, self.sc))

        return config

def load_config(args):
    config = Config()
    for k, v in args.items():
        exec(f'config.{k} = v')
    
    return config