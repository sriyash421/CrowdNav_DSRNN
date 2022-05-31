import numpy as np
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

class SOCIAL_FORCE(Policy):
    def __init__(self, config):
        super().__init__(config)
        _steps_array = np.load(config.noncoop.file)

        self.mean_positions = _steps_array[:, :2]
        self.std_positions = _steps_array[:, 2:]


    def predict(self, state):
        """
        Produce action for agent with circular specification of social force model.
        """
        px, py = state.self_state.px, state.self_state.py

        distances = np.linalg.norm(self.mean_positions-distances, axis=1)
        min_point = np.argmin(distances)
        if min_point+1 < self.std_positions.shape[0]:
            nx, ny = np.random.sample(self.mean_positions[min_point+1], self.std_positions[min_point+1])
        else:
            nx, ny = state.self_state.gx, state.self_state.gy
        
        new_vx, new_vy = (nx-px)/self.config.env.time_step, (ny-py)/self.config.env.time_step
        act_norm = np.linalg.norm([new_vx, new_vy])
        if act_norm > state.self_state.v_pref:
            return ActionXY(new_vx / act_norm * state.self_state.v_pref, new_vy / act_norm * state.self_state.v_pref)
        else:
            return ActionXY(new_vx, new_vy)
