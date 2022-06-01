import os
from crowd_nav.configs.config_vecmpc import Config
from evaluation_mpc import main_config
import sys

controllers = ['sgan', 'csgan', 'cv_noise', 'cv', 'cvkf']

for c in controllers:
    config = Config(c)
    config.save_path = os.path.join('results', c)
    config.exp_name = c
    # sys.stdout = open(os.path.join('results', f'{c}.out'), 'w')
    main_config(config)
    # sys.stdout.close()