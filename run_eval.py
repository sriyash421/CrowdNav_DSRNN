import os
from crowd_nav.configs.config_vecmpc import Config
from evaluation_mpc import main
import sys

controllers = ['sgan', 'csgan', 'cv_noise', 'cv', 'cvkf']

for c in controllers:
    Config.model = c
    Config.save_path = os.path.join('results', c)
    Config.exp_name = c
    sys.stdout = open(os.path.join('results', f'{c}.out'), 'w')
    main()
    sys.stdout.close()