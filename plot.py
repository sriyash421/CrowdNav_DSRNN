import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import argparse
import os
import pathlib

def smooth(vals, window):
    '''Smooths values using a sliding window.'''

    if window > 1:
        if window > len(vals):
            window = len(vals)
        y = np.ones(window)
        x = vals
        z = np.ones(len(vals))
        mode = 'same'
        vals = np.convolve(x, y, mode) / np.convolve(z, y, mode)
	# vals = np.clip(vals, -50, 50)
    return vals

parser = argparse.ArgumentParser()
parser.add_argument('--paths', nargs='+', default=[])
parser.add_argument('--window', default=5)
args = parser.parse_args()

paths = args.paths


log_paths = []
for path in paths:
	if os.path.isdir(path):
		log_paths.extend(pathlib.Path(path).rglob('progress.*'))
	elif path[-12:-3] == 'progress.':
		log_paths.append(path)

log_list = []
legends = []
for path in log_paths:
	sub_path, file = os.path.split(path)
	if file == 'progress.csv':
		log_list.append(pd.read_csv(path))
		legends.append(sub_path.split(os.sep)[-1])

# legends = ['DS-RNN', 'DS-RNN-BallBot-zeros', 'DS-RNN-BallBot-random']

# # add more training curves by directory name here!
# log_list = [pd.read_csv("data/linear_dynamics/progress.csv"),
# 			pd.read_csv("data/ballbot_dynamics_zeros/progress.csv"),
# 			pd.read_csv("data/ballbot_dynamics_random/progress.csv"),
# 		   ]

logDicts = {}
for i in range(len(log_list)):
	logDicts[i] = log_list[i]

# graphDicts={0:'eprewmean', 1:'loss/value_loss'}
graphDicts={0:'eprewmean'}

legendList=[]
# summarize history for accuracy

# for each metric
for i in range(len(graphDicts)):
	plt.figure(i)
	plt.title(graphDicts[i])
	j = 0
	for key in logDicts:
		if graphDicts[i] not in logDicts[key]:
			continue
		else:
			plt.plot(logDicts[key]['misc/total_timesteps'],smooth(logDicts[key][graphDicts[i]], args.window))

			legendList.append(legends[j])
			print('avg', str(key), graphDicts[i], np.average(logDicts[key][graphDicts[i]]))
		j = j + 1
	print('------------------------')

	plt.xlabel('total_timesteps')
	plt.legend(legendList, loc='best')
	legendList=[]



plt.show()


