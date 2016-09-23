import scoring

import numpy as np
import pandas as pd

from ggplot import *

dirs = ['test1', 'test2', 'test3']

res = {}
gs = {}

episodes = {}
rewards = {}

count = 0
for d in dirs:
	res[d] = scoring.score_from_local(d)
	eplens = res[d]['episode_lengths']
	eprews = res[d]['episode_rewards']
	timestamps = res[d]['timestamps']
	irt = res[d]['initial_reset_timestamp']
	buckets = 100
	gs[d] = scoring.compute_graph_stats(eplens, eprews, timestamps, irt, buckets)
	episodes[count] = gs[d]['x_episode_y_reward']['y']
	rewards[count] = gs[d]['x_episode_y_reward']['y']
	count += 1

a = pd.DataFrame.from_dict(rewards, orient='index')
print(ggplot(a,aes('date','beef * 2000')))

# red dashes, blue squares and green triangles
# plt.plot(lines[0]['x'], lines[0]['y'])
# plt.show()