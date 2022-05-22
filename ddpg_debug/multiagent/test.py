import numpy as np
p_pos1 = np.array([2,3])
p_pos2 = np.array([3,4])
delta_pos = p_pos1 - p_pos2
a = np.square(delta_pos)
print(a)