import time

import matplotlib.pyplot as plt
import numpy as np

from create.create import create_grid

num_executions: int = 100

num_filled_target: int = 23

max_depths = list(range(75, 100, 5)) + list(range(100, 250, 10)) + list(range(250, 750, 50))
# max_depths = range(100, 500, 20)
# max_depths = list(range(81, 95)) + list(range(95, 250, 5)) + list(range(250, 2000, 100))
times = np.zeros((len(max_depths), num_executions))

for max_depth_idx, max_depth in enumerate(max_depths):
    for k in range(num_executions):
        start = time.perf_counter()
        _ = create_grid(
            num_filled_target=num_filled_target,
            max_remove_depth=max_depth
        )
        times[max_depth_idx, k] = time.perf_counter() - start

    print("max_depth: ", max_depth)
    print(f"per grid: {1000 * times.mean(axis=1)[max_depth_idx]} ms")
    print(f"std: {1000 * times.std(axis=1)[max_depth_idx]} ms")

plt.plot(max_depths, 1000 * times.mean(axis=1), 'x')

plt.xlabel("max remove depth")
plt.ylabel("average creation time per grid in ms")

plt.title("num_clues = 23, find optimal max remove depth")

plt.show()
