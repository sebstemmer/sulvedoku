from solve.solve import create_filled, random_guess_strategy
import time
import numpy as np
import matplotlib.pyplot as plt

num_executions: int = 500

max_depths = list(range(81, 95)) + list(range(95, 250, 5)) + list(range(250, 2000, 100))
times = np.zeros((len(max_depths), num_executions))

for max_depth_idx, max_depth in enumerate(max_depths):
    for k in range(num_executions):
        start = time.perf_counter()
        filled = create_filled(
            max_depth=max_depth,
            guess_strategy=random_guess_strategy
        )
        times[max_depth_idx, k] = time.perf_counter() - start

    print("max_depth: ", max_depth)
    print(f"per grid: {1000 * times.mean(axis=1)[max_depth_idx]} ms")
    print(f"std: {1000 * times.std(axis=1)[max_depth_idx]} ms")

plt.plot(max_depths, 1000 * times.mean(axis=1), 'x')

plt.xlabel("max depth")
plt.ylabel("average creation time per grid in ms")

plt.title("find optimal max depth")

plt.show()