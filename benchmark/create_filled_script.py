from solve.path import create_filled, random_guess_strategy
import time
import sys

sys.setrecursionlimit(int(1e9))


for i in range(-1, 4):
    start = time.perf_counter()
    for k in range(100):
        filled = create_filled(
            max_go_back_depth=i,
            guess_strategy=random_guess_strategy
        )
    end = time.perf_counter()
    print(f"{i}: {1000 * (end - start)} ms")