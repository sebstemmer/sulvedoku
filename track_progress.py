import time


def format_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def track(
        idx: int,
        total: int,
        output_every: int,
        counter
):
    if idx % output_every == 0:
        elapsed = time.perf_counter() - counter
        progress = idx / total
        if progress > 0:
            eta = elapsed / progress * (1 - progress)
        else:
            eta = 0
        print(f"{progress * 100:5.1f}% | elapsed {format_time(elapsed)} | ETA {format_time(eta)}")
