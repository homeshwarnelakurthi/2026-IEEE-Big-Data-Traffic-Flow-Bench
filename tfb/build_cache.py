"""Build the dense array cache for every panel, split and view."""
from concurrent.futures import ProcessPoolExecutor

from common import SPLITS, build_cache, panels


def job(args):
    panel, split, masked = args
    return str(build_cache(panel, split, masked))


if __name__ == "__main__":
    jobs = [(p, s, m) for p in panels() for s in SPLITS for m in ((False, True) if s == "train" else (True,))]
    with ProcessPoolExecutor(4) as ex:
        for out in ex.map(job, jobs):
            print(out, flush=True)
