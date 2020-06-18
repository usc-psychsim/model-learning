import itertools
from multiprocessing.pool import Pool

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def get_pool_and_map(processes, star=False):
    """
    Returns a process pool and mapping function, or a single-process mapping function, depending on the number of
    requested processes.
    :param int processes: number of processes to use. `<=0` indicates all cores available, `1` uses single process.
    :param bool star: whether to return a starmap function instead of single-argument map.
    :rtype: (Pool, Callable)
    :return: a tuple (pool, mapping_func) containing the pool object (or None) and mapping function.
    """
    # selects mapping function according to number of requested processes
    if processes == 1:
        pool = None
        map_func = itertools.starmap if star else map
    else:
        pool = Pool(processes if processes > 0 else None)
        map_func = pool.starmap if star else pool.map
    return pool, map_func
