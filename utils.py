from concurrent.futures.thread import ThreadPoolExecutor

def parallel_run(func, args, num_threads):
    """Runs func in parallel with the given args and returns an ordered list with the returned values."""
    # Assert list of lists to comply with variadic positional arguments (i.e. the * in fn(*args))
    assert all([isinstance(arg, list) for arg in args]), 'Function arguments must be given as list'
    assert callable(func), 'func must be a callable function'
    # Define output variable and load function wrapper to maintain correct list order
    results = [None] * len(args)
    def _run_load_func(n_, args_):
        results[n_] = func(*args_)
    # Parallel run func and store the results in the right place
    pool = ThreadPoolExecutor(max_workers=num_threads)
    future_tasks = [pool.submit(_run_load_func, n, args) for n, args in enumerate(args)]
    # Check if any exceptions occured during execution
    [future_task.result() for future_task in future_tasks]
    pool.shutdown(wait=True)
    return results


if __name__ == '__main__':
    print(parallel_run(lambda x: x/2, [[1], [2], [3]], num_threads=3))