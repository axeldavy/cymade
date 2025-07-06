This package is meant to be a list of utilities implemented in fast Cython code.

Currently implemented are:

# Atomics

Atomics are a useful tool in parallel programming. The fastest way to replicate this
feature with the
standard Python library is to use a threading.Lock() to increment/decrement your variable.

This package provides several very fast atomics, including a counter that
calls an optional callback when it reaches zero.

Available atomics are:
- U64: for counters. Does support calling a callback when it reaches zero
- I64: integers
- Int: Integers with arbitrary precision (like Python's `int` ). I64 should be enough for 99.9% of needs
- Float: floating point

## Python 3.13.5t, 4 × Intel® Core™ i5-4690 CPU @ 3.50GHz, Arch Linux

| Test Case                                    | cymade.atomic   | threading.Lock   | Normal                      | Speedup   |
|:---------------------------------------------|:----------------|:-----------------|:----------------------------|:----------|
| Single-thread increment (1M)                 | 0.038314s       | 0.337002s        | 0.039194s                   | 8.80x     |
| Multi-thread increment (4 threads, 1M total) | 0.138783s       | 0.597176s        | 0.096429s (not thread-safe) | 4.30x     |
| Float operations (1M)                        | 0.055080s       | 0.575025s        | 0.049287s                   | 10.44x    |
| Multi-thread float ops (4 threads, 1M total) | 0.203577s       | 1.003171s        | 0.162361s (not thread-safe) | 4.93x     |
| High contention (20 threads, 2M total)       | 0.290452s       | 1.100565s        | 0.176304s (not thread-safe) | 3.79x     |
| On-zero callback (100K)                      | 0.027623s       | 0.078574s        | N/A (no callback support)   | 2.84x     |
| Event toggle (1M)                            | N/A             | 0.596541s        | N/A                         | N/A       |

Explanation of the results:
- Python's `int` and `float` do not have thread-safety guarantee. Cymade's items do add an overhead,
    but it is small.
- Cymade's atomic items are significantly faster than using items protected by threading's Lock. 

# ThreadPool

- Do you want a very fast thread pool ?

- Do you want to be able to execute a high priority job before already scheduled ones ?

- Do you want to be able to execute low priority jobs in the background if your workers
    have nothing left to do ?

This package has got you covered.

By implementing the whole pool in Cython, we are able to achieve significantly lower overhead
while introducing the job priority mechanism.

Benchmarks speak for themselves:

## Python 3.13.5t, 4 × Intel® Core™ i5-4690 CPU @ 3.50GHz, Arch Linux
| Benchmark Task                            | ThreadPool                        | ThreadPoolExecutor            | Speedup   |
|:------------------------------------------|:----------------------------------|:------------------------------|:----------|
| Submission Overhead (1M very short tasks) | 1.7251s                           | 6.9317s                       | 4.02x     |
| Submission Overhead (1M long tasks)       | 0.2620s                           | 4.3941s                       | 16.77x    |
| Launch Overhead (1M tasks)                | 1.7325s                           | 8.2352s                       | 4.75x     |
| CPU-bound (1000 fibonacci)                | 2.5791s                           | 2.6222s                       | 1.02x     |
| I/O-bound (100 tasks, 0.01s I/O)          | 0.2526s                           | 0.2531s                       | 1.00x     |
| Mixed workload (100 tasks)                | 0.1316s                           | 0.1321s                       | 1.00x     |
| Short tasks (10000 tasks)                 | 0.0176s                           | 0.0795s                       | 4.51x     |
| Priority scheduling (100 tasks)           | 0.2626s (100% priorities honored) | 0.2530s (no priority support) | N/A       |
| Time to first result (latency)            | 0.0011s                           | 0.0015s                       | 1.30x     |
| Throughput (10000 tasks)                  | 545892/s                          | 131414/s                      | 4.15x     |
| 1 workers                                 | 0.0011s                           | 0.0082s                       | 7.16x     |
| 2 workers                                 | 0.0016s                           | 0.0081s                       | 5.00x     |
| 4 workers                                 | 0.0022s                           | 0.0076s                       | 3.54x     |
| 8 workers                                 | 0.0035s                           | 0.0095s                       | 2.75x     |
| 16 workers                                | 0.0028s                           | 0.0135s                       | 4.78x     |


Explanation of the results:
- Submitting work is significantly faster with cymade's `ThreadPool` against CPython's `ThreadPoolExecutor`. In fact submitting work to `ThreadPoolExecutor` can be so heavy that for small functions it is often better to not run them in a threadpool.
- When the jobs are almost instantaneous to finish, the submission is slower (due to lock contention), but still faster than `ThreadPoolExecutor` .
- For intensive tasks, using either makes no difference... Except `ThreadPool` supports jobs with priority.

Note that for speed, the Future returned by `submit` is not a subclass of `concurrent.futures.Future` and thus won't be accepted by some APIs requiring a direct subclass of it. As it won't work with asyncio's `wrap_future`, we've added the feature that you can await directly the future in your asyncio loop (It is not needed in any way to use asyncio to use `ThreadPool`.)