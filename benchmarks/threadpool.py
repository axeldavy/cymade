
from cymade.threadpool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time

def compare_threadpool_overhead():
    # Create a ThreadPool
    threadpool = ThreadPool(4)
    # Create a ThreadPoolExecutor
    executor = ThreadPoolExecutor(4)

    # Define a function that will be executed
    def f(x):
        return x**2
    
    # Define the number of iterations
    n = 1000000

    # Measure the time taken by the ThreadPool
    start = time.time()
    for i in range(n):
        threadpool.submit(f, i)
    threadpool.shutdown()
    print("ThreadPool time:", time.time() - start)

    # Measure the time taken by the ThreadPoolExecutor
    start = time.time()
    for i in range(n):
        executor.submit(f, i)
    executor.shutdown()
    print("ThreadPoolExecutor time:", time.time() - start)

if __name__ == "__main__":
    compare_threadpool_overhead()