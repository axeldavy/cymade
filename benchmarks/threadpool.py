from cymade.threadpool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
import tabulate
import threading
import time

def submission_benchmark():
    """Benchmark the submission overhead of ThreadPool vs ThreadPoolExecutor"""
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
    threadpool_time = time.time() - start

    # Measure the time taken by the ThreadPoolExecutor
    start = time.time()
    for i in range(n):
        executor.submit(f, i)
    executor_time = time.time() - start
    
    return {
        "Task": "Submission Overhead (1M tasks)",
        "ThreadPool": f"{threadpool_time:.4f}s",
        "ThreadPoolExecutor": f"{executor_time:.4f}s",
        "Speedup": f"{executor_time/threadpool_time:.2f}x"
    }

def launch_overhead_benchmark():
    """Benchmark the overhead to start work items of ThreadPool vs ThreadPoolExecutor"""
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
    threadpool_time = time.time() - start

    # Measure the time taken by the ThreadPoolExecutor
    start = time.time()
    for i in range(n):
        executor.submit(f, i)
    executor.shutdown()
    executor_time = time.time() - start
    
    return {
        "Task": "Launch Overhead (1M tasks)",
        "ThreadPool": f"{threadpool_time:.4f}s",
        "ThreadPoolExecutor": f"{executor_time:.4f}s",
        "Speedup": f"{executor_time/threadpool_time:.2f}x"
    }

def cpu_bound_benchmark(num_workers=4, num_tasks=10000):
    """Benchmark CPU-bound tasks (Fibonacci calculation)"""
    
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    # Create pools
    threadpool = ThreadPool(num_workers)
    executor = ThreadPoolExecutor(num_workers)
    
    # Prepare tasks - calculate fibonacci(20) multiple times
    tasks = [20] * num_tasks
    
    # ThreadPool
    start = time.time()
    futures = [threadpool.submit(fibonacci, n) for n in tasks]
    results = [future.result() for future in futures]
    threadpool_time = time.time() - start
    threadpool.shutdown()
    
    # ThreadPoolExecutor
    start = time.time()
    futures = [executor.submit(fibonacci, n) for n in tasks]
    results = [future.result() for future in futures]
    executor_time = time.time() - start
    executor.shutdown()
    
    print(f"CPU-bound tasks ({num_tasks} fibonacci calculations):")
    print(f"  ThreadPool: {threadpool_time:.4f}s")
    print(f"  ThreadPoolExecutor: {executor_time:.4f}s")
    
    return {
        "Task": f"CPU-bound ({num_tasks} fibonacci)",
        "ThreadPool": f"{threadpool_time:.4f}s",
        "ThreadPoolExecutor": f"{executor_time:.4f}s",
        "Speedup": f"{executor_time/threadpool_time:.2f}x"
    }

def io_bound_benchmark(num_workers=4, num_tasks=100, io_time=0.01):
    """Benchmark I/O-bound tasks (simulated with sleep)"""
    
    def io_task(delay):
        time.sleep(delay)
        return delay
    
    # Create pools
    threadpool = ThreadPool(num_workers)
    executor = ThreadPoolExecutor(num_workers)
    
    # Prepare tasks
    tasks = [io_time] * num_tasks
    
    # ThreadPool
    start = time.time()
    futures = [threadpool.submit(io_task, t) for t in tasks]
    results = [future.result() for future in futures]
    threadpool_time = time.time() - start
    threadpool.shutdown()
    
    # ThreadPoolExecutor
    start = time.time()
    futures = [executor.submit(io_task, t) for t in tasks]
    results = [future.result() for future in futures]
    executor_time = time.time() - start
    executor.shutdown()
    
    print(f"I/O-bound tasks ({num_tasks} tasks with {io_time}s I/O time):")
    print(f"  ThreadPool: {threadpool_time:.4f}s")
    print(f"  ThreadPoolExecutor: {executor_time:.4f}s")
    
    return {
        "Task": f"I/O-bound ({num_tasks} tasks, {io_time}s I/O)",
        "ThreadPool": f"{threadpool_time:.4f}s",
        "ThreadPoolExecutor": f"{executor_time:.4f}s",
        "Speedup": f"{executor_time/threadpool_time:.2f}x"
    }

def mixed_workload_benchmark(num_workers=4, num_tasks=100):
    """Benchmark mixed workload (combination of CPU and I/O tasks)"""
    
    def mixed_task(n, io_time):
        # CPU work
        result = 0
        for i in range(n):
            result += i * i
        # I/O work
        time.sleep(io_time)
        return result
    
    # Create pools
    threadpool = ThreadPool(num_workers)
    executor = ThreadPoolExecutor(num_workers)
    
    # Prepare tasks - mix of CPU and I/O with varying intensity
    tasks = [(i % 10000, 0.001 + (i % 5) * 0.002) for i in range(num_tasks)]
    
    # ThreadPool
    start = time.time()
    futures = [threadpool.submit(mixed_task, n, io) for n, io in tasks]
    results = [future.result() for future in futures]
    threadpool_time = time.time() - start
    threadpool.shutdown()
    
    # ThreadPoolExecutor
    start = time.time()
    futures = [executor.submit(mixed_task, n, io) for n, io in tasks]
    results = [future.result() for future in futures]
    executor_time = time.time() - start
    executor.shutdown()
    
    print(f"Mixed workload ({num_tasks} tasks):")
    print(f"  ThreadPool: {threadpool_time:.4f}s")
    print(f"  ThreadPoolExecutor: {executor_time:.4f}s")
    
    return {
        "Task": f"Mixed workload ({num_tasks} tasks)",
        "ThreadPool": f"{threadpool_time:.4f}s",
        "ThreadPoolExecutor": f"{executor_time:.4f}s",
        "Speedup": f"{executor_time/threadpool_time:.2f}x"
    }

def short_tasks_benchmark(num_workers=4, num_tasks=10000):
    """Benchmark very short tasks to measure overhead"""
    
    def short_task(x):
        return x + 1
    
    # Create pools
    threadpool = ThreadPool(num_workers)
    executor = ThreadPoolExecutor(num_workers)
    
    # ThreadPool
    start = time.time()
    futures = [threadpool.submit(short_task, i) for i in range(num_tasks)]
    results = [future.result() for future in futures]
    threadpool_time = time.time() - start
    threadpool.shutdown()
    
    # ThreadPoolExecutor
    start = time.time()
    futures = [executor.submit(short_task, i) for i in range(num_tasks)]
    results = [future.result() for future in futures]
    executor_time = time.time() - start
    executor.shutdown()
    
    print(f"Short tasks ({num_tasks} minimal operations):")
    print(f"  ThreadPool: {threadpool_time:.4f}s")
    print(f"  ThreadPoolExecutor: {executor_time:.4f}s")
    
    return {
        "Task": f"Short tasks ({num_tasks} tasks)",
        "ThreadPool": f"{threadpool_time:.4f}s",
        "ThreadPoolExecutor": f"{executor_time:.4f}s",
        "Speedup": f"{executor_time/threadpool_time:.2f}x"
    }

def priority_scheduling_benchmark(num_workers=4, num_tasks=100):
    """Benchmark priority scheduling in ThreadPool vs standard ThreadPoolExecutor"""

    completion_order = []
    completion_mutex = threading.Lock()

    def task(delay, task_id):
        nonlocal completion_order, completion_mutex
        time.sleep(delay)
        with completion_mutex:
            completion_order.append(task_id)
    
    # Create pools
    threadpool = ThreadPool(num_workers)
    executor = ThreadPoolExecutor(num_workers)
    
    # Prepare mixed priority tasks for ThreadPool
    # Higher priority (negative values) should complete first
    tasks_threadpool = []
    for i in range(num_tasks):
        # Every 10th task is high priority
        if i % 10 == 0:
            priority = -10  # High priority
        else:
            priority = 0    # Normal priority
        tasks_threadpool.append((priority, 0.01, i))
    
    # ThreadPool with priority scheduling
    start = time.time()
    futures = []
    for priority, delay, task_id in tasks_threadpool:
        futures.append(threadpool.schedule(priority, task, delay, task_id))
    
    # Track completion order
    for future in futures:
        future.result()
    
    threadpool_time = time.time() - start
    threadpool.shutdown()
    
    # Calculate how many high priority tasks finished in first 10%
    high_priority_tasks = [i for i in range(num_tasks) if i % 10 == 0]
    first_10_percent = completion_order[:num_tasks//10]
    high_priority_first = sum(1 for task_id in first_10_percent if task_id in high_priority_tasks)
    high_priority_ratio = high_priority_first / len(high_priority_tasks) if high_priority_tasks else 0

    # reset completion order
    completion_order = []

    # ThreadPoolExecutor (no priority support)
    start = time.time()
    futures = []
    for _, delay, task_id in tasks_threadpool:
        futures.append(executor.submit(task, delay, task_id))
    
    # Track completion order
    for future in futures:
        future.result()
    
    executor_time = time.time() - start
    executor.shutdown()
    
    print(f"Priority scheduling ({num_tasks} tasks):")
    print(f"  ThreadPool: {threadpool_time:.4f}s, {high_priority_ratio:.1%} high priority tasks completed early")
    print(f"  ThreadPoolExecutor: {executor_time:.4f}s (no priority support)")
    
    return {
        "Task": f"Priority scheduling ({num_tasks} tasks)",
        "ThreadPool": f"{threadpool_time:.4f}s ({high_priority_ratio:.0%} priorities honored)",
        "ThreadPoolExecutor": f"{executor_time:.4f}s (no priority support)",
        "Speedup": "N/A"
    }

def latency_benchmark(num_workers=4, num_tasks=100):
    """Measure time to first result (latency)"""
    
    def task(delay, task_id):
        time.sleep(delay)
        return task_id
    
    # Create pools
    threadpool = ThreadPool(num_workers)
    executor = ThreadPoolExecutor(num_workers)
    
    # Prepare tasks - one fast task and many slower ones
    tasks = [(0.001 if i == 0 else 0.05, i) for i in range(num_tasks)]
    
    # ThreadPool latency
    start = time.time()
    futures = []
    for delay, task_id in tasks:
        futures.append(threadpool.submit(task, delay, task_id))
    
    # Get first result
    first_result = futures[0].result()
    first_result_time = time.time() - start
    
    # Wait for all tasks to complete
    for future in futures[1:]:
        future.result()
        
    threadpool_time = time.time() - start
    threadpool.shutdown()
    
    # ThreadPoolExecutor latency
    start = time.time()
    futures = []
    for delay, task_id in tasks:
        futures.append(executor.submit(task, delay, task_id))
    
    # Get first result
    first_result = futures[0].result()
    first_result_time_executor = time.time() - start
    
    # Wait for all tasks to complete
    for future in futures[1:]:
        future.result()
        
    executor_time = time.time() - start
    executor.shutdown()
    
    print(f"Latency benchmark (time to first result):")
    print(f"  ThreadPool: {first_result_time:.4f}s")
    print(f"  ThreadPoolExecutor: {first_result_time_executor:.4f}s")
    
    return {
        "Task": "Time to first result (latency)",
        "ThreadPool": f"{first_result_time:.4f}s",
        "ThreadPoolExecutor": f"{first_result_time_executor:.4f}s",
        "Speedup": f"{first_result_time_executor/first_result_time:.2f}x"
    }

def throughput_benchmark(num_workers=4, task_count=10000):
    """Measure throughput (tasks per second)"""
    
    def noop(x):
        return x
    
    # Create pools
    threadpool = ThreadPool(num_workers)
    executor = ThreadPoolExecutor(num_workers)
    
    # ThreadPool throughput
    start = time.time()
    futures = []
    for i in range(task_count):
        futures.append(threadpool.submit(noop, i))
    
    results = [f.result() for f in futures]
    threadpool_time = time.time() - start
    threadpool_throughput = task_count / threadpool_time
    threadpool.shutdown()
    
    # ThreadPoolExecutor throughput
    start = time.time()
    futures = []
    for i in range(task_count):
        futures.append(executor.submit(noop, i))
    
    results = [f.result() for f in futures]
    executor_time = time.time() - start
    executor_throughput = task_count / executor_time
    executor.shutdown()
    
    print(f"Throughput benchmark ({task_count} tasks):")
    print(f"  ThreadPool: {threadpool_throughput:.0f} tasks/second")
    print(f"  ThreadPoolExecutor: {executor_throughput:.0f} tasks/second")
    
    return {
        "Task": f"Throughput ({task_count} tasks)",
        "ThreadPool": f"{threadpool_throughput:.0f}/s",
        "ThreadPoolExecutor": f"{executor_throughput:.0f}/s",
        "Speedup": f"{threadpool_throughput/executor_throughput:.2f}x"
    }

def worker_scaling_benchmark():
    """Benchmark performance with different numbers of workers"""
    results = []
    
    # Test with 1, 2, 4, 8, 16 workers
    worker_counts = [1, 2, 4, 8, 16]
    
    for num_workers in worker_counts:
        # Use small task count for faster testing
        threadpool = ThreadPool(num_workers)
        executor = ThreadPoolExecutor(num_workers)
        
        task_count = 1000
        
        # ThreadPool
        start = time.time()
        futures = []
        for i in range(task_count):
            futures.append(threadpool.submit(math.sqrt, i))
        
        for future in futures:
            future.result()
        
        threadpool_time = time.time() - start
        threadpool.shutdown()
        
        # ThreadPoolExecutor
        start = time.time()
        futures = []
        for i in range(task_count):
            futures.append(executor.submit(math.sqrt, i))
        
        for future in futures:
            future.result()
        
        executor_time = time.time() - start
        executor.shutdown()
        
        print(f"Workers: {num_workers}, Tasks: {task_count}")
        print(f"  ThreadPool: {threadpool_time:.4f}s")
        print(f"  ThreadPoolExecutor: {executor_time:.4f}s")
        
        results.append({
            "Task": f"{num_workers} workers",
            "ThreadPool": f"{threadpool_time:.4f}s",
            "ThreadPoolExecutor": f"{executor_time:.4f}s",
            "Speedup": f"{executor_time/threadpool_time:.2f}x"
        })
    
    return results

def run_comprehensive_benchmark():
    """Run all benchmarks and generate a table"""
    print("Running comprehensive thread pool benchmarks...")
    print("==============================================")
    
    results = []
    
    # Run all benchmarks
    results.append(submission_benchmark())
    results.append(launch_overhead_benchmark())
    results.append(cpu_bound_benchmark(num_workers=4, num_tasks=1000))
    results.append(io_bound_benchmark(num_workers=4, num_tasks=100, io_time=0.01))
    results.append(mixed_workload_benchmark(num_workers=4, num_tasks=100))
    results.append(short_tasks_benchmark(num_workers=4, num_tasks=10000))
    results.append(priority_scheduling_benchmark(num_workers=4, num_tasks=100))
    results.append(latency_benchmark(num_workers=4, num_tasks=100))
    results.append(throughput_benchmark(num_workers=4, task_count=10000))
    results.extend(worker_scaling_benchmark())
    
    # Generate table
    headers = ["Benchmark Task", "ThreadPool", "ThreadPoolExecutor", "Speedup"]
    table_data = [[r["Task"], r["ThreadPool"], r["ThreadPoolExecutor"], r["Speedup"]] for r in results]
    
    # Print table
    print("\nBenchmark Results:")
    print("=================")
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print markdown table for README
    print("\nMarkdown Table for README:")
    print("========================")
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="pipe"))
    
    return results

if __name__ == "__main__":
    run_comprehensive_benchmark()