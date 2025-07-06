from cymade.atomic import U64, Float, Int, I64
import threading
import time
import gc
import tabulate
from concurrent.futures import ThreadPoolExecutor

gc.disable()  # Disable garbage collection for benchmarking

# ---- Implementation of equivalent types using threading.Lock ----

class LockCounter:
    """Integer counter protected with threading.Lock"""
    def __init__(self, init_value=0, on_zero=None):
        self._value = init_value
        self._lock = threading.Lock()
        self._on_zero = on_zero
        
    def __int__(self):
        with self._lock:
            return int(self._value)
            
    def __float__(self):
        with self._lock:
            return float(self._value)
            
    def __iadd__(self, other):
        with self._lock:
            self._value += other
            if self._value == 0 and self._on_zero:
                self._on_zero()
            return self
    
    def __isub__(self, other):
        with self._lock:
            self._value -= other
            if self._value == 0 and self._on_zero:
                self._on_zero()
            return self
            
    def __imul__(self, other):
        with self._lock:
            self._value *= other
            if self._value == 0 and self._on_zero:
                self._on_zero()
            return self
            
    def __ifloordiv__(self, other):
        with self._lock:
            self._value //= other
            if self._value == 0 and self._on_zero:
                self._on_zero()
            return self
            
    def __imod__(self, other):
        with self._lock:
            self._value %= other
            if self._value == 0 and self._on_zero:
                self._on_zero()
            return self


class LockFloat:
    """Float value protected with threading.Lock"""
    def __init__(self, init_value=0.0):
        self._value = float(init_value)
        self._lock = threading.Lock()
        
    def __float__(self):
        with self._lock:
            return float(self._value)
            
    def __int__(self):
        with self._lock:
            return int(self._value)
            
    def __iadd__(self, other):
        with self._lock:
            self._value += other
            return self
    
    def __isub__(self, other):
        with self._lock:
            self._value -= other
            return self
            
    def __imul__(self, other):
        with self._lock:
            self._value *= other
            return self
            
    def __itruediv__(self, other):
        with self._lock:
            self._value /= other
            return self


# ---- Benchmark Functions ----

def single_thread_increment(counter_type, iterations=1000000):
    """Benchmark single-threaded incrementation"""
    counter = counter_type(0)
    
    start = time.perf_counter()
    for _ in range(iterations):
        counter += 1
    end = time.perf_counter()
    
    return end - start


def single_thread_increment_normal(iterations=1000000):
    """Benchmark single-threaded incrementation with a regular variable"""
    counter = 0
    
    start = time.perf_counter()
    for _ in range(iterations):
        counter += 1
    end = time.perf_counter()
    
    return end - start


def multi_thread_increment(counter_type, num_threads=4, iterations_per_thread=250000):
    """Benchmark multi-threaded incrementation"""
    counter = counter_type(0)
    
    def increment_task():
        nonlocal counter
        for _ in range(iterations_per_thread):
            counter += 1
    
    threads = []
    start = time.perf_counter()
    
    for _ in range(num_threads):
        t = threading.Thread(target=increment_task)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    end = time.perf_counter()
    
    # Verify result to ensure atomicity worked
    expected = num_threads * iterations_per_thread
    actual = int(counter)
    assert actual == expected

    return end - start


def event_toggle_benchmark(iterations=1000000):
    """Benchmark setting and clearing threading.Event"""
    event = threading.Event()
    
    start = time.perf_counter()
    for i in range(iterations):
        if i % 2 == 0:
            event.set()
        else:
            event.clear()
    end = time.perf_counter()
    
    return end - start


def float_operations(float_type, iterations=1000000):
    """Benchmark floating point operations"""
    value = float_type(1.0)
    
    start = time.perf_counter()
    for i in range(iterations):
        value *= 1.01
        value /= 1.01
    end = time.perf_counter()
    
    return end - start


def float_operations_normal(iterations=1000000):
    """Benchmark floating point operations with a regular variable"""
    value = 1.0
    
    start = time.perf_counter()
    for i in range(iterations):
        value *= 1.01
        value /= 1.01
    end = time.perf_counter()
    
    return end - start


def multi_thread_float(float_type, num_threads=4, iterations_per_thread=250000):
    """Benchmark multi-threaded float operations"""
    value = float_type(1.0)
    
    def float_task():
        nonlocal value
        for _ in range(iterations_per_thread):
            value *= 1.01
            value /= 1.01
    
    threads = []
    start = time.perf_counter()
    
    for _ in range(num_threads):
        t = threading.Thread(target=float_task)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    end = time.perf_counter()
    
    return end - start


def contended_updates(counter_type, num_threads=20, iterations_per_thread=100000):
    """Benchmark highly contended updates (many threads)"""
    counter = counter_type(0)
    
    def update_task():
        nonlocal counter
        for _ in range(iterations_per_thread):
            counter += 1
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        start = time.perf_counter()
        futures = [executor.submit(update_task) for _ in range(num_threads)]
        for f in futures:
            f.result()
        end = time.perf_counter()
    
    expected = num_threads * iterations_per_thread
    actual = int(counter)
    assert actual == expected
    
    return end - start


def on_zero_callback_test(counter_type, iterations=100000):
    """Test the performance impact of the on_zero callback (U64 specific)"""
    callback_count = 0
    
    def on_zero():
        nonlocal callback_count
        callback_count += 1
    
    counter = counter_type(0, on_zero=on_zero)
    
    start = time.perf_counter()
    for _ in range(iterations):
        counter += 1
        counter -= 1  # Should trigger on_zero each time
    end = time.perf_counter()
    
    # Verify callback was called expected number of times
    assert callback_count == iterations
    
    return end - start


def run_benchmarks():
    """Run all atomic benchmarks and generate a table"""
    print("Running atomic benchmarks...")
    print("===========================")
    
    results = []
    
    # Single-threaded increment
    cymade_time = single_thread_increment(U64)
    lock_time = single_thread_increment(LockCounter)
    normal_time = single_thread_increment_normal()
    
    results.append({
        "Test": "Single-thread increment (1M)",
        "cymade.atomic": f"{cymade_time:.6f}s",
        "threading.Lock": f"{lock_time:.6f}s",
        "Normal Variable": f"{normal_time:.6f}s",
        "Speedup vs Lock": f"{lock_time/cymade_time:.2f}x"
    })
    
    # Multi-threaded increment
    cymade_time = multi_thread_increment(U64)
    lock_time = multi_thread_increment(LockCounter)
    
    results.append({
        "Test": "Multi-thread increment (4 threads, 1M total)",
        "cymade.atomic": f"{cymade_time:.6f}s",
        "threading.Lock": f"{lock_time:.6f}s",
        "Normal Variable": "N/A (not thread-safe)",
        "Speedup vs Lock": f"{lock_time/cymade_time:.2f}x"
    })
    
    # Float operations
    cymade_float_time = float_operations(Float)
    lock_float_time = float_operations(LockFloat)
    normal_float_time = float_operations_normal()
    
    results.append({
        "Test": "Float operations (1M)",
        "cymade.atomic": f"{cymade_float_time:.6f}s",
        "threading.Lock": f"{lock_float_time:.6f}s",
        "Normal Variable": f"{normal_float_time:.6f}s",
        "Speedup vs Lock": f"{lock_float_time/cymade_float_time:.2f}x"
    })
    
    # Multi-threaded float operations
    cymade_float_mt_time = multi_thread_float(Float)
    lock_float_mt_time = multi_thread_float(LockFloat)
    
    results.append({
        "Test": "Multi-thread float ops (4 threads, 1M total)",
        "cymade.atomic": f"{cymade_float_mt_time:.6f}s",
        "threading.Lock": f"{lock_float_mt_time:.6f}s",
        "Normal Variable": "N/A (not thread-safe)",
        "Speedup vs Lock": f"{lock_float_mt_time/cymade_float_mt_time:.2f}x"
    })
    
    # High contention updates
    cymade_contended_time = contended_updates(U64)
    lock_contended_time = contended_updates(LockCounter)
    
    results.append({
        "Test": "High contention (20 threads, 2M total)",
        "cymade.atomic": f"{cymade_contended_time:.6f}s",
        "threading.Lock": f"{lock_contended_time:.6f}s",
        "Normal Variable": "N/A (not thread-safe)",
        "Speedup vs Lock": f"{lock_contended_time/cymade_contended_time:.2f}x"
    })
    
    # On-zero callback
    cymade_callback_time = on_zero_callback_test(U64)
    lock_callback_time = on_zero_callback_test(LockCounter)
    
    results.append({
        "Test": "On-zero callback (100K)",
        "cymade.atomic": f"{cymade_callback_time:.6f}s",
        "threading.Lock": f"{lock_callback_time:.6f}s",
        "Normal Variable": "N/A (no callback support)",
        "Speedup vs Lock": f"{lock_callback_time/cymade_callback_time:.2f}x"
    })
    
    # Event toggling
    event_time = event_toggle_benchmark()
    
    results.append({
        "Test": "Event toggle (1M)",
        "cymade.atomic": "N/A",
        "threading.Event": f"{event_time:.6f}s",
        "Normal Variable": "N/A",
        "Speedup vs Lock": "N/A"
    })
    
    # Generate table
    headers = ["Test Case", "cymade.atomic", "threading.Lock", "Normal", "Speedup"]
    table_data = [
        [r["Test"],
         r["cymade.atomic"],
         r["threading.Lock"] if "threading.Lock" in r else r.get("threading.Event", "N/A"), 
         r["Normal Variable"], 
         r["Speedup vs Lock"]] 
        for r in results
    ]
    
    # Print table
    print("\nBenchmark Results:")
    print("=================")
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print markdown table
    print("\nMarkdown Table for README:")
    print("========================")
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="pipe"))
    
    return results


if __name__ == "__main__":
    run_benchmarks()