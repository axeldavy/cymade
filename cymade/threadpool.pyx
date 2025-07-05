#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: embedsignature=False
#cython: cdivision=True
#cython: cdivision_warnings=False
#cython: always_allow_keywords=False
#cython: infer_types=False
#cython: initializedcheck=False
#cython: c_line_in_traceback=False
#cython: auto_pickle=False
#cython: freethreading_compatible=True
#distutils: language=c++


from libcpp.queue cimport priority_queue
from libc.stdint cimport int32_t, int64_t

cimport cython

from asyncio import get_event_loop
from concurrent.futures import TimeoutError, CancelledError
from os import cpu_count
from threading import Thread
from traceback import format_exc
from weakref import WeakKeyDictionary

from .cpp_includes cimport mutex, unique_lock, condition_variable, defer_lock_t
from .cpp_includes cimport atomic
from .atomic cimport lock_gil_friendly

cdef extern from * nogil:
    """
    #include <chrono>
    #include <thread>
    void micro_sleep(int microseconds) {
        std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
    }
    
    // Get current time in microseconds since epoch
    long long get_current_time_us() {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }
    
    // Convert seconds to microseconds
    long long seconds_to_us(double seconds) {
        return static_cast<long long>(seconds * 1000000.0);
    }
    std::chrono::microseconds to_chrono_us(long long microseconds) {
        return std::chrono::microseconds(microseconds);
    }
    bool get_wait_success(std::cv_status status) {
        return status == std::cv_status::no_timeout;
    }
    """
    void micro_sleep(int microseconds)
    long long get_current_time_us()
    long long seconds_to_us(double seconds)
    long long to_chrono_us(long long microseconds)
    bint get_wait_success(bint)

cdef class Worker:
    """
    Represents a worker thread running task from the threadpool
    """
    cdef ThreadPool _pool
    cdef object __weakref__

    def __init__(self):
        raise NotImplementedError("Workers are created by the ThreadPool")

    def __cinit__(self):
        self._pool = None

    def _run(self):
        """
        Main loop of the worker
        """
        cdef Future future
        while True:
            future = self._pool.wait_for_work()
            if future is None:
                break
            try:
                future.run()
            except Exception as e:
                self._pool.report_error(
                    future, e, format_exc()
                )

    @staticmethod
    cdef Worker create(ThreadPool pool):
        """
        Create a worker in a new thread and start it
        """
        cdef Worker worker = Worker.__new__(Worker)
        worker._pool = pool
        thread = Thread(target=worker._run)
        thread.daemon = True
        thread.start()
        return worker

@cython.final
cdef class Future:
    """
    An optimized future object representing the result of asynchronous computation.
    
    The Future class provides a way to access the result of a function that has been
    executed asynchronously in a thread pool. This implementation is similar to
    Python's standard `concurrent.futures.Future`
    """
    cdef object _result
    cdef object _exception
    cdef object _callable_info
    cdef bint _cancelled
    cdef bint _running
    cdef bint _done
    cdef mutex _mutex
    cdef condition_variable _condition
    cdef list _callbacks
    cdef ThreadPool _pool
    cdef int64_t _uuid
    cdef double _priority
    
    def __cinit__(self):
        self._result = None
        self._exception = None
        self._callable_info = None
        self._cancelled = False
        self._running = False
        self._done = False
        #self._mutex = mutex()
        #self._condition = condition_variable()
        self._callbacks = []
        self._pool = None
        self._uuid = 0
        self._priority = 0.
    
    @staticmethod
    cdef Future from_callable_info(ThreadPool pool, object callable_info, int64_t uuid, double priority):
        """Create a new Future from a callable"""
        cdef Future self = Future.__new__(Future)
        self._callable_info = callable_info
        self._pool = pool
        self._uuid = uuid
        self._priority = priority
        return self
    
    @property
    def uuid(self):
        """Return the UUID of this future"""
        return self._uuid
    
    def cancel(self):
        """
        Attempt to cancel the call. Return False if the call is currently
        being executed or finished running, True otherwise.
        """
        cdef unique_lock[mutex] lock
        cdef ThreadPool pool
        cdef list callbacks

        lock_gil_friendly(lock, self._mutex)
        
        if self._running or self._done:
            return False
            
        self._cancelled = True
        self._done = True
        callbacks = self._callbacks
        self._callbacks = []
        pool = self._pool
        self._pool = None
        lock.unlock()

        # Notify pool about completion
        if pool is not None:
            pool.on_future_completed(self, self._priority)
        # Notify waiting threads
        self._condition.notify_all()
        # Call the callbacks
        self._call_callbacks(callbacks)

        return True
    
    def cancelled(self):
        """Return True if the call was successfully cancelled."""
        cdef unique_lock[mutex] lock
        lock_gil_friendly(lock, self._mutex)
        return self._cancelled
    
    def running(self):
        """Return True if the call is currently being executed."""
        cdef unique_lock[mutex] lock
        lock_gil_friendly(lock, self._mutex)
        return self._running
    
    def done(self):
        """Return True if the call was successfully cancelled or finished running."""
        cdef unique_lock[mutex] lock
        lock_gil_friendly(lock, self._mutex)
        return self._done

    def __await__(self):
        """
        Makes the Future awaitable in asyncio contexts.
        
        This allows using the Future directly with the 'await' keyword
        in asynchronous functions without wrapping it with asyncio.wrap_future(),
        which isn't compatible with this implementation.
        
        Returns:
        --------
        iterator
            An iterator compatible with the asyncio await protocol.
        """
        # Create an asyncio future to bridge the gap
        loop = get_event_loop()
        asyncio_future = loop.create_future()
        
        # Helper function to transfer the result when done
        def _on_done(Future future, asyncio_future=asyncio_future):
            cdef unique_lock[mutex] lock
            lock_gil_friendly(lock, future._mutex)
            if future._cancelled:
                asyncio_future.cancel()
            elif future._exception is not None:
                asyncio_future.set_exception(future._exception)
            else:
                asyncio_future.set_result(future._result)
        
        self.add_done_callback(_on_done)

        # Return the iterator from the asyncio Future
        return asyncio_future.__await__()

    cdef int _wait_done(self, timeout):
        """Wait until the future is done"""
        cdef unique_lock[mutex] lock
        lock_gil_friendly(lock, self._mutex)

        cdef bint wait_success
        cdef long long end_time_us
        cdef long long current_time_us
        cdef long long remaining_us
        
        if not self._done:
            if timeout is None:
                # Wait indefinitely
                with nogil:
                    self._condition.wait(lock)
            else:
                # Wait with timeout
                end_time_us = get_current_time_us() + seconds_to_us(timeout)
                
                while not self._done:
                    current_time_us = get_current_time_us()
                    if current_time_us >= end_time_us:
                        raise TimeoutError()
                    
                    remaining_us = end_time_us - current_time_us
                    with nogil:
                        wait_success = get_wait_success(self._condition.wait_for(lock, to_chrono_us(remaining_us)))
                    
                    if not wait_success and not self._done:
                        raise TimeoutError()
    
    def result(self, timeout=None):
        """
        Return the result of the call.

        This method blocks until the Future completes or the specified timeout expires.
        Once completed, it returns the result of the callable that was submitted to
        the thread pool, or raises any exception that was raised during execution.

        Parameters:
        -----------
        timeout : float or None, default=None
            Maximum number of seconds to wait for the result. If None, wait indefinitely.
            Can be an integer or a float for sub-second precision.

        Returns:
        --------
        object
            The result of the callable that was submitted to the thread pool.

        Raises:
        -------
        TimeoutError
            If the timeout expires before the Future completes.
        CancelledError
            If the Future was cancelled before completing.
        Exception
            Any exception raised during execution of the callable.
        """
        if self._cancelled:
            raise CancelledError()
            
        self._wait_done(timeout)
        
        if self._exception is not None:
            raise self._exception

        return self._result
    
    def exception(self, timeout=None):
        """
        Return the exception raised by the call.

        This method blocks until the Future completes or the specified timeout expires.
        It returns the exception raised by the callable that was submitted to the
        thread pool, or None if the callable completed without raising an exception.

        Parameters:
        -----------
        timeout : float or None, default=None
            Maximum number of seconds to wait for the Future to complete. If None,
            wait indefinitely. Can be an integer or a float for sub-second precision.

        Returns:
        --------
        Exception or None
            The exception raised by the callable, or None if no exception was raised.

        Raises:
        -------
        TimeoutError
            If the timeout expires before the Future completes.
        CancelledError
            If the Future was cancelled before completing.
        """
        if self._cancelled:
            raise CancelledError()
            
        self._wait_done(timeout)
        
        return self._exception
    
    def add_done_callback(self, fn):
        """
        Attaches a callable to the Future to be called when the Future completes.

        The callback will be called with the Future object as its only argument when
        the Future completes, either by returning a result, raising an exception, or
        being cancelled.

        Parameters:
        -----------
        fn : callable
            A callable that takes a Future object as its only argument. The callable
            will be called when the Future completes.

        Notes:
        ------
        - Added callbacks are executed in the order they were added.
        - If the callable raises an Exception, it will be ignored.
        - If the Future has already completed or been cancelled, the callable is
          called immediately.
        - Callbacks are executed by the thread that completes the Future, not by
          a separate thread.
        """
        cdef unique_lock[mutex] lock
        lock_gil_friendly(lock, self._mutex)
        
        if self._done:
            # Execute the callback immediately
            try:
                fn(self)
            except Exception:
                pass
        else:
            # Store the callback to be executed later
            self._callbacks.append(fn)

    cdef void _call_callbacks(self, list callbacks):
        """Call all callbacks in the order they were registered"""
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                pass
    
    cdef void run(self):
        """Execute the callable and set the result/exception"""
        cdef unique_lock[mutex] lock
        cdef list callbacks
        cdef ThreadPool pool
        
        # Mark that we're running
        lock_gil_friendly(lock, self._mutex)
        if self._cancelled:
            lock.unlock()
            return
        
        self._running = True
        lock.unlock()
        cdef tuple callable_info = self._callable_info
        
        # Run the callable and capture result/exception
        try:
            result = callable_info[0](*callable_info[1], **callable_info[2])
            lock_gil_friendly(lock, self._mutex)
            self._result = result
        except Exception as exc:
            lock_gil_friendly(lock, self._mutex)
            self._exception = exc
        finally:
            self._done = True
            self._running = False
            # Clear callbacks
            callbacks = self._callbacks
            self._callbacks = []
            # Clear pool
            pool = self._pool
            self._pool = None
            lock.unlock()

            # Notify pool about completion
            if pool is not None:
                pool.on_future_completed(self, self._priority)
            # Notify waiting threads
            self._condition.notify_all()
            # Call the callbacks
            self._call_callbacks(callbacks)


cdef extern from * nogil:
    """
    struct Command {
        double priority;
        int64_t uuid;

        Command() : priority(0), uuid(0) {}

        Command(double p, int64_t u) : priority(p), uuid(u) {}

        bool operator<(const Command& other) const {
            if (priority > other.priority)
                return true;
            if (priority < other.priority)
                return false;
            return uuid > other.uuid;
        }

        bool operator>(const Command& other) const {
            return other < *this;
        }

        bool operator==(const Command& other) const {
            return priority == other.priority && uuid == other.uuid;
        }

        bool operator!=(const Command& other) const {
            return !(*this == other);
        }
    };
    // Helper functions for queue operations
    void queue_pop_top(Command &dst, std::priority_queue<Command>& queue) {
        // Move the top element out safely
        dst = std::move(const_cast<Command&>(queue.top()));
        queue.pop();
    }
    """
    cppclass Command:
        double priority
        int64_t uuid

        Command()
        Command(double p, int64_t u)

    void queue_pop_top(Command& dst, priority_queue[Command]& queue) noexcept

@cython.final
cdef class ThreadPool:
    """
    An efficient thread pool implementation that extends the standard threadpool API
    with priority scheduling and optimized internals.
    
    The ThreadPool class manages a group of worker threads that execute submitted tasks
    asynchronously. It provides a flexible and efficient way to parallelize CPU-bound
    and I/O-bound operations across multiple threads.
    
    Key Features:
    -------------
    - Priority-based task scheduling using C++'s priority queue
    - Optimized Cython/C++ implementation for reduced Python overhead
    - Worker threads persist through task errors for robustness
    - Support for task cancellation and timeouts
    - Custom priority levels with special semantics (blocking vs non-blocking)
    - Compatible with Python's context manager protocol (with statement)
    
    Priority System:
    ---------------
    The ThreadPool uses a priority system where:
    - Lower values indicate higher execution priority
    - Negative priority: Blocking tasks that prevent execution of non-negative priority tasks
    - Zero priority: Standard tasks with no special handling
    - Positive priority: Tasks that respect a waiting period (in ms) after blocking tasks
    
    Performance Characteristics:
    --------------------------
    - Minimal Python GIL contention due to C++ internals
    - Efficient task scheduling with O(log n) insertion and removal
    - Low memory overhead compared to pure Python implementations
    - Automatic thread management based on system capabilities
    
    Error Handling:
    --------------
    By default, exceptions in tasks are logged without terminating the worker threads.
    This behavior can be customized by subclassing ThreadPool and overriding the 
    report_error method.
    
    Examples:
    ---------
    Basic usage:
    
    >>> import time
    >>> from cymade.threadpool import ThreadPool
    >>>
    >>> # Create a thread pool with 4 workers
    >>> pool = ThreadPool(max_workers=4)
    >>>
    >>> # Submit tasks to the pool
    >>> def compute_square(x):
    ...     return x * x
    ...
    >>> future = pool.submit(compute_square, 10)
    >>> print(future.result())  # Wait for and retrieve the result
    100
    >>>
    >>> # Don't forget to shut down the pool when done
    >>> pool.shutdown()
    
    Using as a context manager:
    
    >>> with ThreadPool() as pool:
    ...     # Submit multiple tasks
    ...     futures = [pool.submit(compute_square, i) for i in range(10)]
    ...     
    ...     # Collect results
    ...     results = [future.result() for future in futures]
    ...     print(results)
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    
    Priority scheduling:
    
    >>> def slow_task(task_id, duration):
    ...     time.sleep(duration)
    ...     return f"Task {task_id} completed"
    >>>
    >>> with ThreadPool(max_workers=2) as pool:
    ...     # Submit tasks with different priorities
    ...     # High priority (blocking) task
    ...     high_prio = pool.schedule(-10, slow_task, "high", 0.5)
    ...     
    ...     # Normal priority tasks
    ...     normal = pool.schedule(0, slow_task, "normal", 0.5)
    ...     
    ...     # Low priority task (waits 200ms after blocking tasks)
    ...     low_prio = pool.schedule(200, slow_task, "low", 0.5)
    ...     
    ...     # Results will be available in priority order
    ...     print(high_prio.result())
    ...     print(normal.result())
    ...     print(low_prio.result())
    Task high completed
    Task normal completed
    Task low completed
    
    Parallel mapping:
    
    >>> # Process items in parallel using map
    >>> with ThreadPool() as pool:
    ...     # Apply a function to each item
    ...     results = list(pool.map(lambda x: x**2, range(1, 6)))
    ...     print(results)
    [1, 4, 9, 16, 25]
    
    Error handling:
    
    >>> # Create a custom threadpool with error handling
    >>> class MyThreadPool(ThreadPool):
    ...     def report_error(self, future, exception, traceback_str):
    ...         print(f"Task failed with error: {exception}")
    ...         # Log, notify, or take other actions
    >>>
    >>> with MyThreadPool() as pool:
    ...     # This task will raise an exception
    ...     future = pool.submit(lambda: 1/0)
    ...     try:
    ...         result = future.result()
    ...     except ZeroDivisionError:
    ...         print("Caught division by zero")
    Task failed with error: division by zero
    Caught division by zero
    
    Notes:
    ------
    - The pool automatically creates threads based on the number of CPU cores when max_workers is None
    - To prevent memory leaks, always call shutdown() or use the pool as a context manager
    - For long-running applications, consider using multiple pools for different task categories
    - Avoid running CPU-intensive tasks with excessive parallelism as this can degrade performance
    """
    cdef priority_queue[Command] _queue # priority, -uuid
    cdef atomic[int64_t] _next_uuid
    cdef dict _futures # uuid to Future
    cdef object _workers # weak references to workers (WeakKeyDictionary)
    cdef mutex _mutex # mutex for public API
    cdef mutex _queue_mutex # mutex for queue operations
    cdef condition_variable _work_condition
    cdef atomic[bint] _shutdown
    cdef atomic[int32_t] _blocking_jobs  # Count of negative priority jobs
    cdef atomic[int64_t] _last_blocking_job_finished_us  # Timestamp when last low priority job finished
    cdef int _max_workers

    def __cinit__(self):
        self._queue = priority_queue[Command]()
        self._next_uuid.store(0)
        self._futures = dict()
        self._workers = WeakKeyDictionary()
        self._shutdown.store(False)
        self._blocking_jobs.store(0)  # Initialize blocking jobs counter
        self._last_blocking_job_finished_us.store(get_current_time_us())  # Initialize to current time
        self._max_workers = 0

    def __init__(self, max_workers=None):
        """
        Initialize the thread pool with the given number of workers
        
        If max_workers is None, it defaults to the number of processors on the machine
        """
        if max_workers is None:
            max_workers = cpu_count() or 1
        self._max_workers = max_workers
        
        # Start the workers
        for _ in range(max_workers):
            worker = Worker.create(self)
            self._workers[worker] = None

    def __del__(self):
        """
        Destructor to ensure resources are cleaned up
        """
        self.shutdown(wait=False)  # Ensure shutdown is called without waiting

            
    ### Functions for workers ###
    cdef Future wait_for_work(self):
        """
        Called exclusively by workers to get work.

        Returns None if the worker has to shutdown.

        Skips over cancelled futures.
        If negative priority jobs are active, only process negative priority jobs.
        For positive priority jobs, enforce waiting period based on priority in milliseconds.
        """
        cdef Future future
        cdef int64_t uuid

        if self._shutdown.load():
            return None

        # fast path
        future = self._fetch_work()
        if future is not None:
            return future

        with nogil:
            uuid = self._wait_for_work()
        if uuid < 0:
            return None # shutdown

        # Fetch the future from the queue
        future = self._futures.pop(uuid, None)

        # Return the future
        return future

    def report_error(self, Future future, Exception e, str traceback) -> None:
        """
        Called by workers to report an error in a future
        """
        print(f"Error in future {future.uuid}: {e}")
        print(traceback)

    ### Internal functions ###

    cdef Future _fetch_work(self):
        """
        Internal function of wait_for_work which fetches
        the next work item from the queue, None if none
        meet the criteria, or if the queue is busy.
        """
        cdef Command element
        cdef unique_lock[mutex] queue_mutex = unique_lock[mutex](self._queue_mutex, defer_lock_t())
        cdef long long current_time_us
        cdef long long last_blocking_time
        cdef long long wait_time_us

        if not queue_mutex.try_lock():
            return None

        if self._queue.empty():
            return None

        if self._queue.top().priority < 0:
            queue_pop_top(element, self._queue)
            queue_mutex.unlock()
            return self._futures.pop(element.uuid, None)

        # If any blocking jobs is active, top must be negative
        if self._blocking_jobs.load() > 0:
            return None

        # Check time-based priority for positive priority jobs
        if self._queue.top().priority > 0:
            current_time_us = get_current_time_us()
            last_blocking_time = self._last_blocking_job_finished_us.load()
            # Convert priority in ms to microseconds
            wait_time_us = seconds_to_us(self._queue.top().priority / 1000.0)

            # If not enough time has passed, wait and try again
            if current_time_us < last_blocking_time + wait_time_us:
                return None

        queue_pop_top(element, self._queue)
        queue_mutex.unlock()
        return self._futures.pop(element.uuid, None)

    cdef int64_t _wait_for_work(self) noexcept nogil:
        """
        Internal function of wait_for_work which waits
        that a compatible item is available in the queue.
        """
        cdef unique_lock[mutex] queue_mutex = unique_lock[mutex](self._queue_mutex)
        cdef long long current_time_us
        cdef long long last_blocking_time
        cdef long long wait_time_us
        cdef Command element

        while True:
            if self._shutdown.load():
                return -1

            # Wait the queue is filled, but check every 10 ms if we should shutdown
            if self._queue.empty():
                self._work_condition.wait_for(queue_mutex, to_chrono_us(10000))
                continue

            if self._queue.top().priority < 0:
                queue_pop_top(element, self._queue)
                return element.uuid

            # If any blocking jobs is active, top must be negative
            if self._blocking_jobs.load() > 0:
                self._work_condition.wait_for(queue_mutex, to_chrono_us(10000))
                continue

            # Check time-based priority for positive priority jobs
            if self._queue.top().priority > 0:
                current_time_us = get_current_time_us()
                last_blocking_time = self._last_blocking_job_finished_us.load()
                # Convert priority in ms to microseconds
                wait_time_us = seconds_to_us(self._queue.top().priority / 1000.0)
                # clamp wait to 10ms to check for shutdowns
                wait_time_us = min(wait_time_us, 10000)

                # If not enough time has passed, wait and try again
                if current_time_us < last_blocking_time + wait_time_us:
                    self._work_condition.wait_for(
                        queue_mutex,
                        to_chrono_us(last_blocking_time + wait_time_us - current_time_us + 1)
                    )
                    continue

            queue_pop_top(element, self._queue)
            return element.uuid

    cdef void on_future_completed(self, Future future, double priority):
        """Called when a future completes"""
        # Log last medium or high priority job finish time
        if priority <= 0:
            self._last_blocking_job_finished_us.store(get_current_time_us())

        # Unblock the queue if that was the last blocking job
        if priority < 0:
            if self._blocking_jobs.fetch_sub(1) == 1:
                self._work_condition.notify_all()

    cdef Future _submit(self, double priority, object callable_info):
        """
        Submit a callable to the threadpool
        """
        cdef unique_lock[mutex] m
        cdef int64_t uuid = self._next_uuid.fetch_add(1)
        # Create the future
        cdef Future future = Future.from_callable_info(
            self, callable_info, uuid, 
            priority
        )
        
        # If this is a blocking job, increment counter
        if priority < 0:
            self._blocking_jobs.fetch_add(1)

        # Make the work visible to workers
        self._futures[uuid] = future
        lock_gil_friendly(m, self._queue_mutex)
        self._queue.push(Command(priority, uuid))
        m.unlock()

        # Notify workers that work is available
        self._work_condition.notify_one()

        return future
        
    ### Public API - Similar to standard threadpool API ###
    def submit(self, callable, *args, **kwargs):
        """
        Submit a callable to be executed with the given arguments.
        
        This method schedules the callable to be executed and returns a Future object
        representing the execution of the callable.
        
        Parameters:
        -----------
        callable : callable
            The callable object to be executed.
        *args : positional arguments
            Positional arguments to pass to the callable.
        **kwargs : keyword arguments
            Keyword arguments to pass to the callable.
            
        Returns:
        --------
        Future
            A Future object representing the execution of the callable.
            
        Raises:
        -------
        RuntimeError
            If the pool has been shut down.
        """
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self._mutex)
        
        if self._shutdown.load():
            raise RuntimeError("cannot schedule new futures after shutdown")
            
        return self._submit(0.0, (callable, args, kwargs))
        
    def schedule(self, priority, callable, *args, **kwargs):
        """
        Submit a callable with a priority (lower values are executed first).
        
        This method schedules the callable to be executed based on the given priority
        and returns a Future object representing the execution.
        
        Parameters:
        -----------
        priority : float
            The priority of the task. Lower values indicate higher priority.
            Negative values create blocking jobs that prevent execution of non-negative priority jobs.
            Positive values enforce a waiting period (in milliseconds) after blocking jobs.
            Zero priority jobs have standard priority with no waiting period.
        callable : callable
            The callable object to be executed.
        *args : positional arguments
            Positional arguments to pass to the callable.
        **kwargs : keyword arguments
            Keyword arguments to pass to the callable.
            
        Returns:
        --------
        Future
            A Future object representing the execution of the callable.
            
        Raises:
        -------
        RuntimeError
            If the pool has been shut down.
            
        Notes:
        ------
        - Negative priority tasks block execution of zero and positive priority tasks
        - When a negative priority task completes, a timestamp is recorded
        - Positive priority tasks will wait their priority value (in ms) after the last blocking task
        - Multiple negative priority tasks will execute concurrently
        """
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self._mutex)
        
        if self._shutdown.load():
            raise RuntimeError("cannot schedule new futures after shutdown")

        return self._submit(priority, (callable, args, kwargs))
        
    def shutdown(self, wait=True):
        """
        Signal the executor that it should free any resources and shut down.
        
        This method should be called when the thread pool is no longer needed.
        Once shutdown is called, new tasks cannot be submitted to the pool.
        
        Parameters:
        -----------
        wait : bool, default=True
            If True, this method will block until all pending futures are done
            executing, including both running and queued futures.
            If False, the method returns immediately and resources will be freed 
            when all futures are done executing.
        
        Notes:
        ------
        - After shutdown, attempting to submit new tasks will raise RuntimeError
        - This method may be called multiple times safely
        - If an exception was raised by a future and not retrieved, it will be ignored
        """

        if wait:
            while len(self._futures) > 0:
                try:
                    self._futures.values()[0].result()
                except Exception:
                    pass

        self._shutdown.store(True)
        self._work_condition.notify_all()

        cdef unique_lock[mutex] m
        cdef Command command
        cdef Future future
        if not wait:
            lock_gil_friendly(m, self._queue_mutex)
            while not self._queue.empty():
                queue_pop_top(command, self._queue)
                #m.unlock()
                future = self._futures.pop(command.uuid, None)
                if future:
                    future.cancel()
                #lock_gil_friendly(m, self._queue_mutex)
            m.unlock()

    def __enter__(self):
        """
        Enter the runtime context for the ThreadPool.
        
        This method enables the ThreadPool to be used as a context manager
        with the 'with' statement, ensuring proper cleanup of resources.
        
        Returns:
        --------
        ThreadPool
            Returns the ThreadPool instance itself to be used in the context block.

        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context for the ThreadPool.
        
        This method is called automatically when exiting a 'with' block.
        It shuts down the ThreadPool and waits for all submitted tasks to complete.
        
        Parameters:
        -----------
        exc_type : Exception type or None
            The type of exception raised in the context block, if any
        exc_val : Exception or None
            The exception instance raised, if any
        exc_tb : traceback or None
            The traceback of the exception, if any
            
        Returns:
        --------
        bool
            False to indicate that any exceptions should be propagated
            
        Notes:
        ------
        - This method calls shutdown(wait=True) to ensure all tasks complete
        - Exceptions from the context block will be propagated
        - Exceptions from the workers are handled by report_error method
        """
        self.shutdown(wait=True)
        return False
        
    def map(self, fn, *iterables, timeout=None):
        """
        Apply the function to every item in the iterables in parallel.
        
        This method is similar to the built-in map() function but executes
        the function calls asynchronously using the thread pool. Results are
        yielded in the order of the original iterables as they become available.
        
        Parameters:
        -----------
        fn : callable
            A callable that will take as many arguments as there are iterables.
        *iterables : iterable objects
            One or more iterables containing the data to process.
        timeout : float or None, default=None
            The maximum number of seconds to wait for each result.
            If None, there is no timeout.
            
        Yields:
        -------
        Results from the function calls in the order of the input iterables.
            
        Raises:
        -------
        TimeoutError
            If the result isn't available within the given timeout.
        CancelledError
            If the future was cancelled.
        Exception
            Any exception raised by the callable function.
        """
        futures = [self.submit(fn, *args) for args in zip(*iterables)]
        for future in futures:
            yield future.result(timeout=timeout)
