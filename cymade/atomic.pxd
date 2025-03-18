from libc.stdint cimport uint64_t
from .cpp_includes cimport mutex, unique_lock, defer_lock_t

# from DearCyGui
cdef void lock_gil_friendly_block(unique_lock[mutex] &m) noexcept

cdef inline void lock_gil_friendly(unique_lock[mutex] &m,
                                   mutex &target_mutex) noexcept:
    """
    Must be called to lock our mutexes whenever we hold the gil
    """
    m = unique_lock[mutex](target_mutex, defer_lock_t())
    # Fast path
    if m.try_lock():
        return
    # Slow path
    lock_gil_friendly_block(m)


