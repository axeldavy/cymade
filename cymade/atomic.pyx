#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: embedsignature=False
#cython: cdivision=True
#cython: cdivision_warnings=False
#cython: always_allow_keywords=False
#cython: profile=False
#cython: infer_types=False
#cython: initializedcheck=False
#cython: c_line_in_traceback=False
#cython: auto_pickle=False
#cython: freethreading_compatible=True
#distutils: language=c++


from libc.math cimport pow as cpow
from libc.stdint cimport int64_t, uint64_t
from libcpp.atomic cimport atomic

from .cpp_includes cimport mutex, unique_lock, defer_lock_t


# From DearCyGui
cdef void lock_gil_friendly_block(unique_lock[mutex] &m) noexcept:
    """
    Same as lock_gil_friendly, but blocks until the job is done.
    We inline the fast path, but not this one as it generates
    more code.
    """
    # Release the gil to enable python processes eventually
    # holding the lock to run and release it.
    # Block until we get the lock
    cdef bint locked = False
    while not(locked):
        with nogil:
            # Block until the mutex is released
            m.lock()
            # Unlock to prevent deadlock if another
            # thread holding the gil requires m
            # somehow
            m.unlock()
        locked = m.try_lock()

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


cdef class U64:
    """
    Integer Atomic counter, non negative
    Can store up to 2^63 - 1

    This class is full thread safe.

    During creation the on_zero optional argument can
    be passed to specify a function to execute when the
    atomic counter reaches zero.
    """
    cdef atomic[int64_t] value
    cdef object on_zero

    def __init__(self, int64_t init_value=0, object on_zero=None):
        if init_value < 0:
            raise ValueError("Initial value must be non-negative")
        self.value.store(init_value)
        self.on_zero = on_zero
    
    def __dealloc__(self):
        pass

    def __int__(self):
        return self.value.load()    
    
    def __float__(self):
        return float(self.value.load())

    def __iadd__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        while True:
            cur_value = self.value.load()
            new_value = cur_value + other_value
            if new_value < 0:
                raise ValueError("Cannot decrement below zero")
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self

    def __isub__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
            other_value = -other_value  # Convert to decrement
        except:
            raise NotImplemented
        while True:
            cur_value = self.value.load()
            new_value = cur_value + other_value
            if new_value < 0:
                raise ValueError("Cannot decrement below zero")
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self

    """ No check versions
    def __iadd__(self, other):
        cdef int64_t other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if self.value.fetch_add(other_value) + other_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self

    def __isub__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if self.value.fetch_sub(other_value) - other_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self
    """
    
    def __imul__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value < 0:
            raise ValueError("Multiplication by negative value is not allowed")
        while True:
            cur_value = self.value.load()
            new_value = cur_value * other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self
    
    def __ifloordiv__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        if other_value < 0:
            raise ValueError("Division by negative value is not allowed")
        while True:
            cur_value = self.value.load()
            new_value = cur_value // other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self
    
    def __imod__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        while True:
            cur_value = self.value.load()
            new_value = cur_value % other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self
    
    def __ipow__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value < 0:
            raise ValueError("Exponentiation by negative value is not allowed")
        while True:
            cur_value = self.value.load()
            new_value = pow(cur_value, other_value)
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self
    
    def __ilshift__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value < 0:
            raise ValueError("Left shift by negative value is not allowed")
        while True:
            cur_value = self.value.load()
            if other_value >= 64:  # prevent undefined behavior
                new_value = 0
            else:
                new_value = cur_value << other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self
    
    def __irshift__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value < 0:
            raise ValueError("Right shift by negative value is not allowed")
        while True:
            cur_value = self.value.load()
            if other_value >= 64:  # prevent undefined behavior
                new_value = 0
            else:
                new_value = cur_value >> other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self
    
    def __iand__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        while True:
            cur_value = self.value.load()
            new_value = cur_value & other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self
    
    def __ior__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        while True:
            cur_value = self.value.load()
            new_value = cur_value | other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self
    
    def __ixor__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        while True:
            cur_value = self.value.load()
            new_value = cur_value ^ other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        if new_value == 0 and self.on_zero is not None:
            self.on_zero()
        return self

cdef class Float:
    """
    Float value that is thread safe
    """
    cdef atomic[double] value

    def __init__(self, double init_value=0.0):
        self.value.store(init_value)
    
    def __dealloc__(self):
        pass

    def __float__(self):
        return self.value.load()
    
    def __int__(self):
        return int(self.value.load())

    def __iadd__(self, other):
        cdef double other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        self.value.fetch_add(other_value)
        return self

    def __isub__(self, other):
        cdef double cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        self.value.fetch_sub(other_value)
        return self
        
    def __imul__(self, other):
        cdef double cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        while True:
            cur_value = self.value.load()
            new_value = cur_value * other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __itruediv__(self, other):
        cdef double cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value == 0:
            raise ZeroDivisionError("float division by zero")
        while True:
            cur_value = self.value.load()
            new_value = cur_value / other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __ifloordiv__(self, other):
        cdef double cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value == 0:
            raise ZeroDivisionError("float divmod()")
        while True:
            cur_value = self.value.load()
            new_value = (cur_value // other_value)
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __imod__(self, other):
        cdef double cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value == 0:
            raise ZeroDivisionError("float modulo")
        while True:
            cur_value = self.value.load()
            new_value = cur_value % other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __ipow__(self, other):
        cdef double cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        while True:
            cur_value = self.value.load()
            new_value = cpow(cur_value, other_value)
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self

cdef class Int:
    """
    Integer Atomic counter with arbitrary precision (Python int).
    This class is full thread safe.
    """
    cdef object value  # Python int object
    cdef mutex m

    def __init__(self, object init_value=0):
        self.value = init_value
    
    def __dealloc__(self):
        pass

    def __int__(self):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        return self.value
    
    def __float__(self):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        return float(self.value)

    def __iadd__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        self.value += other_value
        return self

    def __isub__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        self.value -= other_value
        return self
    
    def __imul__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        self.value *= other_value
        return self
    
    def __ifloordiv__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        if other_value == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        self.value //= other_value
        return self
    
    def __imod__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        if other_value == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        self.value %= other_value
        return self
    
    def __ipow__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        self.value **= other_value
        return self
    
    def __ilshift__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        self.value <<= other_value
        return self
    
    def __irshift__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        self.value >>= other_value
        return self
    
    def __iand__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        self.value &= other_value
        return self
    
    def __ior__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        self.value |= other_value
        return self
    
    def __ixor__(self, other):
        cdef unique_lock[mutex] m
        lock_gil_friendly(m, self.m)
        cdef object other_value
        try:
            other_value = int(other)
        except:
            raise NotImplemented
        self.value ^= other_value
        return self

cdef class I64:
    """
    Integer Atomic counter, 64-bit signed.
    Can store from -2^63 to 2^63 - 1
    
    This class is full thread safe.
    """
    cdef atomic[int64_t] value

    def __init__(self, int64_t init_value=0):
        self.value.store(init_value)
    
    def __dealloc__(self):
        pass

    def __int__(self):
        return self.value.load()    
    
    def __float__(self):
        return float(self.value.load())

    def __iadd__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        self.value.fetch_add(other_value)
        return self

    def __isub__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        self.value.fetch_sub(other_value)
        return self
        
    def __imul__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        while True:
            cur_value = self.value.load()
            new_value = cur_value * other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __ifloordiv__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        while True:
            cur_value = self.value.load()
            new_value = cur_value // other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __imod__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        while True:
            cur_value = self.value.load()
            new_value = cur_value % other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __ipow__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value < 0:
            raise ValueError("Exponentiation by negative value is not allowed")
        while True:
            cur_value = self.value.load()
            new_value = pow(cur_value, other_value)
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __ilshift__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value < 0:
            raise ValueError("Left shift by negative value is not allowed")
        while True:
            cur_value = self.value.load()
            new_value = cur_value << other_value
            if other_value >= 64:  # prevent undefined behavior
                new_value = 0
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __irshift__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        if other_value < 0:
            raise ValueError("Right shift by negative value is not allowed")
        while True:
            cur_value = self.value.load()
            if other_value >= 64:  # prevent undefined behavior
                new_value = 0 if cur_value >= 0 else -1  # arithmetic shift for signed
            else:
                new_value = cur_value >> other_value
            if self.value.compare_exchange_weak(cur_value, new_value):
                break
        return self
    
    def __iand__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        self.value.fetch_and(other_value)
        return self
    
    def __ior__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        self.value.fetch_or(other_value)
        return self
    
    def __ixor__(self, other):
        cdef int64_t cur_value, new_value, other_value
        try:
            other_value = other
        except:
            raise NotImplemented
        self.value.fetch_xor(other_value)
        return self