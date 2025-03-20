# generated with pxdgen /usr/include/c++/11/mutex -x c++

cdef extern from "<mutex>" namespace "std" nogil:
    cppclass mutex:
        mutex()
        mutex(mutex&)
        mutex& operator=(mutex&)
        void lock()
        bint try_lock()
        void unlock()
    cppclass defer_lock_t:
        defer_lock_t()
    cppclass recursive_mutex:
        recursive_mutex()
        recursive_mutex(recursive_mutex&)
        recursive_mutex& operator=(recursive_mutex&)
        void lock()
        bint try_lock()
        void unlock()
    int try_lock[_Lock1, _Lock2, _Lock3](_Lock1&, _Lock2&, _Lock3 &...)
    void lock[_L1, _L2, _L3](_L1&, _L2&, _L3 &...)
    cppclass unique_lock[_Mutex]:
        ctypedef _Mutex mutex_type
        unique_lock()
        unique_lock(mutex_type&)
        unique_lock(mutex_type&, defer_lock_t)
        unique_lock(mutex_type&, try_to_lock_t)
        unique_lock(mutex_type&, adopt_lock_t)
        unique_lock(unique_lock&)
        unique_lock& operator=(unique_lock&)
        #unique_lock(unique_lock&&)
        #unique_lock& operator=(unique_lock&&)
        void lock()
        bint try_lock()
        void unlock()
        void swap(unique_lock&)
        mutex_type* release()
        bint owns_lock()
        mutex_type* mutex()
        mutex_type* get_lk "mutex"()  # Alias for mutex() to avoid name collision
    void swap[_Mutex](unique_lock[_Mutex]&, unique_lock[_Mutex]&)

cdef extern from "<condition_variable>" namespace "std" nogil:
    cppclass condition_variable:
        condition_variable()
        condition_variable(const condition_variable&)
        void notify_one()
        void notify_all()
        void wait(unique_lock[mutex]&)
        bint wait_for(unique_lock[mutex]&, long long)  # Simplified for microseconds

# Atomics definition with C++20 atomic wait support
cdef extern from "<atomic>" namespace "std" nogil:

    cdef enum memory_order:
        memory_order_relaxed
        memory_order_consume
        memory_order_acquire
        memory_order_release
        memory_order_acq_rel
        memory_order_seq_cst

    cdef cppclass atomic[T]:
        atomic()
        atomic(T)

        bint is_lock_free()
        void store(T)
        void store(T, memory_order)
        T load()
        T load(memory_order)
        T exchange(T)
        T exchange(T, memory_order)

        bint compare_exchange_weak(T&, T, memory_order, memory_order)
        bint compare_exchange_weak(T&, T, memory_order)
        bint compare_exchange_weak(T&, T)
        bint compare_exchange_strong(T&, T, memory_order, memory_order)
        bint compare_exchange_strong(T&, T, memory_order)
        bint compare_exchange_strong(T&, T)

        T fetch_add(T, memory_order)
        T fetch_add(T)
        T fetch_sub(T, memory_order)
        T fetch_sub(T)
        T fetch_and(T, memory_order)
        T fetch_and(T)
        T fetch_or(T, memory_order)
        T fetch_or(T)
        T fetch_xor(T, memory_order)
        T fetch_xor(T)

        T operator++()
        T operator++(int)
        T operator--()
        T operator--(int)

        bint operator==(atomic[T]&, atomic[T]&)
        bint operator==(atomic[T]&, T&)
        bint operator==(T&, atomic[T]&)
        bint operator!=(atomic[T]&, atomic[T]&)
        bint operator!=(atomic[T]&, T&)
        bint operator!=(T&, atomic[T]&)

        void wait(T, memory_order)
        void wait(T)
        void notify_one()
        void notify_all()

    cdef cppclass atomic_flag:
        atomic_flag()
        atomic_flag(bint)

        bint test(memory_order)
        bint test()
        bint test_and_set(memory_order)
        bint test_and_set()
        void clear(memory_order)
        void clear()

        void wait(bint, memory_order)
        void wait(bint)
        void notify_one()
        void notify_all()
