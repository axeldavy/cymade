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