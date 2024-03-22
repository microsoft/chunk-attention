#pragma once

#include <atomic>

class SpinLock {

  public:
    SpinLock(){};
    SpinLock(const SpinLock&){};
    SpinLock& operator=(const SpinLock&) { return *this; };

    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // Spin until the lock is acquired
        }
    }

    void unlock() { flag_.clear(std::memory_order_release); }

  private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
    ;
};