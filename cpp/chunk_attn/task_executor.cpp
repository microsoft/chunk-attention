#include "task_executor.h"
#include "logging.h"

namespace GPT {

TaskExecutor::TaskExecutor(int num_threads)
  : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            std::function<void()> task;
            while (true) {
                {
                    std::unique_lock lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_) {
                        return;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

void TaskExecutor::enqueue(std::function<void()> f) {
    {
        std::lock_guard lg(queue_mutex_);
        tasks_.emplace(f);
    }
    condition_.notify_one();
}

TaskExecutor::~TaskExecutor() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

}
