#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future> 

namespace GPT {

class TaskExecutor {
  public:
    TaskExecutor(int num_workers);
    virtual ~TaskExecutor();
 
    void enqueue(std::function<void()> f);

  private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;

    bool stop_;
};
}