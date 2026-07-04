#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace cosq::qnet {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    void parallel_for(size_t count, const std::function<void(size_t)>& fn);

    size_t num_threads() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<size_t> remaining_{0};
    bool stop_ = false;

    void worker_loop();
};

} // namespace cosq::qnet
