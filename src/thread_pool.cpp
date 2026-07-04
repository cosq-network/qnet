#include <qnet/thread_pool.hpp>

namespace cosq::qnet {

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    if (num_threads == 0) num_threads = 1;

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
        remaining_.fetch_sub(1, std::memory_order_release);
        cv_.notify_all();
    }
}

void ThreadPool::parallel_for(size_t count,
                               const std::function<void(size_t)>& fn) {
    if (count == 0) return;

    size_t num_threads = workers_.size();
    if (num_threads == 0 || count == 1) {
        for (size_t i = 0; i < count; ++i) fn(i);
        return;
    }

    remaining_.store(count, std::memory_order_release);

    size_t chunk = (count + num_threads - 1) / num_threads;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, count);
            if (start >= end) break;

            tasks_.emplace([start, end, &fn] {
                for (size_t i = start; i < end; ++i) fn(i);
            });
        }
    }
    cv_.notify_all();

    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return remaining_.load(std::memory_order_acquire) == 0; });
    }
}

} // namespace cosq::qnet
