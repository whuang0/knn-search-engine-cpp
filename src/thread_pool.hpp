#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

// Fixed-size thread pool. Workers block on a condition variable until tasks
// are enqueued. On destruction, the pool drains remaining tasks then joins.
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    size_t getThreadCount() const { return workers.size(); }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};