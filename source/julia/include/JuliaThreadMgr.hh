#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <dlfcn.h>

#include "julia.h"
#include "EmbedJulia.hh"

// Class Worker
class Worker {
    
    bool running = true;
    std::thread t;
    std::mutex mtx;
    std::condition_variable cond;
    std::deque<std::function<void()>> tasks;

public:
    
    Worker();
    ~Worker();

    template <typename F> auto spawn(const F& f) -> std::packaged_task<decltype(f())()> {
        std::packaged_task<decltype(f())()> task(f);
        {
            std::unique_lock<std::mutex> lock(mtx);
            tasks.push_back([&task] { task(); });
        }
        cond.notify_one();
        return task;
    }

    template <typename F> auto run(const F& f) -> decltype(f()) { return spawn(f).get_future().get(); }

private:
    
    void threadFunc() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx);
                while (tasks.empty() && running) {
                    cond.wait(lock);
                }
                if (!running) {
                    break;
                }
                task = std::move(tasks.front());
                tasks.pop_front();
            }
            task();
        }
    }
};

// Class Julia
class JuliaMgr {
    Worker worker;
public:
    static JuliaMgr& GetInstance() {
        static JuliaMgr instance;
        return instance;
    }
    JuliaMgr();
    virtual ~JuliaMgr();

    template <typename F> static auto spawn(const F& f) -> std::packaged_task<decltype(f())()> {
        return GetInstance().worker.spawn(f);
    }
    template <typename F> static auto run(const F& f) -> decltype(f()) { return GetInstance().worker.run(f); }
    static void run(const char* s) {
        return GetInstance().worker.run([&] { p_jl_eval_string(s); });
    }
};