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

#include "JuliaThreadMgr.hh"

Worker::Worker() : t{&Worker::threadFunc, this} {}

Worker::~Worker() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        running = false;
    }
    cond.notify_one();
    t.join();
}

JuliaMgr::JuliaMgr() {
    worker.run([] {
        //p_jl_init();
        p_jl_init_with_image("/home/agmez/julia-1.3.1/bin", "/home/agmez/gate/Gate/julia_environments/custom_sysimage.so");
    });
}

JuliaMgr::~JuliaMgr() {
    worker.run([] {
        p_jl_atexit_hook(0);
    });
}

