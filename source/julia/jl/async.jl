using CUDAnative
using CUDAdrv

async_send(data::Ptr{Cvoid}) = ccall(:uv_async_send, Cint, (Ptr{Cvoid},), data)

function launch_host_func(f, stream::CuStream=CuDefaultStream())
    cond = Base.AsyncCondition() do asyc_cond
        f()
        close(asyc_cond)
    end

    callback = @cfunction(async_send, Cint, (Ptr{Cvoid},))
    CUDAdrv.cuLaunchHostFunc(stream, callback, cond)
end

c = Condition()
launch_host_func() do
    println("42")
    notify(c)
end
wait(c)