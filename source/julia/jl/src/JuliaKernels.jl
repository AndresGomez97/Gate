module JuliaKernels

using CUDAdrv
using CuArrays
using CUDAnative

include("Structs.jl")
include("GateCollim_gpu.jl")

end