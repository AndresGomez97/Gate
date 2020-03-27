module GateCommonKernels

using CUDAdrv
using CuArrays
using CUDAnative

export CUStackParticle, JStackParticle, Activities, Int3, Float3, f_kernel_brent_init, f_kernel_voxelized_source_b2b

# Cuda StackParticle struct
struct CUStackParticle
	E::CuPtr{Float32}
	dx::CuPtr{Float32}
    dy::CuPtr{Float32}
    dz::CuPtr{Float32}
    px::CuPtr{Float32}
    py::CuPtr{Float32}
    pz::CuPtr{Float32}
    t::CuPtr{Float32}
    type::CuPtr{UInt16}
    eventID::CuPtr{UInt32}
    trackID::CuPtr{UInt32}
    seed::CuPtr{UInt32}
    active::CuPtr{Char}
	endsimu::CuPtr{Char}
	table_x_brent::CuPtr{Int64}
    size::UInt32
end

# Julia StackParticle struct
struct JStackParticle
	E::CuArray{Float32}
	dx::CuArray{Float32}
    dy::CuArray{Float32}
    dz::CuArray{Float32}
    px::CuArray{Float32}
    py::CuArray{Float32}
    pz::CuArray{Float32}
    t::CuArray{Float32}
    type::CuArray{UInt16}
    eventID::CuArray{UInt32}
    trackID::CuArray{UInt32}
    seed::CuArray{UInt32}
    active::CuArray{Char}
    endsimu::CuArray{Char}
    table_x_brent::CuArray{Int64}
    size::UInt32
end

# Activities struct
struct Activities
    nb_activities::UInt32
    tot_activity::Float32
    act_index::CuPtr{UInt32}
    act_cdf::CuPtr{Float32}
end

# Int3 struct
struct Int3
    x::Int32
    y::Int32
    z::Int32
end

# Float3 struct
struct Float3
    x::Float32
    y::Float32
    z::Float32
end

# Kernel kernl_brent_init from GateCommon_fun.cu
function f_kernel_brent_init(E::Array{Float32},px::Array{Float32},py::Array{Float32},pz::Array{Float32},dx::Array{Float32},dy::Array{Float32},dz::Array{Float32},t::Array{Float32},type::Array{UInt16},eventID::Array{UInt32},trackID::Array{UInt32},seed::Array{UInt32},active::Array{Char},endsimu::Array{Char},table_x_brent::Array{Int64},size::UInt32,nBlocks::CuDim,nThreads::CuDim)
    
    ctx = CUDAnative.context()

    md = CuModuleFile(joinpath(@__DIR__,"..","..","gpu","src","GateGPUManager.ptx"))

    kernel_brent_init = CuFunction(md,"kernel_brent_init")

    d_E = CuArray(E)
    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_dx = CuArray(dx)
    d_dy = CuArray(dy)
    d_dz = CuArray(dz)
    d_t = CuArray(t)
    d_type = CuArray(type)
    d_eventID = CuArray(eventID)
    d_trackID = CuArray(trackID)
    d_seed = CuArray(seed)
    d_active = CuArray(active)
    d_endsimu = CuArray(endsimu)
    d_table_x_brent = CuArray(table_x_brent)

    cudacall(kernel_brent_init,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{UInt16},CuPtr{UInt32},CuPtr{UInt32},CuPtr{UInt32},CuPtr{Char},CuPtr{Char},CuPtr{Int64},UInt32}, d_E,d_px,d_py,d_pz,d_dx,d_dy,d_dz,d_t,d_type,d_eventID,d_trackID,d_seed,d_active,d_endsimu,d_table_x_brent,size;blocks=nBlocks,threads=nThreads)
end

# Kernel kernel_voxelized_source_b2b from GateCommon_fun.cu
function f_kernel_voxelized_source_b2b(g1::JStackParticle, g2::JStackParticle, act::Activities, E::Float32, size_in_vox::Int3, voxel_size::Float3, nBlocks::CuDim, nThreads::CuDim)
    
    ctx = CUDAnative.context()

    md = CuModuleFile(joinpath(@__DIR__,"..","..","gpu","src","GateGPUManager.ptx"))

    kernel_voxelized_sorce_b2b = CuFunction(md,"kernel_voxelized_source_b2b")

    cudacall(kernel_voxelized_source_b2b,Tuple{StackParticle,StackParticle, Activities, Float32, Int3, Float3}, g1, g2, act, E, size_in_vox, voxel_size;blocks=nBlocks,threads=nThreads)
end

end
