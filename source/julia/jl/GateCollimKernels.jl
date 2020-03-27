module GateCollimKernels

using CUDAdrv
using CuArrays
using CUDAnative

export f_kernel_map_entry, f_kernel_map_projection, f_kernel_map_exit

# Kernel kernel_map_entry from GateCollim_gpu.cu
function f_kernel_map_entry(px::Array{Float32,2},py::Array{Float32,2},pz::Array{Float32,2},entry_collim_y::Array{Float32,2},entry_collim_z::Array{Float32,2},hole::Array{Int32,2},y_size::UInt32,z_size::UInt32,particle_size::Int32,nBlocks::CuDim,nThreads::CuDim)
    
    ctx = CUDAnative.context()

    md = CuModuleFile(joinpath(@__DIR__,"..","..","gpu","src","GateGPUManager.ptx"))
    
    kernel_map_entry = CuFunction(md,"kernel_map_entry")

    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_hole = CuArray(hole)
    d_entry_collim_y = CuArray(entry_collim_y)
    d_entry_collim_z = CuArray(entry_collim_z)
    
    cudacall(kernel_map_entry,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},UInt32,UInt32,Int32}, d_px, d_py, d_pz, d_entry_collim_y, d_entry_collim_z, d_hole, y_size, z_size, particle_size;blocks=nBlocks,threads=nThreads)
    
    return d_hole
end

# Kernel kernel_map_projection from GateCollim_gpu.cu
function f_kernel_map_projection(px::Array{Float32,2},py::Array{Float32,2},pz::Array{Float32,2},dx::Array{Float32,2},dy::Array{Float32,2},dz::Array{Float32,2},hole::Array{Int32,2},planeToProject::Float32,particle_size::UInt32,nBlocks::CuDim,nThreads::CuDim)
    
    ctx = CUDAnative.context()

    md = CuModuleFile(joinpath(@__DIR__,"..","..","gpu","src","GateGPUManager.ptx"))    
    
    kernel_map_projection = CuFunction(md,"kernel_map_projection")

    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_dx = CuArray(dx)
    d_dy = CuArray(dy)
    d_dz = CuArray(dz)
    d_hole = CuArray(hole)

    cudacall(kernel_map_projection,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},Float32,UInt32}, d_px, d_py, d_pz, d_dx, d_dy, d_dz, d_hole, planeToProject,particle_size;blocks=nBlocks,threads=nThreads)

    return d_px,d_py,d_pz

end

# Kernel kernel_map_exit from GateCollim_gpu.cu
function f_kernel_map_exit(px::Array{Float32,2},py::Array{Float32,2},pz::Array{Float32,2},entry_collim_y::Array{Float32,2},entry_collim_z::Array{Float32,2},hole::Array{Int32,2},y_size::UInt32,z_size::UInt32,particle_size::Int32,nBlocks::CuDim,nThreads::CuDim)
    
    ctx = CUDAnative.context()

    md = CuModuleFile(joinpath(@__DIR__,"..","..","gpu","src","GateGPUManager.ptx"))

    kernel_map_exit = CuFunction(md,"kernel_map_exit")

    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_hole = CuArray(hole)
    d_entry_collim_y = CuArray(entry_collim_y)
    d_entry_collim_z = CuArray(entry_collim_z)
    
    cudacall(kernel_map_exit,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},UInt32,UInt32,Int32}, d_px, d_py, d_pz, d_entry_collim_y, d_entry_collim_z, d_hole, y_size, z_size, particle_size;blocks=nBlocks,threads=nThreads)
    
    return d_hole
end

end