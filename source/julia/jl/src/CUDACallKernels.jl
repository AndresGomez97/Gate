module GateKernels

using CUDA

include("Structs.jl")

# Function that return the module file to perform cudacalls to the ptx
function ctx()
    ctx = CUDA.context()
    md = CuModuleFile(joinpath(@__DIR__,"..","..","..","gpu","src","GateGPUManager.ptx"))
    return md
end

# Function for calling the 3 SPECT kernels
function call_all(px::Array{Float32},py::Array{Float32},pz::Array{Float32},
                      dx::Array{Float32},dy::Array{Float32},dz::Array{Float32},
                      entry_collim_y::Array{Float32},entry_collim_z::Array{Float32},
                      exit_collim_y::Array{Float32},exit_collim_z::Array{Float32},
                      hole::Array{Int32},size_y::UInt32,size_z::UInt32,planeToProject::Float32,
                      particle_size::Int32,nBlocks::CuDim,nThreads::CuDim)

    # Get the 3 kernels address to cudacall them
    md = ctx()
    kernel_map_entry = CuFunction(md,"kernel_map_entry")
    kernel_map_projection = CuFunction(md,"kernel_map_projection")
    kernel_map_exit = CuFunction(md,"kernel_map_exit")

    # Host to Device memcpy
    d_px = CuArray(px)
    d_py = CuArray(py) 
    d_pz = CuArray(pz) 
    d_dx = CuArray(dx)
    d_dy = CuArray(dy)
    d_dz = CuArray(dz)
    d_hole = CuArray(hole) 
    d_entry_collim_y = CuArray(entry_collim_y)
    d_entry_collim_z = CuArray(entry_collim_z)
    d_exit_collim_y = CuArray(exit_collim_y)
    d_exit_collim_z = CuArray(exit_collim_z) 

    # Cudacall kernel_map_entry
    cudacall(kernel_map_entry,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},UInt32,UInt32,Int32}, 
            d_px, d_py, d_pz, d_entry_collim_y, d_entry_collim_z, d_hole, size_y, size_z, particle_size;blocks=nBlocks,threads=nThreads)
    # Cudacall kernel_map_projection
    cudacall(kernel_map_projection,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},Float32,UInt32}, 
            d_px, d_py, d_pz, d_dx, d_dy, d_dz, d_hole, planeToProject,particle_size;blocks=nBlocks,threads=nThreads)
    # Cudacall kernel_map_exit
    cudacall(kernel_map_exit,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},UInt32,UInt32,Int32}, 
            d_px, d_py, d_pz, d_exit_collim_y, d_exit_collim_z, d_hole, size_y, size_z, particle_size;blocks=nBlocks,threads=nThreads)

    # Devie to Host memcpy
    px = Array{Float32}(d_px)
    py = Array{Float32}(d_py)
    pz = Array{Float32}(d_pz)
    hole = Array{Int32}(d_hole)

    return (px,py,pz,hole)
end

#------------------------------------- GateCollim_gpu.cu -----------------------------------------
# Cudacall kernel kernel_map_entry from GateCollim_gpu.cu
function f_kernel_map_entry(px::Array{Float32},py::Array{Float32},pz::Array{Float32},
                            entry_collim_y::Array{Float32},entry_collim_z::Array{Float32},
                            hole::Array{Int32},y_size::UInt32,z_size::UInt32,particle_size::Int32,
                            nBlocks::CuDim,nThreads::CuDim) 
    md = ctx()
    kernel_map_entry = CuFunction(md,"kernel_map_entry")

    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_hole = CuArray(hole)
    d_entry_collim_y = CuArray(entry_collim_y)
    d_entry_collim_z = CuArray(entry_collim_z)

    cudacall(kernel_map_entry,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},UInt32,UInt32,Int32}, d_px, d_py, d_pz, d_entry_collim_y, d_entry_collim_z, d_hole, y_size, z_size, particle_size;blocks=nBlocks,threads=nThreads)
    return Array{Int32}(d_hole)
end

# Cudacall kernel kernel_map_projection from GateCollim_gpu.cu
function f_kernel_map_projection(px::Array{Float32},py::Array{Float32},pz::Array{Float32},
                                 dx::Array{Float32},dy::Array{Float32},dz::Array{Float32},
                                 hole::Array{Int32},planeToProject::Float32,particle_size::Int32,
                                 nBlocks::CuDim,nThreads::CuDim)
    md = ctx() 
    kernel_map_projection = CuFunction(md,"kernel_map_projection")

    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_dx = CuArray(dx)
    d_dy = CuArray(dy)
    d_dz = CuArray(dz)
    d_hole = CuArray(hole)

    cudacall(kernel_map_projection,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},Float32,UInt32}, d_px, d_py, d_pz, d_dx, d_dy, d_dz, d_hole, planeToProject,UInt32(particle_size);blocks=nBlocks,threads=nThreads)
    return (Array{Float32}(d_px),Array{Float32}(d_py),Array{Float32}(d_pz))
end

# Cudacall kernel kernel_map_exit from GateCollim_gpu.cu
function f_kernel_map_exit(px::Array{Float32},py::Array{Float32},pz::Array{Float32},
                           exit_collim_y::Array{Float32},exit_collim_z::Array{Float32},hole::Array{Int32},
                           y_size::UInt32,z_size::UInt32,particle_size::Int32,
                           nBlocks::CuDim,nThreads::CuDim)
    md = ctx()
    kernel_map_exit = CuFunction(md,"kernel_map_exit")

    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_hole = CuArray(hole)
    d_exit_collim_y = CuArray(exit_collim_y)
    d_exit_collim_z = CuArray(exit_collim_z)

    cudacall(kernel_map_exit,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},UInt32,UInt32,Int32}, d_px, d_py, d_pz, d_exit_collim_y, d_exit_collim_z, d_hole, y_size, z_size, particle_size;blocks=nBlocks,threads=nThreads)
    return Array{Int32}(d_hole)
end
#-----------------------------------------------------------------------------------------------------
