
#------------------------------------- GateCollim_gpu.cu -----------------------------------------
# Kernel kernel_map_entry from GateCollim_gpu.cu
function kernel_map_entry(px::Array{Float32},py::Array{Float32},pz::Array{Float32},entry_collim_y::Array{Float32},entry_collim_z::Array{Float32},hole::Array{Int32},y_size::UInt32,z_size::UInt32,particle_size::Int32,nBlocks::Int32,nThreads::Int32)
    
    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_hole = CuArray(hole)
    d_entry_collim_y = CuArray(entry_collim_y)
    d_entry_collim_z = CuArray(entry_collim_z)
    
    return Array{Int32}(d_hole)
end

# Kernel kernel_map_projection from GateCollim_gpu.cu
function f_kernel_map_projection(px::Array{Float32},py::Array{Float32},pz::Array{Float32},dx::Array{Float32},dy::Array{Float32},dz::Array{Float32},hole::Array{Int32},planeToProject::Float32,particle_size::UInt32,nBlocks::Int32,nThreads::Int32)
    
    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_dx = CuArray(dx)
    d_dy = CuArray(dy)
    d_dz = CuArray(dz)
    d_hole = CuArray(hole)

    return (Array{Float32}(d_px),Array{Float32}(d_py),Array{Float32}(d_pz))
end

# Kernel kernel_map_exit from GateCollim_gpu.cu
function f_kernel_map_exit(px::Array{Float32},py::Array{Float32},pz::Array{Float32},exit_collim_y::Array{Float32},exit_collim_z::Array{Float32},hole::Array{Int32},y_size::UInt32,z_size::UInt32,particle_size::Int32,nBlocks::Int32,nThreads::Int32)

    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_hole = CuArray(hole)
    d_exit_collim_y = CuArray(exit_collim_y)
    d_exit_collim_z = CuArray(exit_collim_z)

    return Array{Int32}(d_hole)
end
#-----------------------------------------------------------------------------------------------