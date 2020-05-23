
#------------------------------------- GateCollim_gpu.cu -----------------------------------------
# vector_dot method
function vector_dot(u::Float3,v::Float3)
    return (u.x * v.x) + (u.y * v.y) + (u.z * v.z)
end

# vector_sub method
function vector_sub(u::Float3,v::Float3)
    return Float3(u.x - v.x, u.y - v.y, u.z - v.z)
end

# vector_add method
function vector_add(u::Float3,v::Float3)
    return Float3(u.x + v.x, u.y + v.y, u.z + v.z)
end

# vector_mag method
function vector_mag(u::Float3,a::Float32)
    return Float3(u.x * a, u.y * a, u.z * a)
end

# binary_search
function binary_search(position::Float32,tab,endIdx::UInt32)
    begIdx = 1
    medIdx = floor(Int,endIdx / 2)
    while endIdx-begIdx > 1
        position < tab[medIdx] ? begIdx = medIdx : endIdx = medIdx
        medIdx = floor(Int,(begIdx + endIdx) / 2)
    end
    return medIdx-1
end

# Check if there is in col bounds
function is_in(point,col)
    return point > col[1] || point < col[end]
end

# Kernel kernel_map_entry from GateCollim_gpu.cu
function kernel_map_entry(d_py,d_pz,
                          d_entry_collim_y,d_entry_collim_z,d_hole,
                          size_y::UInt32,size_z::UInt32,particle_size::Int32)
    
    id = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if id > particle_size 
        return
    elseif is_in(d_py[id], d_entry_collim_y) || is_in(d_pz[id], d_entry_collim_z)
        d_hole[id] = -1
        return
    end
    
    index_entry_y = binary_search(d_py[id],d_entry_collim_y,size_y)
    index_entry_z = binary_search(d_pz[id],d_entry_collim_z,size_z)
    
    d_hole[id] = iseven(index_entry_y) && iseven(index_entry_z) ? index_entry_y * size_z + index_entry_z : -1
    return
end

# Kernel kernel_map_projection from GateCollim_gpu.cu
function kernel_map_projection(d_px, d_py, d_pz, d_dx, d_dy, d_dz, d_hole, planeToProject, particle_size)
    
    id = (blockIdx().x-1) * blockDim().x + threadIdx().x
    
    if id > particle_size || d_hole[id] == -1
        return 
    end

    n = Float3(-1.0, 0.0, 0.0)
    v0 = Float3(planeToProject, 0.0, 0.0)
    d = Float3(d_dx[id], d_dy[id], d_dz[id])
    p = Float3(d_px[id], d_py[id], d_pz[id])
    s = vector_dot(n, vector_sub(v0, p)) / vector_dot(n, d)
    newp = vector_add(p, vector_mag(d, s))
    d_px[id] = newp.x
    d_py[id] = newp.y
    d_pz[id] = newp.z
    return
end

# Kernel kernel_map_exit from GateCollim_gpu.cu
function kernel_map_exit(d_py,d_pz,
                         d_exit_collim_y,d_exit_collim_z,d_hole,
                         size_y::UInt32,size_z::UInt32,particle_size::Int32)

    id = (blockIdx().x-1) * blockDim().x + threadIdx().x

    if id > particle_size || d_hole[id] == -1
        return
    elseif is_in(d_py[id], d_exit_collim_y) || is_in(d_pz[id], d_exit_collim_z)
        d_hole[id] = -1
        return 
    end

    index_exit_y = binary_search(d_py[id],d_exit_collim_y,size_y)
    index_exit_z = binary_search(d_pz[id],d_exit_collim_z,size_z)

    if isodd(index_exit_y) || isodd(index_exit_z) || d_hole[id] != index_exit_y * size_z + index_exit_z
        d_hole[id] = -1
        return
    end   
end
#-----------------------------------------------------------------------------------------------


#------------------------------------- Kernel calls --------------------------------------------
function call_all(px::Array{Float32},py::Array{Float32},pz::Array{Float32},
                  dx::Array{Float32},dy::Array{Float32},dz::Array{Float32},
                  entry_collim_y::Array{Float32},entry_collim_z::Array{Float32},
                  exit_collim_y::Array{Float32},exit_collim_z::Array{Float32},
                  hole::Array{Int32},size_y::UInt32,size_z::UInt32,planeToProject::Float32,
                  particle_size::Int32,nBlocks::CuDim,nThreads::CuDim)
    
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

    @cuda blocks=nBlocks threads=nThreads kernel_map_entry(d_py, d_pz, d_entry_collim_y, d_entry_collim_z, d_hole, size_y, size_z, particle_size)
    @cuda blocks=nBlocks threads=nThreads kernel_map_projection(d_px, d_py, d_pz, d_dx, d_dy, d_dz, d_hole, planeToProject, particle_size)
    @cuda blocks=nBlocks threads=nThreads kernel_map_exit(d_py, d_pz, d_exit_collim_y, d_exit_collim_z, d_hole, size_y, size_z, particle_size)
    
    return (Array{Float32}(d_px),Array{Float32}(d_py),Array{Float32}(d_pz),Array{Int32}(d_hole))
end

function call_entry(px::Array{Float32},py::Array{Float32},pz::Array{Float32},
                    entry_collim_y::Array{Float32},entry_collim_z::Array{Float32},
                    hole::Array{Int32},y_size::UInt32,z_size::UInt32,particle_size::Int32,
                    nBlocks::CuDim,nThreads::CuDim)
               
    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_hole = CuArray(hole)
    d_entry_collim_y = CuArray(entry_collim_y)
    d_entry_collim_z = CuArray(entry_collim_z) 

    @cuda blocks=nBlocks threads=nThreads kernel_map_entry(d_py, d_pz, d_entry_collim_y, d_entry_collim_z, d_hole, y_size, z_size, particle_size)
    return Array{Int32}(d_hole)
end

function call_projection(px::Array{Float32},py::Array{Float32},pz::Array{Float32},
                         dx::Array{Float32},dy::Array{Float32},dz::Array{Float32},
                         hole::Array{Int32},planeToProject::Float32,particle_size::Int32,
                         nBlocks::CuDim,nThreads::CuDim)
    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_dx = CuArray(dx)
    d_dy = CuArray(dy)
    d_dz = CuArray(dz)
    d_hole = CuArray(hole)

    @cuda blocks=nBlocks threads=nThreads kernel_map_projection(d_px, d_py, d_pz, d_dx, d_dy, d_dz, d_hole, planeToProject, particle_size)
    return (Array{Float32}(d_px),Array{Float32}(d_py),Array{Float32}(d_pz))
end

function call_exit(px::Array{Float32},py::Array{Float32},pz::Array{Float32},
                   exit_collim_y::Array{Float32},exit_collim_z::Array{Float32},
                   hole::Array{Int32},y_size::UInt32,z_size::UInt32,particle_size::Int32,
                   nBlocks::CuDim,nThreads::CuDim)

    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_hole = CuArray(hole)
    d_exit_collim_y = CuArray(exit_collim_y)
    d_exit_collim_z = CuArray(exit_collim_z) 

    @cuda blocks=nBlocks threads=nThreads kernel_map_exit(d_py, d_pz, d_exit_collim_y, d_exit_collim_z, d_hole, y_size, z_size, particle_size)
    return Array{Int32}(d_hole)
end

#-----------------------------------------------------------------------------------------------