using CUDAdrv
using CuArrays
using CUDAnative

include("Structs.jl")

##################################################################################################
################################## Calling kernel methods ########################################
##################################################################################################

# Context and module
function ctx()
    ctx = CUDAnative.context()
    md = CuModuleFile(joinpath(@__DIR__,"..","..","gpu","src","GateGPUManager.ptx"))
    return md
end


#------------------------------------- GateCollim_gpu.cu -----------------------------------------

# Kernel kernel_map_entry from GateCollim_gpu.cu
function f_kernel_map_entry(px::Array{Float32},py::Array{Float32},pz::Array{Float32},entry_collim_y::Array{Float32},entry_collim_z::Array{Float32},hole::Array{Int32},y_size::UInt32,z_size::UInt32,particle_size::Int32,nBlocks::CuDim,nThreads::CuDim) 
    md = ctx()
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
function f_kernel_map_projection(px::Array{Float32},py::Array{Float32},pz::Array{Float32},dx::Array{Float32},dy::Array{Float32},dz::Array{Float32},hole::Array{Int32},planeToProject::Float32,particle_size::UInt32,nBlocks::CuDim,nThreads::CuDim)
    md = ctx() 
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
function f_kernel_map_exit(px::Array{Float32},py::Array{Float32},pz::Array{Float32},exit_collim_y::Array{Float32},exit_collim_z::Array{Float32},hole::Array{Int32},y_size::UInt32,z_size::UInt32,particle_size::Int32,nBlocks::CuDim,nThreads::CuDim)
    md = ctx()
    kernel_map_exit = CuFunction(md,"kernel_map_exit")

    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_hole = CuArray(hole)
    d_exit_collim_y = CuArray(exit_collim_y)
    d_exit_collim_z = CuArray(exit_collim_z)
    
    cudacall(kernel_map_exit,Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},CuPtr{Int32},UInt32,UInt32,Int32}, d_px, d_py, d_pz, d_exit_collim_y, d_exit_collim_z, d_hole, y_size, z_size, particle_size;blocks=nBlocks,threads=nThreads)
    return d_hole
end

#-----------------------------------------------------------------------------------------------------





#--------------------------------------- GateCommon_fun.cu -------------------------------------------

# Kernel kernl_brent_init from GateCommon_fun.cu
function f_kernel_brent_init(jlStackpart::JlStackParticle,nBlocks::CuDim,nThreads::CuDim)
    md = ctx()
    kernel_brent_init = CuFunction(md,"kernel_brent_init")
    
    jlCuStackpart = convert(JlCuStackParticle,jlStackpart)
    
    cudacall(kernel_brent_init,Tuple{CuStackParticle},jlCuStackpart;blocks=nBlocks,threads=nThreads)
end

# Kernel kernel_voxelized_source_b2b from GateCommon_fun.cu
function f_kernel_voxelized_source_b2b(g1::JlStackParticle, g2::JlStackParticle, act::JlActivities, E::Float32, size_in_vox::Int3, voxel_size::Float3, nBlocks::CuDim, nThreads::CuDim)
    md = ctx()
    kernel_voxelized_source_b2b = CuFunction(md,"kernel_voxelized_source_b2b")
    
    d_g1 = convert(JlCuStackParticle, g1)
    d_g2 = convert(JlCuStackParticle, g2)
    d_act = convert(JlCuActivities, act)
    
    cudacall(kernel_voxelized_source_b2b,Tuple{CuStackParticle, CuStackParticle, CuActivities, Float32, Int3, Float3}, d_g1, d_g2, d_act, E, size_in_vox, voxel_size;blocks=nBlocks,threads=nThreads)
end

# Kernel kernel_NavRegularPhan_Photon_NoSec from GateCommon_fun.cu
function f_kernel_NavRegularPhan_Photon_NoSec(photons::JlStackParticle, volume::JlVolume, materials::JlMaterials, count_d::Array{Int32}, nBlocks::CuDim, nThreads::CuDim)
    md = ctx()
    kernel_NavRegularPhan_Photon_NoSec = CuFunction(md,"kernel_NavRegularPhan_Photon_NoSec")
    
    d_photons = convert(JlCuStackParticle,photons)
    d_volume = convert(JlCuVolume,volume)
    d_materials = convert(JlCuMaterials,materials)
    d_count_d = CuArray(count_d)
    
    cudacall(kernel_NavRegularPhan_Photon_NoSec,Tuple{CuStackParticle,CuVolume,CuMaterials,CuPtr{Int32}}, d_photons, d_volume, d_materials, d_count_d;blocks=nBlocks,threads=nThreads)
end

# Kernel kernel_NavHexaColli_Photon_NoSec from GateCommon_fun.cu
function f_kernel_NavHexaColli_Photon_NoSec(photons::JlStackParticle, colli::Colli, centerOfHexagons::JlCoordHex2, materials::JlMaterials, count_d::Array{Int32}, nBlocks::CuDim, nThreads::CuDim)
    md = ctx()
    kernel_NavHexaColli_Photon_NoSec = CuFunction(md,"kernel_NavHexaColli_Photon_NoSec")

    d_photons = convert(JlCuStackParticle, photons)
    d_materials = convert(JlCuMaterials, materials)
    d_centerOfHexagons = convert(JlCuCoordHex2, centerOfHexagons)
    d_count_d = CuArray(count_d)
    
    cudacall(kernel_NavHexaColli_Photon_NoSec,Tuple{JlCuStackParticle,Colli,CuCoordHex2,CuMaterials,CuPtr{Int32}}, d_photons, colli, d_centerOfHexagons, d_materials, d_count_d;blocks=nBlocks,threads=nThreads)
end

# Kernel kernel_NavRegularPhan_Photon_WiSec from GateCommon_fun.cu
function f_kernel_NavRegularPhan_Photon_WiSec(photons::JlStackParticle, electrons::JlStackParticle, phantom::JlVolume, materials::JlMaterials, dosemap::JlDosimetry, count_d::Array{Int32}, step_limiter::Float32, nBlocks::CuDim, nThreads::CuDim)
    md = ctx()
    kernel_NavRegularPhan_Photon_WiSec = CuFunction(md,"kernel_NavRegularPhan_Photon_WiSec")

    d_photons = convert(JlCuStackParticle, photons)
    d_electrons = convert(JlCuStackParticle, electrons)
    d_phantom = convert(JlCuVolume, phantom)
    d_materials = convert(JlCuMaterials, materials)
    d_dosemap = convert(JlCuDosimetry, dosemap)
    d_count_d = CuArray(count_d)
    
    cudacall(kernel_NavRegularPhan_Photon_WiSec,Tuple{CuStackParticle,CuStackParticle,CuVolume,CuMaterials,CuDosimetry,CuPtr{Int32},Float32}, d_photons, d_electrons, d_phantom, d_materials, d_dosemap, d_count_d, step_limiter; blocks=nBlocks, threads=nThreads)
end

# Kernel kernel_NavRegularPhan_Electron_BdPhoton from GateCommon_fun.cu
function f_kernel_NavRegularPhan_Electron_BdPhoton(electrons::JlStackParticle, photons::JlStackParticle, phantom::JlVolume, materials::JlMaterials, dosemap::JlDosimetry, count_d::Array{Int32}, step_limiter::Float32, nBlocks::CuDim, nThreads::CuDim)
    md = ctx()
    kernel_NavRegularPhan_Electron_BdPhoton = CuFunction(md,"kernel_NavRegularPhan_Electron_BdPhoton")
    
    d_electrons = convert(JlCuStackParticle, electrons)
    d_photons = convert(JlCuStackParticle, photons)
    d_phantom = convert(JlCuVolume, phantom)
    d_materials = convert(JlCuMaterials, materials)
    d_dosemap = convert(JlCuDosimetry, dosemap)
    d_count_d = CuArray(count_d)
    
    cudacall(kernel_NavRegularPhan_Electron_BdPhoton,Tuple{CuStackParticle,CuStackParticle,CuVolume,CuMaterials,CuDosimetry,CuPtr{Int32},Float32}, d_electrons, d_photons, d_phantom, d_materials, d_dosemap, d_count_d, step_limiter; blocks=nBlocks, threads=nThreads)
end

#-----------------------------------------------------------------------------------------------------



#--------------------------------------- GateOptical_fun.cu ------------------------------------------

# Kernel kernel_optical_voxelized_source from GateOptical_fun.cu
function f_kernel_optical_voxelized_source(photons::JlStackParticle, phantom_mat::JlVolume, phantom_act::Array{Float32}, phantom_ind::Array{UInt32}, E::Float32, nBlocks::CuDim, nThreads::CuDim)
    md = ctx()
    kernel_optical_voxelized_source = CuFunction(md,"kernel_optical_voxelized_source")

    d_photons = convert(JlCuStackParticle,photons)
    d_phantom_mat = convert(JlCuVolume,phantom_mat)
    d_phantom_act = CuArray(phantom_act)
    d_phantom_ind = CuArray(phantom_ind)

    cudacall(kernel_optical_voxelized_source,Tuple{CuStackParticle,CuVolume,CuPtr{Float32},CuPtr{UInt32},Float32}, d_photons, d_phantom_mat, d_phantom_act, d_phantom_ind, E; blocks=nBlocks, threads=nThreads)

    return d_photons

end

# Kernel kernel_optical_navigation_regular from GateOptical_fun.cu
function f_kernel_optical_navigation_regular(photons::JlStackParticle, phantom::JlVolume, count_d::Array{Int32}, nBlocks::CuDim, nThreads::CuDim)
    md = ctx()
    kernel_optical_navigation_regular = CuFunction(md,"kernel_optical_navigation_regular")

    d_photons = convert(JlCuStackParticle,photons)
    d_phantom = convert(JlCuVolume,phantom)
    d_count_d = CuArray(count_d)

    cudacall(kernel_optical_navigation_regular,Tuple{CuStackParticle,CuVolume,CuPtr{Int32}}, d_photons, d_phantom, d_count_d; blocks=nBlocks, threads=nThreads)
end

#-----------------------------------------------------------------------------------------------------
