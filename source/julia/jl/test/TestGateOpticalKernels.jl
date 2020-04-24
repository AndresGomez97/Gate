include("GateKernels.jl")

# Define dimensions
#dims = (3,3)

# TESTING KERNEL OPTICAL VOXELIZED SOURCE
function test_kernel_optical_voxelized_source()

    # Define input param photons
    E = Array{Float32}([])
    
    dx = Array{Float32}([])
    dy = Array{Float32}([])
    dz = Array{Float32}([])

    px = Array{Float32}([])
    py = Array{Float32}([])
    pz = Array{Float32}([])
    
    t = Array{Float32}([])
    
    type = Array{UInt16}([])

    eventID = Array{UInt32}([])
    trackID = Array{UInt32}([])
    
    seed = Array{UInt32}([])

    active = Array{Char}([])
    endsimu = Array{Char}([])

    table_x_brent = Array{UInt64}([])

    size = UInt32(size(E,1))

    photons = JlStackParticle(E,dx,dy,dz,px,py,pz,t,type,evemtID,trackID,seed,active,endsimu,table_x_brent,size)

    # phantom_mat
    data = Array{UInt16}([])
    mem_data = UInt32()
    
    x = Float32()
    y = Float32()
    z = Float32()
    size_in_mm = Float3(x,y,z)

    x = Int32()
    y = Int32()
    z = Int32()
    size_in_vox = Int3(x,y,z)
    
    x = Float32()
    y = Float32()
    z = Float32()
    voxel_size = Float3(x,y,z)

    nb_voxel_volume = Int32()
    nb_voxel_slice = Int32()

    x = Float32()
    y = Float32()
    z = Float32()
    phantom_position = Float3(x,y,z)

    phantom_mat = JlVolume(data,mem_data,size_in_mm,size_in_vox,voxel_size,nb_voxel_volume,nb_voxel_slice,phantom_position)
    
    # phantom_act
    phantom_act = Array{Float32}([])

    # phantom_ind
    phantom_ind = Array{Uint32}([])

    # E
    E = 1.3

    # Debug info and calling method
    res_photons = f_kernel_optical_voxelized_source(photons,phantom_mat,phantom_act,phantom_ind,E)

end

# TESTING KERNEL OPTICAL NAVIGATION REGULAR
function test_kernel_optical_navigation_regular()
    
    # Define input param photons

    # Define input param phantom

    # Define the rest of input params
    
    # Debug info and calling method
    f_kernel_optical_navigation_regular()
end

#---------------------------------------- Calls ----------------------------------------------

#test_kernel_optical_voxelized_source()
#test_kernel_optical_navigation_regular()

#---------------------------------------------------------------------------------------------