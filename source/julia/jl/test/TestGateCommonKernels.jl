include("GateKernels.jl")

# Define dimensions
dims = (3,4)

# TESTING KERNEL BRENT INIT
function test_kernel_brent_init()

    # Define input param stackpart
    E = round.(rand(Float32, dims) * 100)
    px = round.(rand(Float32, dims) * 100)
	py = round.(rand(Float32, dims) * 100)
    pz = round.(rand(Float32, dims) * 100)
    dx = round.(rand(Float32, dims) * 100)
	dy = round.(rand(Float32, dims) * 100)
    dz = round.(rand(Float32, dims) * 100)
    t = round.(rand(Float32, dims) * 100)
    type = round.(rand(UInt16, dims))
    eventID = round.(rand(UInt32, dims))
    trackID = round.(rand(UInt32, dims))
    seed = round.(rand(UInt32, dims))
    active = rand(Char, dims)
    endsimu = rand(Char, dims)
    table_x_brent = round.(rand(Int64, dims) * 100)
    size = UInt32(prod(dims))

    # Define the rest of input params
    nBlocks = 1
    nThreads = prod(dims)

    jlStackpart = JlStackParticle(E,px,py,pz,dx,dy,dz,t,type,eventID,trackID,seed,active,endsimu,table_x_brent,size) 
    
    # Debug info and calling method
    f_kernel_brent_init(jlStackpart,nBlocks,nThreads)
end

# TESTING KERNEL VOXELIZED SOURCE B2B
function test_kernel_voxelized_source_b2b()

    # Define input param StackParticle g1
    E = round.(rand(Float32, dims) * 100)
    px = round.(rand(Float32, dims) * 100)
    py = round.(rand(Float32, dims) * 100)
    pz = round.(rand(Float32, dims) * 100)
    dx = round.(rand(Float32, dims) * 100)
    dy = round.(rand(Float32, dims) * 100)
    dz = round.(rand(Float32, dims) * 100)
    t = round.(rand(Float32, dims) * 100)
    type = round.(rand(UInt16, dims))
    eventID = round.(rand(UInt32, dims))
    trackID = round.(rand(UInt32, dims))
    seed = round.(rand(UInt32, dims))
    active = rand(Char, dims)
    endsimu = rand(Char, dims)
    table_x_brent = round.(rand(Int64, dims) * 100)
    size = UInt32(prod(dims))

    g1 = JlStackParticle(E,px,py,pz,dx,dy,dz,t,type,eventID,trackID,seed,active,endsimu,table_x_brent,size) 
    
    # Define input param StackParticle g1
    E = round.(rand(Float32, dims) * 100)
    px = round.(rand(Float32, dims) * 100)
    py = round.(rand(Float32, dims) * 100)
    pz = round.(rand(Float32, dims) * 100)
    dx = round.(rand(Float32, dims) * 100)
    dy = round.(rand(Float32, dims) * 100)
    dz = round.(rand(Float32, dims) * 100)
    t = round.(rand(Float32, dims) * 100)
    type = round.(rand(UInt16, dims))
    eventID = round.(rand(UInt32, dims))
    trackID = round.(rand(UInt32, dims))
    seed = round.(rand(UInt32, dims))
    active = rand(Char, dims)
    endsimu = rand(Char, dims)
    table_x_brent = round.(rand(Int64, dims) * 100)
    
    g2 = JlStackParticle(E,px,py,pz,dx,dy,dz,t,type,eventID,trackID,seed,active,endsimu,table_x_brent,size) 

    # Define input param Activities act
    nb_activities = UInt32(1)
    tot_activity = Float32(1)
    act_index = [UInt32(0)] 
    act_cdf = round.(rand(Float32, 1) * 100)

    act = JlActivities(nb_activities,tot_activity,act_index,act_cdf)
    
    # Define rest of input params
    E_param = Float32(2.42)

    size_in_vox = Int3(Int32(20),Int32(20),Int32(20))
    voxel_size = Float3(Float32(2.5),Float32(2.5),Float32(2.5))

    nBlocks = 1
    nThreads = prod(dims)

    # Debug info and calling method
    f_kernel_voxelized_source_b2b(g1, g2, act, E_param, size_in_vox, voxel_size, nBlocks, nThreads)
end

# TEST KERNEL NAV REGULAR PHAN PHOTON NOSEC
function test_kernel_NavRegularPhan_Photon_NoSec()

    # Define input param photons

    # Define input param volume

    # Define input param materials

    # Define the rest of input params

    # Debug info and calling method
    f_kernel_NavRegularPhan_Photon_NoSec()
end

# TEST KERNEL NAV HEXA COLLI PHOTON NOSEC
function test_kernel_NavHexaColli_Photon_NoSec()
    
    # Define input param photons

    # Define input param colli 

    # Define input param centerOfHexagons

    # Define input param materials

    # Define the rest of input params
    
    # Debug info and calling method
    f_kernel_NavHexaColli_Photon_NoSec()
end

# TEST KERNEL NAV REGULAR PHAN PHOTON WISEC
function test_kernel_NavRegularPhan_Photon_WiSec()

    # Define input param photons

    # Define input param electrons

    # Define input param phantom

    # Define input param materials

    # Define input param dosemap

    # Define the rest of input params
    
    # Debug info and calling method
    f_kernel_NavRegularPhan_Photon_WiSec()
end

# TEST KERNEL NAV REGULAR PHAN ELECTRON BDPHOTON
function test_kernel_NavRegularPhan_Electron_BdPhoton()

    # Define input param electrons

    # Define input param photons

    # Define input param phantom

    # Define input param materials

    # Define input param dosemap

    # Define the rest of input params 

    # Debug info and calling method
    f_kernel_NavRegularPhan_Electron_BdPhoton()
end

#---------------------------------------- Calls ----------------------------------------------

#test_kernel_brent_init()
#test_kernel_voxelized_source_b2b()
#test_kernel_NavRegularPhan_Photon_NoSec()
#test_kernel_NavHexaColli_Photon_NoSec()
#test_kernel_NavRegularPhan_Photon_WiSec()
#test_kernel_NavRegularPhan_Electron_BdPhoton()

#---------------------------------------------------------------------------------------------