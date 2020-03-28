include("GateKernels.jl")

# Define dimensions
dims = (3,4)

# TESTING KERNEL BRENT INIT
function test_kernel_brent_init()

    # Define input params StackParticle
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

    nBlocks = 1
    nThreads = prod(dims)

    jlStackpart = JlStackParticle(E,px,py,pz,dx,dy,dz,t,type,eventID,trackID,seed,active,endsimu,table_x_brent,size) 
    
    # Debug info and calling method
    f_kernel_brent_init(jlStackpart,nBlocks,nThreads)

end

function test_kernel_voxelized_source_b2b()

    # Define input params StackParticle
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

    nb_activities = UInt32(1)
    tot_activity = Float32(1)
    act_index = [UInt32(0)] 
    act_cdf = round.(rand(Float32, 1) * 100)

    act = JlActivities(nb_activities,tot_activity,act_index,act_cdf)

    E_param = Float32(2.42)

    size_in_vox = Int3(Int32(20),Int32(20),Int32(20))
    voxel_size = Float3(Float32(2.5),Float32(2.5),Float32(2.5))

    nBlocks = 1
    nThreads = prod(dims)

    # Debug info and calling method
    f_kernel_voxelized_source_b2b(g1, g2, act, E_param, size_in_vox, voxel_size, nBlocks, nThreads)
    
end

# Calls
test_kernel_voxelized_source_b2b()
