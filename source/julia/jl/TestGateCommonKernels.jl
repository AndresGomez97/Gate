include("GateCommonKernels.jl")

# Define dimensions
dims = (3,4)

# TESTING KERNEL BRENT INIT
function test_kernel_brent_init()

    # Define input params
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
    
    # Debug info and calling method
    GateCommonKernels.f_kernel_brent_init(E,px,py,pz,dx,dy,dz,t,type,eventID,trackID,seed,active,endsimu,table_x_brent,size,nBlocks,nThreads)

end

