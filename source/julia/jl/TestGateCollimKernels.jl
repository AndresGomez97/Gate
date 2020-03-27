include("GateCollimKernels.jl")

# DIMENSION FOR MATRIX 
dims = (3,4)

# TESTING KERNEL MAP ENTRY
function test_kernel_map_entry()

	# Define input params
	px = round.(rand(Float32, dims) * 100)
	py = round.(rand(Float32, dims) * 100)
    pz = round.(rand(Float32, dims) * 100)
    
	entry_collim_y = round.(rand(Float32, dims) * 100)
    entry_collim_z = round.(rand(Float32, dims) * 100)
    
	hole = round.(rand(Int32, dims))

	size_y = UInt32(12)
	size_z = UInt32(12)

	particle_size = Int32(40)

    nBlocks = 1
    nThreads = prod(dims)

	# Debug info and calling kernel
	println("")
    println("----------------------------------------------------------------------")
	println("Hole before calling kernel: \n",hole)
	println("----------------------------------------------------------------------")

	res = GateCollimKernels.f_kernel_map_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

    println("")
	println("----------------------------------------------------------------------")
	println("Hole after calling kernel: \n",res)
    println("----------------------------------------------------------------------")
	println("")
end

# TESTING KERNEL MAP PROJECTION
function test_kernel_map_projection()

    # Define input params
	px = round.(rand(Float32, dims) * 100)
	py = round.(rand(Float32, dims) * 100)
    pz = round.(rand(Float32, dims) * 100)

    dx = round.(rand(Float32, dims) * 100)
	dy = round.(rand(Float32, dims) * 100)
    dz = round.(rand(Float32, dims) * 100)

    hole = round.(rand(Int32, dims))

    planeToProject = Float32(3.0)

    particle_size = UInt32(40)

    nBlocks = 1
    nThreads = prod(dims)

    # Debug info and calling kernel
	println("")
    println("----------------------------------------------------------------------")
	println("PX before calling kernel: \n",px)
    println("----------------------------------------------------------------------")
    println("")
    println("----------------------------------------------------------------------")
	println("PY before calling kernel: \n",py)
    println("----------------------------------------------------------------------")
    println("")
    println("----------------------------------------------------------------------")
	println("PZ before calling kernel: \n",pz)
	println("----------------------------------------------------------------------")

	res_x,res_y,res_z = GateCollimKernels.f_kernel_map_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)

	println("")
    println("----------------------------------------------------------------------")
	println("PX after calling kernel: \n",res_x)
    println("----------------------------------------------------------------------")
    println("")
    println("----------------------------------------------------------------------")
	println("PY after calling kernel: \n",res_y)
    println("----------------------------------------------------------------------")
    println("")
    println("----------------------------------------------------------------------")
	println("PZ after calling kernel: \n",res_z)
	println("----------------------------------------------------------------------")
	println("")

end

# TESTING KERNEL MAP EXIT
function test_kernel_map_exit()

    # Define input params
	px = round.(rand(Float32, dims) * 100)
	py = round.(rand(Float32, dims) * 100)
    pz = round.(rand(Float32, dims) * 100)
    
	entry_collim_y = round.(rand(Float32, dims) * 100)
    entry_collim_z = round.(rand(Float32, dims) * 100)
    
	hole = round.(rand(Int32, dims))

	size_y = UInt32(12)
	size_z = UInt32(12)

	particle_size = Int32(40)

    nBlocks = 1
    nThreads = prod(dims)

	# Debug info and calling kernel
	println("")
    println("----------------------------------------------------------------------")
	println("Hole before calling kernel: \n",hole)
	println("----------------------------------------------------------------------")

	res = GateCollimKernels.f_kernel_map_exit(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

    println("")
	println("----------------------------------------------------------------------")
	println("Hole after calling kernel: \n",res)
    println("----------------------------------------------------------------------")
	println("")

end