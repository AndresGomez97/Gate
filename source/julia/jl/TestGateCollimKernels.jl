include("GateKernels.jl")

using Test

#using GateKernels: f_kernel_map_entry,f_kernel_map_exit,f_kernel_map_projection

# Dims
dims = (3,3)

# TESTING KERNEL MAP ENTRY
function test_kernel_map_entry()

	println("######################## MAP ENTRY #################################")
	println("")

	# Define input params
	px = Array{Float32}([[0.1,0.2,0.3] [0.4,0.5,0.6] [0.7,0.8,1.0]])
	py = Array{Float32}([[0.1,0.2,0.3] [0.4,0.5,0.6] [0.7,0.8,1.0]])
	pz = Array{Float32}([[0.1,0.2,0.3] [0.4,0.5,0.6] [0.7,0.8,1.0]])

	entry_collim_y = Array{Float32}([0.1,0.2,0.3,0.4])

	entry_collim_z = Array{Float32}([0.1,0.2,0.3,0.4])
	
	hole = zeros(Int32, (3,3))

	size_y = UInt32(4)
	size_z = UInt32(4)

	particle_size = Int32(10)

	nBlocks = 1
	nThreads = 9

	# Debug info and calling kernel
	println("########################### First test ##############################")
	println("PX: \n",px)
	println("")
	println("PY: \n",py)
	println("")
	println("PZ: \n",pz)
	println("")
	println("Exit Collim Y: \n",entry_collim_y)
	println("")
	println("Exit Collim Z: \n",entry_collim_z)
	println("")
	println("Size Y: \n",size_y)
	println("")
	println("Size Z: \n",size_z)
	println("")
	println("Particle Size: \n",particle_size)
	println("")
	println("----------------------------------------------------------------------")
	println("Hole before calling kernel: \n",hole)
	println("----------------------------------------------------------------------")
	println("")
	res = GateKernels.f_kernel_map_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

	@test res == [-1 -1 -1; -1 -1 -1; -1 -1 -1]

	println("Test passed")
	println("")
	println("----------------------------------------------------------------------")
	println("Hole after calling kernel: \n",res)
	println("----------------------------------------------------------------------")
	println("######################################################################")
	println("")
	
	# Define input params
	entry_collim_y = Array{Float32}([0.4,0.3,0.2,0.14])

    entry_collim_z = Array{Float32}([0.4,0.3,0.2,0.14])
    
	hole = zeros(Int32, dims)

	# Debug info and calling kernel
	println("########################### Second test ##############################")
	println("PX: \n",px)
	println("")
	println("PY: \n",py)
	println("")
	println("PZ: \n",pz)
	println("")
	println("Exit Collim Y: \n",entry_collim_y)
	println("")
	println("Exit Collim Z: \n",entry_collim_z)
	println("")
	println("Size Y: \n",size_y)
	println("")
	println("Size Z: \n",size_z)
	println("")
	println("Particle Size: \n",particle_size)
	println("")
    println("----------------------------------------------------------------------")
	println("Hole before calling kernel: \n",hole)
	println("----------------------------------------------------------------------")
	println("")

	res = GateKernels.f_kernel_map_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

	@test res == [-1 0 -1; -1 -1 -1; 0 -1 -1]

	println("Test passed")
    println("")
	println("----------------------------------------------------------------------")
	println("Hole after calling kernel: \n",res)
    println("----------------------------------------------------------------------")
	println("######################################################################")
	println("")
	# Define input params
	hole = zeros(Int32, dims)

	particle_size = Int32(0)

	# Debug info and calling kernel
	println("########################### Third test ###############################")
	println("PX: \n",px)
	println("")
	println("PY: \n",py)
	println("")
	println("PZ: \n",pz)
	println("")
	println("Exit Collim Y: \n",entry_collim_y)
	println("")
	println("Exit Collim Z: \n",entry_collim_z)
	println("")
	println("Size Y: \n",size_y)
	println("")
	println("Size Z: \n",size_z)
	println("")
	println("Particle Size: \n",particle_size)
	println("")
	println("----------------------------------------------------------------------")
	println("Hole before calling kernel: \n",hole)
	println("----------------------------------------------------------------------")
	println("")

	res = GateKernels.f_kernel_map_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

	@test res == [0 0 0; 0 0 0; 0 0 0]

	println("Test passed")
	println("")
	println("----------------------------------------------------------------------")
	println("Hole after calling kernel: \n",res)
	println("----------------------------------------------------------------------")
	println("######################################################################")
	println("")
end

# TESTING KERNEL MAP PROJECTION
function test_kernel_map_projection()

	println("##################### MAP PROJECTION #################################")
	println("")
    # Define input params
	px = round.(rand(Float32, dims) * 100)
	py = round.(rand(Float32, dims) * 100)
    pz = round.(rand(Float32, dims) * 100)

    dx = round.(rand(Float32, dims) * 100)
	dy = round.(rand(Float32, dims) * 100)
    dz = round.(rand(Float32, dims) * 100)

    hole = Array{Int32}([-1 -1 -1; -1 -1 -1; -1 -1 -1])

    planeToProject = Float32(3.0)

    particle_size = UInt32(40)

    nBlocks = 1
    nThreads = prod(dims)

	# Debug info and calling kernel
	println("########################### First test ##############################")
	println("Hole: \n",hole)
	println("")
	println("PlaneToProject: \n",planeToProject)
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
	println("")
	println("----------------------------------------------------------------------")
	println("DX before calling kernel: \n",dx)
	println("----------------------------------------------------------------------")
	println("")
	println("----------------------------------------------------------------------")
	println("DY before calling kernel: \n",dy)
	println("----------------------------------------------------------------------")
	println("")
	println("----------------------------------------------------------------------")
	println("DZ before calling kernel: \n",dz)
	println("----------------------------------------------------------------------")
	println("")

	res_x,res_y,res_z = GateKernels.f_kernel_map_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)

	@test res_x == px
	@test res_y == py
	@test res_z == pz

	println("Test passed")

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
	println("######################################################################")
	println("")

	# Define input params
	px = Array{Float32}([1 4 2])
	py = Array{Float32}([2 1 4])
	pz = Array{Float32}([3 21 8])

	dx = Array{Float32}([4 2 3])
	dy = Array{Float32}([5 5 2])
	dz = Array{Float32}([6 0 1])

	hole = Array{Int32}([0 0 -1])

	nThreads=3

	# Debug info and calling kernel
	println("############################ Second test #############################")
	println("Hole: \n",hole)
	println("")
	println("PlaneToProject: \n",planeToProject)
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
	println("")
	println("----------------------------------------------------------------------")
	println("DX before calling kernel: \n",dx)
	println("----------------------------------------------------------------------")
	println("")
	println("----------------------------------------------------------------------")
	println("DY before calling kernel: \n",dy)
	println("----------------------------------------------------------------------")
	println("")
	println("----------------------------------------------------------------------")
	println("DZ before calling kernel: \n",dz)
	println("----------------------------------------------------------------------")
	println("")
	
	res_x,res_y,res_z = GateKernels.f_kernel_map_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)

	@test res_x == [3.0 3.0 2.0]
	@test res_y == [4.5 -1.5 4.0]
	@test res_z == [6.0 21.0 8.0]
	
	println("Test passed")

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
	println("######################################################################")
	println("")

	# Define input params
	px = Array{Float32}([14 2 4])
	py = Array{Float32}([7 4 2])
	pz = Array{Float32}([5 8 0])

	dx = Array{Float32}([-1 3 5])
	dy = Array{Float32}([4 2 20])
	dz = Array{Float32}([0 1 5])

	hole = Array{Int32}([0 -1 0])

	# Debug info and calling kernel
	println("############################ Third test #############################")
	println("Hole: \n",hole)
	println("")
	println("PlaneToProject: \n",planeToProject)
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
	println("")
	println("----------------------------------------------------------------------")
	println("DX before calling kernel: \n",dx)
	println("----------------------------------------------------------------------")
	println("")
	println("----------------------------------------------------------------------")
	println("DY before calling kernel: \n",dy)
	println("----------------------------------------------------------------------")
	println("")
	println("----------------------------------------------------------------------")
	println("DZ before calling kernel: \n",dz)
	println("----------------------------------------------------------------------")
	println("")
	
	res_x,res_y,res_z = GateKernels.f_kernel_map_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)

	@test res_x == [3.0 2.0 3.0]
	@test res_y == [51.0 4.0 -2.0]
	@test res_z == [5.0 8.0 -1.0]
	
	println("Test passed")

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
	println("######################################################################")
	println("")
end

# TESTING KERNEL MAP EXIT
function test_kernel_map_exit()

	println("########################## MAP EXIT #################################")
	println("")

	# Define input params
	px = Array{Float32}([[0.1,0.2,0.3] [0.4,0.5,0.6] [0.7,0.8,1.0]])
	py = Array{Float32}([[0.1,0.2,0.3] [0.4,0.5,0.6] [0.7,0.8,1.0]])
	pz = Array{Float32}([[0.1,0.2,0.3] [0.4,0.5,0.6] [0.7,0.8,1.0]])

	exit_collim_y = Array{Float32}([0.1,0.2,0.3,0.4])

	exit_collim_z = Array{Float32}([0.1,0.2,0.3,0.4])
	
	hole = zeros(Int32, dims)

	size_y = UInt32(4)
	size_z = UInt32(4)

	particle_size = Int32(8)

	nBlocks = 1
	nThreads = 9

	# Debug info and calling kernel
	println("")
	println("########################### First test ###############################")
	println("PX: \n",px)
	println("")
	println("PY: \n",py)
	println("")
	println("PZ: \n",pz)
	println("")
	println("Exit Collim Y: \n",exit_collim_y)
	println("")
	println("Exit Collim Z: \n",exit_collim_z)
	println("")
	println("Size Y: \n",size_y)
	println("")
	println("Size Z: \n",size_z)
	println("")
	println("Particle Size: \n",particle_size)
	println("")
    println("----------------------------------------------------------------------")
	println("Hole before calling kernel: \n",hole)
	println("----------------------------------------------------------------------")
	println("")

	res = GateKernels.f_kernel_map_exit(px,py,pz,exit_collim_y,exit_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

	@test res == [-1 -1 -1; -1 -1 -1; -1 -1 0]

	println("Test passed")
    println("")
	println("----------------------------------------------------------------------")
	println("Hole after calling kernel: \n",res)
	println("----------------------------------------------------------------------")
	println("######################################################################")
	println("")

	# Define input params
	particle_size = Int32(9)

	exit_collim_y = Array{Float32}([0.4,0.3,0.2,0.14])

	exit_collim_z = Array{Float32}([0.4,0.3,0.2,0.14])
	
	hole = zeros(Int32, dims)

	# Debug info and calling kernel
	println("########################### Second test ##############################")
	println("PX: \n",px)
	println("")
	println("PY: \n",py)
	println("")
	println("PZ: \n",pz)
	println("")
	println("Exit Collim Y: \n",exit_collim_y)
	println("")
	println("Exit Collim Z: \n",exit_collim_z)
	println("")
	println("Size Y: \n",size_y)
	println("")
	println("Size Z: \n",size_z)
	println("")
	println("Particle Size: \n",particle_size)
	println("")
	println("----------------------------------------------------------------------")
	println("Hole before calling kernel: \n",hole)
	println("----------------------------------------------------------------------")
	println("")

	res = GateKernels.f_kernel_map_exit(px,py,pz,exit_collim_y,exit_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

	@test res == [-1 0 -1; -1 -1 -1; 0 -1 -1]

	println("Test passed")
	println("")
	println("----------------------------------------------------------------------")
	println("Hole after calling kernel: \n",res)
	println("----------------------------------------------------------------------")
	println("######################################################################")
	println("")

	# Define input params
	hole = zeros(Int32, dims)

	particle_size = Int32(0)

	# Debug info and calling kernel
	println("########################### Third test ###############################")
	println("PX: \n",px)
	println("")
	println("PY: \n",py)
	println("")
	println("PZ: \n",pz)
	println("")
	println("Exit Collim Y: \n",exit_collim_y)
	println("")
	println("Exit Collim Z: \n",exit_collim_z)
	println("")
	println("Size Y: \n",size_y)
	println("")
	println("Size Z: \n",size_z)
	println("")
	println("Particle Size: \n",particle_size)
	println("")
	println("----------------------------------------------------------------------")
	println("Hole before calling kernel: \n",hole)
	println("----------------------------------------------------------------------")
	println("")
	
	res = GateKernels.f_kernel_map_exit(px,py,pz,exit_collim_y,exit_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

	@test res == [0 0 0; 0 0 0; 0 0 0]

	println("Test passed")
	println("")
	println("----------------------------------------------------------------------")
	println("Hole after calling kernel: \n",res)
	println("----------------------------------------------------------------------")
	println("######################################################################")
	println("")
end

#---------------------------------------- Calls ----------------------------------------------

test_kernel_map_entry()
test_kernel_map_projection()
test_kernel_map_exit()

#---------------------------------------------------------------------------------------------
