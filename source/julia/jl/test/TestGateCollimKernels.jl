include("../src/CUDACallKernels.jl")
include("../src/JuliaKernels.jl")

using Test

# Dims
dims = (3,3)

# TESTING DEVICE METHOD binary_search
function test_binary_search()

	println("####################### BINARY SEARCH #############################")
	println("")
	
	entry_collim_y = Array{Float32}([0.4,0.3,0.2,0.14])
    entry_collim_z = Array{Float32}([0.4,0.3,0.2,0.14])

	py = Array{Float32}([0.2,0.3,0.4])
	pz = Array{Float32}([0.2,0.3,0.4])
	
	size_y = UInt32(4)
	size_z = UInt32(4)

	res_y = [1,0,0]
	res_z = [1,0,0]

	println("PY: \n",py)
	println("")
	println("PZ: \n",pz)
	println("")
	println("Entry Collim Y: \n",entry_collim_y)
	println("")
	println("Entry Collim Z: \n",entry_collim_z)
	println("")
	println("Size Y: \n",size_y)
	println("")
	println("Size Z: \n",size_z)
	println("")
	println("----------------------------------------------------------------------")
	println("Espected res_y: \n",res_y)
	println("----------------------------------------------------------------------")
	println("")
	println("----------------------------------------------------------------------")
	println("Espected res_z: \n",res_z)
	println("----------------------------------------------------------------------")
	println("")
	i = 1
	while i < 4
		@test res_y[i] == JuliaKernels.binary_search(py[i],entry_collim_y,size_y)
		@test res_z[i] == JuliaKernels.binary_search(pz[i],entry_collim_z,size_z)
		println("Iteration $i passed")
		println("")
		i+=1
	end
	println("##################################################################")
end


# TESTING kernel_map_entry
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

	particle_size = Int32(9)

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
	println("Entry Collim Y: \n",entry_collim_y)
	println("")
	println("Entry Collim Z: \n",entry_collim_z)
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
	#CUDACallKernels.jl
	res = GateKernels.f_kernel_map_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)
	
	#GateCollim_gpu.jl
	res2 = JuliaKernels.call_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)
	
	@test res == [-1 -1 -1; -1 -1 -1; -1 -1 -1]
	@test res2 == res

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
	#CUDACallKernels.jl
	res = GateKernels.f_kernel_map_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)
	
	#GateCollim_gpu.jl
	res2 = JuliaKernels.call_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

	@test res == [-1 0 -1; -1 -1 -1; 0 -1 -1]
	@test res2 == res

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

	#CUDACallKernels.jl
	res = GateKernels.f_kernel_map_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)
	
	#GateCollim_gpu.jl
	res2 = JuliaKernels.call_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)

	@test res == [0 0 0; 0 0 0; 0 0 0]
	@test res2 == res

	println("Test passed")
	println("")
	println("----------------------------------------------------------------------")
	println("Hole after calling kernel: \n",res)
	println("----------------------------------------------------------------------")
	println("######################################################################")
	println("")
end

# TESTING kernel_map_projection
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

    particle_size = Int32(40)

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

	#CUDACallKernels.jl
	res_x,res_y,res_z = GateKernels.f_kernel_map_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)
	
	#GateCollim_gpu.jl
	res2_x,res2_y,res2_z = JuliaKernels.call_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)
	
	@test res_x == px
	@test res_y == py
	@test res_z == pz

	@test res2_x == res_x
	@test res2_y == res_y
	@test res2_z == res_z

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
	
	#CUDACallKernels.jl
	res_x,res_y,res_z = GateKernels.f_kernel_map_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)
	
	#GateCollim_gpu.jl
	res2_x,res2_y,res2_z = JuliaKernels.call_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)
	
	@test res_x == [3.0 3.0 2.0]
	@test res_y == [4.5 -1.5 4.0]
	@test res_z == [6.0 21.0 8.0]

	@test res2_x == res_x
	@test res2_y == res_y
	@test res2_z == res_z
	
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
	
	#CUDACallKernels.jl
	res_x,res_y,res_z = GateKernels.f_kernel_map_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)

	#GateCollim_gpu.jl
	res2_x,res2_y,res2_z = JuliaKernels.call_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)
	
	@test res_x == [3.0 2.0 3.0]
	@test res_y == [51.0 4.0 -2.0]
	@test res_z == [5.0 8.0 -1.0]

	@test res2_x == res_x
	@test res2_y == res_y
	@test res2_z == res_z
	
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

# TESTING kernel_map_exit
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

function test_call_all()
	
	particle_size = Int32(22540)

	size_y = UInt32(262)
	size_z = size_y

	planeToProject = Float32(54.5)
	hole = zeros(Int32,particle_size)

	nBlocks = 45
	nThreads = 512

	px = Array{Float32}(undef,particle_size)
	py = Array{Float32}(undef,particle_size)
	pz = Array{Float32}(undef,particle_size)

	dx = Array{Float32}(undef,particle_size)
	dy = Array{Float32}(undef,particle_size)
	dz = Array{Float32}(undef,particle_size)

	entry_collim_y = Array{Float32}(undef,size_y)
	entry_collim_z = Array{Float32}(undef,size_z)

	exit_collim_y = Array{Float32}(undef,size_y)
	exit_collim_z = Array{Float32}(undef,size_z)

	io = open("data/px.txt","r")
    for i = 1:particle_size 
		px[i] = read(io,Float32)
    end
	close(io)
	
    io = open("data/py.txt","r")
    for i = 1:particle_size
		py[i] = read(io,Float32)
    end
    close(io)
    
    io = open("data/pz.txt","r")  
    for i = 1:particle_size
		pz[i] = read(io,Float32)
    end
    close(io)
    
    io = open("data/dx.txt","r")  
    for i = 1:particle_size
		dx[i] = read(io,Float32)
    end
    close(io)

    io = open("data/dy.txt","r")  
    for i = 1:particle_size
		dy[i] = read(io,Float32)
    end
    close(io)

    io = open("data/dz.txt","r")  
    for i = 1:particle_size
		dz[i] = read(io,Float32)
    end
    close(io)

    io = open("data/entry_collim_y.txt","r")  
    for i = 1:size_y
		entry_collim_y[i] = read(io,Float32)
    end
    close(io)

    io = open("data/entry_collim_z.txt","r")  
    for i = 1:size_z
		entry_collim_z[i] = read(io,Float32)
    end
    close(io)

    io = open("data/exit_collim_y.txt","r")  
    for i = 1:size_y
		exit_collim_y[i] = read(io,Float32)
    end
    close(io)

    io = open("data/exit_collim_z.txt","r")  
    for i = 1:size_z
        exit_collim_z[i] = read(io,Float32)
    end
	close(io)

	hole = JuliaKernels.call_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)
	cchole = GateKernels.f_kernel_map_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)
	println("kernel_map_entry")
	println(hole == cchole)
	px,py,pz = JuliaKernels.call_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)
	ccpx,ccpy,ccpz = GateKernels.f_kernel_map_projection(px,py,pz,dx,dy,dz,cchole,planeToProject,particle_size,nBlocks,nThreads)
	println("kernel_map_projection")
	println("px")
	println(px==ccpx)
	println("py")
	println(py==ccpy)
	println("pz")
	println(pz==ccpz)
	hole = JuliaKernels.call_exit(px,py,pz,exit_collim_y,exit_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)
	cchole = GateKernels.f_kernel_map_exit(ccpx,ccpy,ccpz,exit_collim_y,exit_collim_z,cchole,size_y,size_z,particle_size,nBlocks,nThreads)
	println("kernel_map_exit")
	println(hole==cchole)
end

#---------------------------------------- Calls ----------------------------------------------
#test_binary_search()
#test_kernel_map_entry()
#test_kernel_map_projection()
#test_kernel_map_exit()
test_call_all()
#---------------------------------------------------------------------------------------------
