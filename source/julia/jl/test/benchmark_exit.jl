include("../src/JuliaKernels.jl")

using Test
using BenchmarkTools
using CuArrays

particle_size = Int32(22540)

size_y = UInt32(262)
size_z = size_y

planeToProject = Float32(54.5)
hole = Array{Int32}(undef,particle_size)

nBlocks = 45
nThreads = 512

px = Array{Float32}(undef,particle_size)
py = Array{Float32}(undef,particle_size)
pz = Array{Float32}(undef,particle_size)

exit_collim_y = Array{Float32}(undef,size_y)
exit_collim_z = Array{Float32}(undef,size_z)

io = open("data/new_hole.txt","r")
for i = 1:particle_size
    hole[i] = read(io,Int32)
end
close(io)

io = open("data/new_px.txt","r")
for i = 1:particle_size 
	px[i] = read(io,Float32)
end
close(io)

io = open("data/new_py.txt","r")
for i = 1:particle_size
	py[i] = read(io,Float32)
end
close(io)

io = open("data/new_pz.txt","r")  
for i = 1:particle_size
	pz[i] = read(io,Float32)
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

println("---------- USING @benchmark MACRO ----------")
println("Warm up call")
nothole = JuliaKernels.call_exit(px,py,pz,exit_collim_y,exit_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)
println("Exit")
@benchmark CuArrays.@sync $hole = JuliaKernels.call_exit($px,$py,$pz,$exit_collim_y,$exit_collim_z,$hole,$size_y,$size_z,$particle_size,$nBlocks,$nThreads)