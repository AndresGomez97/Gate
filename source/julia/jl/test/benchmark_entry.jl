include("../src/JuliaKernels.jl")

using Test
using BenchmarkTools
using CuArrays

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

entry_collim_y = Array{Float32}(undef,size_y)
entry_collim_z = Array{Float32}(undef,size_z)

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

println("---------- USING @benchmark MACRO ----------")
println("Warm up call")
nothole = JuliaKernels.call_entry(px,py,pz,entry_collim_y,entry_collim_z,hole,size_y,size_z,particle_size,nBlocks,nThreads)
println("Entry")
@benchmark CuArrays.@sync $hole = JuliaKernels.call_entry($px,$py,$pz,$entry_collim_y,$entry_collim_z,$hole,$size_y,$size_z,$particle_size,$nBlocks,$nThreads)
#=
io = open("data/new_hole.txt","w")
for i = 1:particle_size
    write(io,hole[i])
end
close(io)
=#

