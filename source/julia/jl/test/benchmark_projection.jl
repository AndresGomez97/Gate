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

dx = Array{Float32}(undef,particle_size)
dy = Array{Float32}(undef,particle_size)
dz = Array{Float32}(undef,particle_size)

io = open("data/new_hole.txt","r")
for i = 1:particle_size 
    hole[i] = read(io,Int32)
end
close(io)

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


println("---------- USING @benchmark MACRO ----------")
println("Warm up call")
notpx,notpy,notpz = JuliaKernels.call_projection(px,py,pz,dx,dy,dz,hole,planeToProject,particle_size,nBlocks,nThreads)
println("Projection")
@benchmark CuArrays.@sync $px,$py,$pz = JuliaKernels.call_projection($px,$py,$pz,$dx,$dy,$dz,$hole,$planeToProject,$particle_size,$nBlocks,$nThreads)
#=
io = open("data/new_px.txt","w")
for i = 1:particle_size
    write(io,px[i])
end
close(io)

io = open("data/new_py.txt","w")
for i = 1:particle_size
    write(io,py[i])
end
close(io)

io = open("data/new_pz.txt","w")
for i = 1:particle_size
    write(io,pz[i])
end
close(io)
=#