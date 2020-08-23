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

println("---------- USING @benchmark MACRO ----------")
println("Warm up call")
nothole = JuliaKernels.call_all(px,py,pz,dx,dy,dz,entry_collim_y,entry_collim_z,exit_collim_y,exit_collim_z,hole,size_y,size_z,planeToProject,particle_size,nBlocks,nThreads)
println("All")
@benchmark CuArrays.@sync $hole = JuliaKernels.call_all($px,$py,$pz,$dx,$dy,$dz,$entry_collim_y,$entry_collim_z,$exit_collim_y,$exit_collim_z,$hole,$size_y,$size_z,$planeToProject,$particle_size,$nBlocks,$nThreads)
