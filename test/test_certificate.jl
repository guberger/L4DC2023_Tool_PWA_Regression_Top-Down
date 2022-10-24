using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/PWARegression.jl")
else
    using PWARegression
end
PWAR = PWARegression

solver() = Model(optimizer_with_attributes(
    HiGHS.Optimizer, "output_flag"=>false
))

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
aref = [1, 2, 1]
for xt in Iterators.product(0:0.2:1, 0:0.4:2)
    local x = vcat(collect(xt), 1.0)
    local η = dot(aref, x)
    push!(nodes, PWAR.Node(x, η))
end
x = [0.5, 1.0, 1.0]
push!(nodes, PWAR.Node(x, dot(aref, x) - 1.0))
push!(nodes, PWAR.Node(x, 1000.0))

inodes = BitSet(1:length(nodes)-1)
xc = [0.4, 0.8, 1.0]
λus, λls = PWAR.infeasibility_certificate(nodes, inodes, 0.5, xc, solver)

@testset "infeasibility_certificate" begin
    for inode in inodes
        if nodes[inode].x ≈ [0.5, 1.0, 1.0]
            @test λus[inode] ≈ 1/2
            @test λls[inode] < 1e-9
        elseif nodes[inode].x ≈ [0.4, 1.2, 1.0]
            @test λus[inode] < 1e-9
            @test λls[inode] ≈ 1/4
        elseif nodes[inode].x ≈ [0.6, 0.8, 1.0]
            @test λus[inode] < 1e-9
            @test λls[inode] ≈ 1/4
        else
            @test λus[inode] < 1e-9
            @test λls[inode] < 1e-9
        end
    end
end