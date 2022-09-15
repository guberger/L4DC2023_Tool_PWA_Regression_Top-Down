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

# LInf_residual

NT = PWAR.Node{Vector{Float64},Float64}
graph = PWAR.Graph(NT[])
aref = [1, 2, 1]
for xt in Iterators.product(0:0.1:1, 0:0.2:2)
    local x = vcat(collect(xt), 1.0)
    local η = dot(aref, x)
    PWAR.add_node!(graph, PWAR.Node(x, η))
end
x = [-0.1, 2.2, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, dot(aref, x) - 1.0))
x = [1.1, -0.2, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, dot(aref, x) - 1.0))
PWAR.add_node!(graph, PWAR.Node([5.0, 5.0, 1.0], 100.0))
PWAR.add_node!(graph, PWAR.Node([-5.0, -5.0, 1.0], 100.0))

inodes = BitSet(1:length(graph)-4)
r = PWAR.LInf_residual(graph, inodes, 1000, 3, solver)

@testset "LInf_residual no error" begin
    @test r ≈ 0
end

r = PWAR.LInf_residual(graph, inodes, 0, 3, solver)

@testset "LInf_residual small BD" begin
    @test r ≈ 6
end

inodes = BitSet(1:length(graph)-2)
r = PWAR.LInf_residual(graph, inodes, 1000, 3, solver)

@testset "LInf_residual small error" begin
    @test r ≈ 1/2
end

inodes = BitSet(1:length(graph))
r = PWAR.LInf_residual(graph, inodes, 1000, 3, solver)

@testset "LInf_residual big error" begin
    @test r ≈ 99/2
end

# local_L2_residual

NT = PWAR.Node{Vector{Float64},Float64}
graph = PWAR.Graph(NT[])
aref = [1, 2, 1]
for xt in Iterators.product(0:0.1:1, 0:0.2:2)
    local x = vcat(collect(xt), 1.0)
    local η = dot(aref, x)
    PWAR.add_node!(graph, PWAR.Node(x, η))
end
x = [0.5, 1.0, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, dot(aref, x) - 1.0))
PWAR.add_node!(graph, PWAR.Node(x, 1000.0))

inodes = BitSet(1:length(graph)-1)
xc = [0.0, 0.0, 1.0]
σ = 0.2
β = 1e-8
res = PWAR.local_L2_residual(graph, inodes, xc, σ, β, 3)

@testset "local_L2_residual small σ small β" begin
    @test res < 1e-5
end

σ = 0.2
β = 1e-4
res = PWAR.local_L2_residual(graph, inodes, xc, σ, β, 3)

@testset "local_L2_residual small σ large β" begin
    @test res > 1e-4
end

σ = 0.3
β = 1e-8
res = PWAR.local_L2_residual(graph, inodes, xc, σ, β, 3)

@testset "local_L2_residual large σ small β" begin
    @test res > 1e-5
end

inodes = BitSet(1:length(graph))
xc = [0.0, 0.0, 1.0]
σ = 0.2
β = 1e-8
res = PWAR.local_L2_residual(graph, inodes, xc, σ, β, 3)

@testset "local_L2_residual large error" begin
    @test res > 1e-5
end

inodes = BitSet(1:length(graph)-1)
σ = 0.2
β = 1e-8
res, inode = PWAR.max_local_L2_residual(graph, inodes, σ, β, 3)

@testset "max_local_L2_residual" begin
    @test graph.nodes[inode].x ≈ [0.5, 1, 1]
end