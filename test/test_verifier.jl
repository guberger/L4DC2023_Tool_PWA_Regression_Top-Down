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

NT = PWAR.Node{Vector{Float64},Vector{Float64}}
graph = PWAR.Graph(NT[])
Aref = [1 2 1]
for xt in Iterators.product(0:0.1:1, 0:0.2:2)
    local x = vcat(collect(xt), 1.0)
    local y = Aref*x
    PWAR.add_node!(graph, PWAR.Node(x, y))
end
x = [-0.1, 2.2, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, Aref*x .- 1.0))
x = [1.1, -0.2, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, Aref*x .- 1.0))
PWAR.add_node!(graph, PWAR.Node([5.0, 5.0, 1.0], [100.0]))
PWAR.add_node!(graph, PWAR.Node([-5.0, -5.0, 1.0], [100.0]))

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-4))
r1 = PWAR.verify(subgraph, 1000, 3, 1, solver)
r2 = PWAR.verify(subgraph, 0, 3, 1, solver)

@testset "verify no error / small arad" begin
    @test r1 ≈ 0
    @test r2 ≈ 6
end

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-2))
r = PWAR.verify(subgraph, 1000, 3, 1, solver)

@testset "verify small error" begin
    @test r ≈ 1/2
end

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)))
r = PWAR.verify(subgraph, 1000, 3, 1, solver)

@testset "verify big error" begin
    @test r ≈ 99/2
end