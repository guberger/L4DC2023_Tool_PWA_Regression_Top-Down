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
Aref = [1 2 1; 0 0 0]
for xt in Iterators.product(0:0.1:1, 0:0.2:2)
    local x = vcat(collect(xt), 1.0)
    local y = Aref*x
    PWAR.add_node!(graph, PWAR.Node(x, y))
end
x = [0.55, 1.1, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, Aref*x - [1, 0]))
PWAR.add_node!(graph, PWAR.Node(x, [1000.0, 0.0]))

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-1))
xc = [0.5, 1.0, 1.0]
λus, λls, obj = PWAR.infeasibility_certificate(subgraph, 0.5, xc, 1, solver)

@testset "infeasibility_certificate" begin
    @test λus[122] ≈ 1/2
    @test λls[62] ≈ 1/4
    @test λls[72] ≈ 1/4
    @test obj ≈ (0.05^2 + 0.1^2)/2 + (0.1^2)/4 + (0.2^2)/4
end