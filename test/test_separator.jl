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
x = [0.5, 1.0, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, Aref*x - [1, 0]))
PWAR.add_node!(graph, PWAR.Node(x, [1000.0, 0.0]))

res = fill(NaN, 2)

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-1))
xc = [0.0, 0.0, 1.0]
σ = 0.2
PWAR.local_residual!(res, subgraph, xc, σ, 3, 2)

@testset "local_residual very local" begin
    @test res[1] < 1e-5
    @test res[2] < 1e-9
end

σ = 0.3
PWAR.local_residual!(res, subgraph, xc, σ, 3, 2)

@testset "local_residual less local" begin
    @test res[1] > 1e-5
    @test res[2] < 1e-9
end

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)))
xc = [0.0, 0.0, 1.0]
σ = 0.2
PWAR.local_residual!(res, subgraph, xc, σ, 3, 2)

@testset "local_residual large error" begin
    @test res[1] > 1e-5
    @test res[2] < 1e-9
end

res_max = fill(-Inf, 2)
inodes_opt = fill(0, 2)

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-1))
σ = 0.2
PWAR.max_local_residual!(res_max, inodes_opt, res, subgraph, σ, 3, 2)

@testset "max_local_residual" begin
    @test graph.nodes[inodes_opt[1]].x ≈ [0.5, 1, 1]
    @test res_max[2] < 1e-9
end

xc = [0.5, 1.0, 1.0]
λus, λls, obj = PWAR.find_infeasible(subgraph, 0.5, xc, 1, solver)

@testset "find_infeasible" begin
    @test λus[122] ≈ 0.5
    @test λls[61] ≈ 0.5
    @test obj < 1e-9
end

xc = [0.4, 0.8, 1.0]
λus, λls, obj = PWAR.find_infeasible(subgraph, 0.5, xc, 1, solver)

@testset "find_infeasible" begin
    @test λus[122] ≈ 0.5
    @test λls[61] ≈ 0.5
    @test obj ≈ 0.05
end