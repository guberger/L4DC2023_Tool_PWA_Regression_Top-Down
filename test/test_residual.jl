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

NT = PWAR.Node{Vector{Float64},Vector{Float64}}
graph = PWAR.Graph(NT[])
Aref = [1 2 1; 0 0 0]
for xt in Iterators.product(0:0.1:1, 0:0.2:2)
    local x = vcat(collect(xt), 1.0)
    local y = Aref*x
    PWAR.add_node!(graph, PWAR.Node(x, y))
end
x = [-0.1, 2.2, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, Aref*x - [1.0, 0.0]))
x = [1.1, -0.2, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, Aref*x - [1.0, 0.0]))
PWAR.add_node!(graph, PWAR.Node([5.0, 5.0, 1.0], [100.0, 0.0]))
PWAR.add_node!(graph, PWAR.Node([-5.0, -5.0, 1.0], [100.0, 0.0]))

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-4))
r = PWAR.LInf_residual(subgraph, 1000, 3, 2, solver)

@testset "LInf_residual no error" begin
    @test r ≈ 0
end

r = PWAR.LInf_residual(subgraph, 0, 3, 2, solver)

@testset "LInf_residual small arad" begin
    @test r ≈ 6
end

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-2))
r = PWAR.LInf_residual(subgraph, 1000, 3, 2, solver)

@testset "LInf_residual small error" begin
    @test r ≈ 1/2
end

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)))
r = PWAR.LInf_residual(subgraph, 1000, 3, 2, solver)

@testset "LInf_residual big error" begin
    @test r ≈ 99/2
end

# local_L2_residual

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

rescs = fill(NaN, 2)

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-1))
xc = [0.0, 0.0, 1.0]
σ = 0.2
PWAR.local_L2_residual_comp!(rescs, subgraph, xc, σ, 3, 2)

@testset "local_L2_residual very local" begin
    @test rescs[1] < 1e-5
    @test rescs[2] < 1e-9
end

σ = 0.3
PWAR.local_L2_residual_comp!(rescs, subgraph, xc, σ, 3, 2)

@testset "local_L2_residual less local" begin
    @test rescs[1] > 1e-5
    @test rescs[2] < 1e-9
end

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)))
xc = [0.0, 0.0, 1.0]
σ = 0.2
PWAR.local_L2_residual_comp!(rescs, subgraph, xc, σ, 3, 2)

@testset "local_L2_residual large error" begin
    @test rescs[1] > 1e-5
    @test rescs[2] < 1e-9
end

rescs_max = fill(-Inf, 2)
inodes_opt = fill(0, 2)

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-1))
σ = 0.2
PWAR.max_local_L2_residual_comp!(
    rescs_max, inodes_opt, rescs, subgraph, σ, 3, 2
)

@testset "max_local_L2_residual" begin
    @test graph.nodes[inodes_opt[1]].x ≈ [0.5, 1, 1]
    @test rescs_max[2] < 1e-9
end