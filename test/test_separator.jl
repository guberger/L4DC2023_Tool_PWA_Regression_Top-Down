using LinearAlgebra
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/PWARegression.jl")
else
    using PWARegression
end
PWAR = PWARegression

NT = PWAR.Node{Vector{Float64},Vector{Float64}}
graph = PWAR.Graph(NT[])
Aref = [1 2 1]
for xt in Iterators.product(0:0.1:1, 0:0.2:2)
    local x = vcat(collect(xt), 1.0)
    local y = Aref*x
    PWAR.add_node!(graph, PWAR.Node(x, y))
end
x = [0.5, 1.0, 1.0]
PWAR.add_node!(graph, PWAR.Node(x, Aref*x .- 1.0))
PWAR.add_node!(graph, PWAR.Node(x, [1000.0]))

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-1))
xc = [0.0, 0.0, 1.0]
σ = 0.2
A, res, ω_tot = PWAR.local_residual(subgraph, xc, σ, 3, 1)

@testset "local_residual very local" begin
    @test res < 1e-9
end

σ = 0.3
A, res, ω_tot = PWAR.local_residual(subgraph, xc, σ, 3, 1)

@testset "local_residual less local" begin
    @test res > 1e-9
end

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)))
xc = [0.0, 0.0, 1.0]
σ = 0.2
A, res, ω_tot = PWAR.local_residual(subgraph, xc, σ, 3, 1)

@testset "local_residual large error" begin
    @test res > 1e-9
end

subgraph = PWAR.Subgraph(graph, BitSet(1:length(graph)-1))
σ = 0.2
res, inode = PWAR.max_local_residual(subgraph, σ, 3, 1)

@testset "max_local_residual" begin
    @test graph.nodes[inode].x ≈ [0.5, 1, 1]
end