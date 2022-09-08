using Test
@static if isdefined(Main, :TestLocal)
    include("../src/PWARegression.jl")
else
    using PWARegression
end
PWAR = PWARegression

NT = PWAR.Node{Vector{Int},Vector{Int}}
graph = PWAR.Graph(NT[])

@testset "length #0" begin
    @test length(graph) == 0
end

PWAR.add_node!(graph, PWAR.Node([1, 2], [2, 3, 4]))
PWAR.add_node!(graph, PWAR.Node([0, 5], [2, 3, 4]))

@testset "length #1" begin
    @test length(graph) == 2
end

for xt in Iterators.product(1:6, 9:15)
    PWAR.add_node!(graph, PWAR.Node([1, 2], [2, 3, 4]))
end

subgraph = PWAR.Subgraph(graph, BitSet(1))

@testset "length #1" begin
    @test length(subgraph) == 1
end

PWAR.add_inode!(subgraph, 5)
PWAR.add_inode!(subgraph, 5)
PWAR.add_inode!(subgraph, 1)

@testset "length #2" begin
    @test length(subgraph) == 2
end