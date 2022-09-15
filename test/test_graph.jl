using Test
@static if isdefined(Main, :TestLocal)
    include("../src/PWARegression.jl")
else
    using PWARegression
end
PWAR = PWARegression

NT = PWAR.Node{Vector{Int},Int}
graph = PWAR.Graph(NT[])

@testset "length #0" begin
    @test length(graph) == 0
end

PWAR.add_node!(graph, PWAR.Node([1, 2], 4))
PWAR.add_node!(graph, PWAR.Node([0, 5], 4))

@testset "length #1" begin
    @test length(graph) == 2
end