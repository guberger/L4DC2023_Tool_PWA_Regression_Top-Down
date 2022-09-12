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
graph = PWAR.Graph(NT[])
aref1 = [1, 2, 1]
aref2 = [0, 2, 2]
for xt in Iterators.product(0:0.2:1, 0:0.4:2)
    local x = vcat(collect(xt), 1.0)
    local η = xt[1] > 0.5 ? dot(aref1, x) : dot(aref2, x)
    PWAR.add_node!(graph, PWAR.Node(x, η))
end

ϵ = 0.1
BD = 100
γ = 0.01
σ = 0.2
θ = 1e-5
δ = 1e-5
subgraphs = PWAR.maximal_regions(graph, ϵ, BD, γ, σ, θ, δ, 3, solver)

inodes_covered = BitSet()
for subgraph in subgraphs
    for inode in subgraph.inodes
        push!(inodes_covered, inode)
    end
end

@testset "maximal_regions" begin
    @test inodes_covered == BitSet(1:length(graph))
end