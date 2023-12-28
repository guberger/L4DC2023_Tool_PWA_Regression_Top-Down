module MyTest

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
nodes = NT[]
Aref1 = [1 2 1]
Aref2 = [0 2 2]
for xt in Iterators.product(0:0.2:1, 0:0.4:2)
    x = vcat(collect(xt), 1.0)
    y = xt[1] > 0.5 ? Aref1 * x : Aref2 * x
    push!(nodes, PWAR.Node(x, y))
end

ϵ = 0.1
BD = 100
β = 1e-8
γ = 0.01
σ = 0.2
δ = 1e-5
Atemp = [1 0 0; 0 1 0; -1 0 0; 0 -1 0]
inodes_list = PWAR.optimal_covering(nodes, ϵ, BD, γ, δ, 1, 3, Atemp,
                                    solver, solver, solver)

inodes_covered = BitSet()
for inodes in inodes_list
    union!(inodes_covered, inodes)
end

@testset "optimal_covering" begin
    @test inodes_covered == BitSet(1:length(nodes))
end

end # module