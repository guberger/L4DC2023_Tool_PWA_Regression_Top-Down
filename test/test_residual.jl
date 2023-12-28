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

# LInf_residual

NT = PWAR.Node{Vector{Float64},Vector{Float64}}
nodes = NT[]
Aref = [1 2 1]
for xt in Iterators.product(0:0.1:1, 0:0.2:2, (1.0,))
    x = collect(xt)
    y = Aref * x
    push!(nodes, PWAR.Node(x, y))
end
x = [-0.1, 2.2, 1.0]
push!(nodes, PWAR.Node(x, Aref * x - [1.0]))
x = [1.1, -0.2, 1.0]
push!(nodes, PWAR.Node(x, Aref * x - [1.0]))
push!(nodes, PWAR.Node([5.0, 5.0, 1.0], [100.0]))
push!(nodes, PWAR.Node([-5.0, -5.0, 1.0], [100.0]))

inodes = BitSet(1:(length(nodes) - 4))
r = PWAR.LInf_residual(nodes, inodes, 1000, 1, 3, solver)

@testset "LInf_residual no error" begin
    @test r ≈ 0
end

r = PWAR.LInf_residual(nodes, inodes, 0, 1, 3, solver)

@testset "LInf_residual small BD" begin
    @test r ≈ 6
end

inodes = BitSet(1:length(nodes)-2)
r = PWAR.LInf_residual(nodes, inodes, 1000, 1, 3, solver)

@testset "LInf_residual small error" begin
    @test r ≈ 1/2
end

inodes = BitSet(1:length(nodes))
r = PWAR.LInf_residual(nodes, inodes, 1000, 1, 3, solver)

@testset "LInf_residual big error" begin
    @test r ≈ 99/2
end

end # module