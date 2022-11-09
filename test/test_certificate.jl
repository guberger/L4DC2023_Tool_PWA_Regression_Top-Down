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
nodes = NT[]
aref = [1, 2, 1]
for xt in Iterators.product(0:0.2:1, 0:0.4:2, (1.0,))
    local x = collect(xt)
    local η = dot(aref, x)
    push!(nodes, PWAR.Node(x, η))
end
x = [0.5, 1.0, 1.0]
push!(nodes, PWAR.Node(x, dot(aref, x) - 1.0))
push!(nodes, PWAR.Node(x, 1000.0))

inodes = BitSet(1:length(nodes)-1)
xc = [0.4, 0.8, 1.0]

obj_LP, λus, λls = PWAR.infeasibility_certificate_LP(
    nodes, inodes, 0.5, xc, 3, solver
)

@testset "infeasibility_certificate_LP" begin
    @test obj_LP ≈ 0.05/2 + 0.16/4 + 0.04/4
    for inode in inodes
        if nodes[inode].x ≈ [0.5, 1.0, 1.0]
            @test λus[inode] ≈ 1/2
            @test λls[inode] < 1e-9
        elseif nodes[inode].x ≈ [0.4, 1.2, 1.0]
            @test λus[inode] < 1e-9
            @test λls[inode] ≈ 1/4
        elseif nodes[inode].x ≈ [0.6, 0.8, 1.0]
            @test λus[inode] < 1e-9
            @test λls[inode] ≈ 1/4
        else
            @test λus[inode] < 1e-9
            @test λls[inode] < 1e-9
        end
    end
end

obj_MILP, bins = PWAR.infeasibility_certificate_MILP(
    nodes, inodes, 0.5, xc, 3, solver
)

@testset "infeasibility_certificate_MILP" begin
    @test obj_MILP ≈ 0.05 + 0.16 + 0.04
end

inodes_cert_LP = PWAR.extract_infeasibility_certificate_LP(
    nodes, inodes, λus, λls, 0.499, 0.1, 1e4, 3, solver
)

@testset "extract_infeasibility_certificate_LP" begin
    @test length(inodes_cert_LP) == 3
    @test any(inode -> nodes[inode].x ≈ [0.5, 1.0, 1.0], inodes_cert_LP)
    @test any(inode -> nodes[inode].x ≈ [0.4, 1.2, 1.0], inodes_cert_LP)
    @test any(inode -> nodes[inode].x ≈ [0.6, 0.8, 1.0], inodes_cert_LP)
end

inodes_cert_MILP = PWAR.extract_infeasibility_certificate_MILP(
    inodes, bins, 0.5
)

@testset "extract_infeasibility_certificate_LP" begin
    @test length(inodes_cert_MILP) == 3
    @test any(inode -> nodes[inode].x ≈ [0.5, 1.0, 1.0], inodes_cert_MILP)
    @test any(inode -> nodes[inode].x ≈ [0.4, 1.2, 1.0], inodes_cert_MILP)
    @test any(inode -> nodes[inode].x ≈ [0.6, 0.8, 1.0], inodes_cert_MILP)
end