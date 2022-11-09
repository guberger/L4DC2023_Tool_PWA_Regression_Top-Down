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

xlist = [[100, 1000], [7, 2], [2, 3], [4, 5], [3, 6], [-100, 1000]]

xc = PWAR.compute_center(xlist, 1:length(xlist), 2)
lb, ub = PWAR.compute_lims(xlist, 1:length(xlist), 2)

@testset "utils: all" begin
    @test xc ≈ [8/3, 1008/3]
    @test lb ≈ [-100, 2]
    @test ub ≈ [100, 1000]
end

xc = PWAR.compute_center(xlist, 2:5, 2)
lb, ub = PWAR.compute_lims(xlist, 2:5, 2)

@testset "utils: subset" begin
    @test xc ≈ [4, 4]
    @test lb ≈ [2, 2]
    @test ub ≈ [7, 6]
end
