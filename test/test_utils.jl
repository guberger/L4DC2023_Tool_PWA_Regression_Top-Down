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

xlist = [[100, 1000], [7, 2], [2, 3], [4, 5], [3, 6], [-100, 1000]]

xc = zeros(2)
xc = PWAR.compute_center!(xlist, 1:length(xlist), xc)
Atemp = [1 0; 0 1; -1 0; 0 -1]
ctemp = PWAR.compute_lims(xlist, 1:length(xlist), Atemp)

@testset "utils: all" begin
    @test xc ≈ [8/3, 1008/3]
    @test ctemp ≈ [100, 1000, 100, -2]
end

fill!(xc, 0)
xc = PWAR.compute_center!(xlist, 2:5, xc)
ctemp = PWAR.compute_lims(xlist, 2:5, Atemp)

@testset "utils: subset" begin
    @test xc ≈ [4, 4]
    @test ctemp ≈ [7, 6, -2, -2]
end

end # module