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

N = 10
sets = [Set([k, k + 1]) for k = 1:N-1]

bs = PWAR.optimal_set_cover(N, sets, solver)

@testset "set_cover pairs" begin
    for k = 1:N-1
        @test bs[k] ≈ float(isodd(k))
    end
end

N = 11
sets = [Set([k, k + 1]) for k = 1:N-1]

bs = PWAR.optimal_set_cover(N, sets, solver)

@testset "set_cover pairs" begin
    @test sum(bs) ≈ 6
    for i = 1:N
        b = 0.0
        for (k, set) in enumerate(sets)
            b += (i ∈ set)*bs[k]
        end
        @test b ≥ 1
    end
end

N = 100
sets = vcat([Set([k, k + 1]) for k = 1:N-1], [Set(1:N)])

bs = PWAR.optimal_set_cover(N, sets, solver)

@testset "set_cover pairs" begin
    @test sum(bs) ≈ 1
    @test bs[N] ≈ 1
end

end # module