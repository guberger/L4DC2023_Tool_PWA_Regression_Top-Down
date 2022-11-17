using Random
using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

Random.seed!(0)

include("../src/PWARegression.jl")
PWAR = PWARegression

include("./utils.jl")

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))
solver_MILP() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>true, "TimeLimit" => 1e3
))

colors = collect(keys(matplotlib.colors.TABLEAU_COLORS))

str = readlines(string(@__DIR__, "/exa_acrobot_data.txt"))
NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
for ln in str
    local ln = replace(ln, r"[\(\),]"=>"")
    local words = split(ln)
    @assert length(words) == 4
    local x = parse.(Float64, [words[1], words[2], "1"])
    local η = parse(Float64, words[3])
    push!(nodes, PWAR.Node(x, η))
end
shuffle!(nodes)
# 50: 17 secs
# 51: 350 secs
# 52: > 1000 secs
κ = maximum(node -> norm(node.x, 1), nodes)
display(κ)

# MILP

ϵ = 0.01
BD = 1e5
γ = 0.01
δ = 1e-6
M = 9
Γ = 2*κ*BD

_optimal_covering(nodes) = PWAR.optimal_covering(
    nodes, ϵ, BD, γ, δ, 3, solver, solver, solver
)
_switched_bounded_regression(nodes) = switched_bounded_regression(
    nodes, BitSet(eachindex(nodes)), M, ϵ, BD, 3, Γ, solver_MILP
)
_optimal_covering(nodes[1:10])
_switched_bounded_regression(nodes[1:10])

fig = figure(figsize=(4.5, 3.5))
ax = fig.add_subplot()
ax.set_yscale("log")

time2_old = 0

for nnode in 10:10:11
    local nodes2 = nodes[1:nnode]
    local time1 = @elapsed _optimal_covering(nodes2)
    ax.plot(nnode, time1, marker=".", ms=10, c="tab:orange")
    time2_old > 1e3 && continue
    local time2 = NaN
    try
        time2 = @elapsed _switched_bounded_regression(nodes2)
    catch err
        display(err)
        if isa(err, AssertionError)
            time2 = Inf
        else
            throw(err)
        end
    end
    global time2_old = time2
    time2_old > 1e3 && continue
    ax.plot(nnode, time2, marker="x", ms=10, c="tab:purple")
end

ax.set_xlabel("Number of data points")
ax.set_ylabel("Computation time [sec]")

hs = (
    matplotlib.lines.Line2D(
        (0,), (0,), ls="none", marker=".", ms=10, c="tab:orange"
    ),
    matplotlib.lines.Line2D(
        (0,), (0,), ls="none", marker="x", ms=10, c="tab:purple"
    )
)
ax.legend(hs, ("Our approach", "SA regression"))

fig.savefig(
    string("./examples/figures/exa_acrobot_compare.png"),
    bbox_inches="tight", dpi=100
)

as, bins = _switched_bounded_regression(nodes[1:51])

inodes_list_sw = [BitSet() for q in 1:M]
for (inode, bin) in bins
    for q in 1:M
        if round(Int, bin[q]) == 1
            push!(inodes_list_sw[q], inode)
        end
    end
end

fig = figure()
ax = fig.add_subplot()

for (inode, bin) in bins
    local node = nodes[inode]
    q = findfirst(q -> round(Int, bin[q]) == 1, 1:M)
    ax.plot(
        node.x[1], node.x[2],
        ls="none", marker=".", ms=10, c=colors[q]
    )
end

ax.set_xlabel(L"x_1")
ax.set_ylabel(L"x_2")

hs = [
    matplotlib.lines.Line2D(
        (0,), (0,), ls="none", c=colors[q], marker=".", ms=10
    ) for q in 1:M
]
ax.legend(
    hs, [string("cluster ", q) for q in 1:M],
    bbox_to_anchor=(1.0, 1.01)
)

fig.savefig(
    string("./examples/figures/exa_acrobot_milp.png"),
    bbox_inches="tight", dpi=100
)