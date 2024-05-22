using Random
using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

Random.seed!(0)

include("../src/main.jl")
TK = ToolKit

include("./utils.jl")

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

colors = ("tab:orange", "tab:green", "tab:purple")

F(x) = atan(10*x)*exp(-abs(x))

NT = TK.Node{Vector{Float64},Float64}
nodes = NT[]
push!(nodes, TK.Node([0.0, 1.0], 0.0))
for xt in Iterators.product(0.2:0.2:1, (1.0,))
    for sg in (-1, 1)
        local x = collect(xt.*(sg, 1))
        local η = F(x[1])
        push!(nodes, TK.Node(x, η))
    end
end
xlist = map(node -> node.x, nodes)

fig = figure(figsize=(4.5, 3))
ax = fig.add_subplot()

xplot = range(-1, 1, length=100)
yplot = map(x -> F(x), xplot)
ax.plot(xplot, yplot, ls="dashed", lw=3, c="k")

for node in nodes
    ax.plot(node.x[1], node.η, ls="none", marker=".", ms=15, c="tab:blue")
end

# regions

ϵ = 0.1
BD = 100
γ = 0.01
δ = 1e-5
inodes_list = TK.optimal_covering(
    nodes, ϵ, BD, γ, δ, 2, solver, solver, solver
)

bs = TK.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end

for (q, inodes) in enumerate(inodes_list_opt)
    local a = minimax_regression(nodes, inodes, BD, 2, solver)
    # plot
    local lb, ub = TK.compute_lims(xlist, inodes, 2)
    local x1_ = (lb[1], ub[1])
    local Xt_ = Iterators.product(x1_)
    local X1_ = getindex.(Xt_, 1)
    local Y_ = map(xt -> dot(a, (xt..., 1.0)), Xt_)
    display((lb, ub))
    display(a)
    ax.plot(X1_, Y_, lw=3, c=colors[q])
end

ax.set_xlabel(L"x")
ax.set_ylabel(L"y")

fig.savefig(
    "./examples/figures/exa_synthetic1D.png",
    bbox_inches="tight", dpi=100
)

# MILP

M = 3
Γ = 2*BD
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>true
))

inodes_all = BitSet(eachindex(nodes))
as, bins = switched_bounded_regression(
    nodes, inodes_all, M, ϵ, BD, 2, Γ, solver
)

inodes_list_sw = [BitSet() for q in 1:M]
for (inode, bin) in bins
    for q in 1:M
        if round(Int, bin[q]) == 1
            push!(inodes_list_sw[q], inode)
        end
    end
end

fig = figure(figsize=(6, 3))
ax = fig.add_subplot()

ax.plot(xplot, yplot, ls="dashed", lw=3, c="k")

for (q, inodes) in enumerate(inodes_list_sw)
    for inode in inodes
        local node = nodes[inode]
        ax.plot(
                node.x[1], node.η, ls="none", marker=".", ms=15, c=colors[q]
        )
    end
    local lb, ub = TK.compute_lims(xlist, inodes, 2)
    local x1_ = (lb[1], ub[1])
    local Xt_ = Iterators.product(x1_)
    local X1_ = getindex.(Xt_, 1)
    local Y_ = map(xt -> dot(as[q], (xt..., 1.0)), Xt_)
    ax.plot(X1_, Y_, lw=3, c=colors[q])
end

ax.set_xlabel(L"x")
ax.set_ylabel(L"y")