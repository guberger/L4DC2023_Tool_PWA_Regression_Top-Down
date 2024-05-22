module Example

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
gurobi_() = Gurobi.Optimizer(GUROBI_ENV)
solver() = Model(optimizer_with_attributes(gurobi_, MOI.Silent()=>true))

colors = ("tab:orange", "tab:green", "tab:purple")

## Piecewise

NT = TK.Node{Vector{Float64},Float64}
nodes = NT[]
aref = [1, 2, 1]
θ = 0.5
for xt in Iterators.product(0:0.1:1, 0:0.2:2, (1.0,))
    local x = collect(xt)
    local η =
        xt[1] > 0.5 && xt[2] > 1.0 ? dot(aref, x) :
        xt[1] ≤ 0.5 && xt[2] > 1.0 ? +θ :
        xt[1] > 0.5 && xt[2] ≤ 1.0 ? -θ :
                                     (rand() - 0.5) * θ
    push!(nodes, TK.Node(x, η))
end
xlist = map(node -> node.x, nodes)

fig = figure()
ax = fig.add_subplot(projection="3d")
for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η, ls="none", marker=".", ms=5, c="tab:blue"
    )
end

ϵ = 0.76*θ
BD = 100
γ = 0.01
δ = 1e-5
inodes_list = @time TK.optimal_covering(
    nodes, ϵ, BD, γ, δ, 3, solver, solver, solver
)

bs = TK.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end

for (q, inodes) in enumerate(inodes_list_opt)
    local a = minimax_regression(nodes, inodes, BD, 3, solver)
    # plot
    local lb, ub = TK.compute_lims(xlist, inodes, 3)
    local x1rect = (lb[1], ub[1], ub[1], lb[1], lb[1])
    local x2rect = (lb[2], lb[2], ub[2], ub[2], lb[2])
    ax.plot(x1rect, x2rect, -1, c=colors[q])
    local x1_ = (lb[1], ub[1])
    local x2_ = (lb[2], ub[2])
    local Xt_ = Iterators.product(x1_, x2_)
    local X1_ = getindex.(Xt_, 1)
    local X2_ = getindex.(Xt_, 2)
    local Y_ = map(xt -> dot(a, (xt..., 1.0)), Xt_)
    ax.plot_surface(X1_, X2_, Y_, color=colors[q], lw=0, alpha=0.5)
end

ax.view_init(elev=14, azim=-70)
fig.savefig(
    "./examples/figures/exa_noisy2D.png", bbox_inches="tight", dpi=100
)

## Sine

NT = TK.Node{Vector{Float64},Float64}
nodes = NT[]
for xt in Iterators.product(0:0.05:1, 0:0.2:2, (1.0,))
    local x = collect(xt)
    local η = sin(xt[1]*7) + 2*xt[3]
    push!(nodes, TK.Node(x, η))
end
xlist = map(node -> node.x, nodes)

fig = figure()
ax = fig.add_subplot(projection="3d")
for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η,
        ls="none", marker=".", ms=5, c="tab:blue"
    )
end

ϵ = 0.15
BD = 100
γ = 0.01
δ = 1e-5
inodes_list = @time TK.optimal_covering(
    nodes, ϵ, BD, γ, δ, 3, solver, solver, solver
)

bs = TK.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end

for (q, inodes) in enumerate(inodes_list_opt)
    local a = minimax_regression(nodes, inodes, BD, 3, solver)
    # plot
    local lb, ub = TK.compute_lims(xlist, inodes, 3)
    local x1rect = (lb[1], ub[1], ub[1], lb[1], lb[1])
    local x2rect = (lb[2], lb[2], ub[2], ub[2], lb[2])
    ax.plot(x1rect, x2rect, -0.1, c=colors[q])
    local x1_ = (lb[1], ub[1])
    local x2_ = (lb[2], ub[2])
    local Xt_ = Iterators.product(x1_, x2_)
    local X1_ = getindex.(Xt_, 1)
    local X2_ = getindex.(Xt_, 2)
    local Y_ = map(xt -> dot(a, (xt..., 1.0)), Xt_)
    ax.plot_surface(X1_, X2_, Y_, color=colors[q], lw=0, alpha=0.5)
end

ax.view_init(elev=14, azim=-70)
fig.savefig(
    "./examples/figures/exa_sine2D.png", bbox_inches="tight", dpi=100
)

end # module