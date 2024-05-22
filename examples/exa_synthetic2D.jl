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

colors = collect(keys(matplotlib.colors.TABLEAU_COLORS))

NT = TK.Node{Vector{Float64},Float64}
nodes = NT[]
aref1 = [1, 2, 0]
aref2 = [0, 2, 0]
aref3 = [0, -2, 2]
aref4 = [1, 0, 2]
for xt in Iterators.product(-1:0.05:1, -1:0.05:1, (1.0,))
    local x = collect(xt)
    local η = xt[1] > 0.2 && xt[2] > -0.2 ? dot(aref1, x) :
        xt[1] ≤ 0.2 && xt[2] > 0.2 ? dot(aref2, x) :
        xt[1] ≤ -0.2 && xt[2] ≤ 0.2 ? dot(aref3, x) :
        xt[1] > -0.2 && xt[2] ≤ -0.2 ? dot(aref4, x) : 3.0
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

# regions

ϵ = 0.01
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
    "./examples/figures/exa_synthetic2D.png", bbox_inches="tight", dpi=100
)