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

bbox_figs = ((1.7, 0.7), (5.2, 3.65))

str = readlines(string(@__DIR__, "/exa_acrobot_data.txt"))
NT = TK.Node{Vector{Float64},Float64}
nodes = NT[]
for ln in str
    local ln = replace(ln, r"[\(\),]"=>"")
    local words = split(ln)
    @assert length(words) == 4
    local x = parse.(Float64, [words[1], words[2], "1"])
    local η = parse(Float64, words[3])
    push!(nodes, TK.Node(x, η))
end
shuffle!(nodes)
xlist = map(node -> node.x, nodes)
nodes = nodes[1:100]
x1_list = sort(map(node -> node.x[1], nodes))
x2_list = sort(map(node -> node.x[2], nodes))
display(minimum(diff(x1_list)))
display(minimum(diff(x2_list)))
# filter!(node -> node.x[1] ≥ 0.05 && node.x[2] ≤ -0.05, nodes)

fig = figure()
ax = fig.add_subplot(projection="3d")
for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η, ls="none", marker=".", ms=7, c="k"
    )
end

ax.xaxis.pane.fill = false
ax.yaxis.pane.fill = false
ax.zaxis.pane.fill = false
ax.set_xlabel(L"x_1")
ax.set_ylabel(L"x_2")
ax_data = ax

ax.view_init(elev=14, azim=-70)
fig.savefig(
    "./examples/figures/exa_acrobot_data.png",
    bbox_inches=matplotlib.transforms.Bbox(bbox_figs), dpi=100
)

# regions

ϵ = 0.01
BD = 1e5
γ = 0.01
δ = 1e-6
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

fig = figure(figsize=(6, 3))
ax = fig.add_subplot()

nplot = 50

for (q, inodes) in enumerate(inodes_list_opt)
    local a = minimax_regression(nodes, inodes, BD, 3, solver)
    # plot
    local lb, ub = TK.compute_lims(xlist, inodes, 3)
    local x1rect = (lb[1], ub[1], ub[1], lb[1], lb[1])
    local x2rect = (lb[2], lb[2], ub[2], ub[2], lb[2])
    ax.plot(x1rect, x2rect, 0, c=colors[q])
    display((lb, ub))
    display(a)
    local x1_ = range(lb[1], ub[1], length=nplot)
    local x2_ = range(lb[2], ub[2], length=nplot)
    local Xt_ = Iterators.product(x1_, x2_)
    local X1_ = getindex.(Xt_, 1)
    local X2_ = getindex.(Xt_, 2)
    local Y_ = map(xt -> dot(a, (xt..., 1.0)), Xt_)
    ax.contourf(X1_, X2_, Y_, -20:2:20, cmap=matplotlib.cm.coolwarm)
end

for node in nodes
    ax.plot(
        node.x[1], node.x[2], ls="none", marker=".", ms=5, c="k"
    )
end

fig.colorbar(matplotlib.cm.ScalarMappable(
    norm=matplotlib.colors.Normalize(vmin=-20, vmax=20),
    cmap=matplotlib.cm.coolwarm
))

ax.set_xlim(ax_data.get_xlim())
ax.set_ylim(ax_data.get_ylim())
ax.set_xlabel(L"x_1")
ax.set_ylabel(L"x_2")

fig.savefig(
    string("./examples/figures/exa_acrobot_regr.png"),
    bbox_inches="tight", dpi=100
)