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

colors = ("tab:blue", "tab:orange", "tab:green")

bbox_figs = ((1.7, 0.7), (5.2, 3.65))

UID(x1, x2) = (3.2667 + 0.0313*x1)*x2/(253.52 + x2)

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
nsample = 100
for isample = 1:nsample
    local xt = (rand()*50, rand()*200, 1)
    # local xt = (rand()*500, rand()*1000, 1) 
    local x = Float64.(collect(xt))
    local η = UID(xt[1], xt[2])
    push!(nodes, PWAR.Node(x, η))
end
xlist = map(node -> node.x, nodes)

fig = figure()
ax = fig.add_subplot(projection="3d")

nplot = 100

x1_ = range(0, 50, length=nplot)
x2_ = range(0, 200, length=nplot)
Xt_ = Iterators.product(x1_, x2_)
X1_ = getindex.(Xt_, 1)
X2_ = getindex.(Xt_, 2)
Y_ = map(xt -> UID(xt[1], xt[2]), Xt_)
ax.plot_surface(
    X1_, X2_, Y_, cmap=matplotlib.cm.coolwarm, lw=0, alpha=0.5
)

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
    "./examples/figures/exa_glucose_data.png",
    bbox_inches=matplotlib.transforms.Bbox(bbox_figs), dpi=100
)

# regions

ϵ = 0.2
BD = 100
γ = 0.01
δ = 1e-5
inodes_list = @time PWAR.optimal_covering(
    nodes, ϵ, BD, γ, δ, 3, solver, solver, solver
)

bs = PWAR.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end

fig = figure()
ax = fig.add_subplot(projection="3d")

for (q, inodes) in enumerate(inodes_list_opt)
    local a = minimax_regression(nodes, inodes, BD, 3, solver)
    # plot
    local lb, ub = PWAR.compute_lims(xlist, inodes, 3)
    local x1rect = (lb[1], ub[1], ub[1], lb[1], lb[1])
    local x2rect = (lb[2], lb[2], ub[2], ub[2], lb[2])
    ax.plot(x1rect, x2rect, 0, c=colors[q])
    display((lb, ub))
    display(a)
    local x1_ = (lb[1], ub[1])
    local x2_ = (lb[2], ub[2])
    local Xt_ = Iterators.product(x1_, x2_)
    local X1_ = getindex.(Xt_, 1)
    local X2_ = getindex.(Xt_, 2)
    local Y_ = map(xt -> dot(a, (xt..., 1.0)), Xt_)
    ax.plot_surface(X1_, X2_, Y_, color=colors[q], lw=0, alpha=0.5)
end

for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η, ls="none", marker=".", ms=7, c="k"
    )
end

ax.set_xlim(ax_data.get_xlim())
ax.set_ylim(ax_data.get_ylim())
ax.set_zlim(ax_data.get_zlim())
ax.xaxis.pane.fill = false
ax.yaxis.pane.fill = false
ax.zaxis.pane.fill = false
ax.set_xlabel(L"x_1")
ax.set_ylabel(L"x_2")

ax.view_init(elev=14, azim=-70)
fig.savefig(
    string("./examples/figures/exa_glucose_", ϵ, ".png"),
    bbox_inches=matplotlib.transforms.Bbox(bbox_figs), dpi=100
)

# MILP

ϵ = 0.05
M = 3
Γ = 1000*BD
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>true, "TimeLimit" => 10
))

inodes_all = BitSet(eachindex(nodes))
as, bins = switched_bounded_regression(
    nodes, inodes_all, M, ϵ, BD, 3, Γ, solver
)

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

for (q, inodes) in enumerate(inodes_list_sw)
    for inode in inodes
        local node = nodes[inode]
        ax.plot(
            node.x[1], node.x[2],
            ls="none", marker=".", ms=10, c=colors[q]
        )
    end
end

ax.set_xlabel(L"x_1")
ax.set_ylabel(L"x_2")

hs = [
    matplotlib.lines.Line2D(
        (0,), (0,), ls="none", c=colors[q], marker=".", ms=10
    ) for q in 1:M
]
ax.legend(hs, [string("cluster ", q) for q in 1:M])

fig.savefig(
    string("./examples/figures/exa_glucose_milp.png"),
    bbox_inches="tight", dpi=100
)