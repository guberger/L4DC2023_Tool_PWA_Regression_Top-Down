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

colors = ("tab:orange", "tab:green", "tab:purple")

bbox_figs = ((1.7, 0.75), (5.35, 3.65))

## Piecewise

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
aref = [1, 2, 1]
θ = 0.5
for xt in Iterators.product(0:0.1:1, 0:0.2:2, (1.0,))
    local x = collect(xt)
    local η = xt[1] > 0.5 && xt[2] > 1.0 ? dot(aref, x) :
        xt[1] ≤ 0.5 && xt[2] > 1.0 ? θ :
        xt[1] > 0.5 && xt[2] ≤ 1.0 ? -θ :
        (rand() - 0.5)*θ
    push!(nodes, PWAR.Node(x, η))
end
xlist = map(node -> node.x, nodes)

fig = figure()
ax = fig.add_subplot(projection="3d")
ax.tick_params(axis="both", which="major", labelsize=15)

for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η,
        ls="none", marker=".", ms=5, c="tab:blue"
    )
end

ax.xaxis.pane.fill = false
ax.yaxis.pane.fill = false
ax.zaxis.pane.fill = false
ax.set_xticks(())
ax.set_xlabel(L"x_1", labelpad=0, fontsize=15)
ax.set_yticks(())
ax.set_ylabel(L"x_2", labelpad=0, fontsize=15)
ax.set_zticks((0, 6))
ax.set_zlabel("y", labelpad=0, fontsize=15)
ax.grid(false)

ax.view_init(elev=14, azim=-70)
fig.savefig(
    "./examples/figures/exa_illustration_pwa_data.png",
    bbox_inches=matplotlib.transforms.Bbox(bbox_figs), dpi=100
)

ϵ = 0.76*θ
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
ax.tick_params(axis="both", which="major", labelsize=15)

for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η,
        ls="none", marker=".", ms=5, c="tab:blue"
    )
end

for (q, inodes) in enumerate(inodes_list_opt)
    local a = minimax_regression(nodes, inodes, BD, 3, solver)
    # plot
    local lb, ub = PWAR.compute_lims(xlist, inodes, 3)
    local x1_ = (lb[1], ub[1])
    local x2_ = (lb[2], ub[2])
    local Xt_ = Iterators.product(x1_, x2_)
    local X1_ = getindex.(Xt_, 1)
    local X2_ = getindex.(Xt_, 2)
    local Y_ = map(xt -> dot(a, (xt..., 1.0)), Xt_)
    ax.plot_surface(X1_, X2_, Y_, color=colors[q], lw=0, alpha=0.5)
end

ax.xaxis.pane.fill = false
ax.yaxis.pane.fill = false
ax.zaxis.pane.fill = false
ax.set_xticks(())
ax.set_xlabel(L"x_1", labelpad=0, fontsize=15)
ax.set_yticks(())
ax.set_ylabel(L"x_2", labelpad=0, fontsize=15)
ax.set_zticks((0, 6))
ax.set_zlabel("y", labelpad=0, fontsize=15)
ax.grid(false)

ax.view_init(elev=14, azim=-70)
fig.savefig(
    "./examples/figures/exa_illustration_pwa_regr.png",
    bbox_inches=matplotlib.transforms.Bbox(bbox_figs), dpi=100
)

## Sine

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
for xt in Iterators.product(0:0.05:1, 0:0.2:2, (1.0,))
    local x = collect(xt)
    local η = sin(xt[1]*7) + 2*xt[3]
    push!(nodes, PWAR.Node(x, η))
end
xlist = map(node -> node.x, nodes)

fig = figure()
ax = fig.add_subplot(projection="3d")
ax.tick_params(axis="both", which="major", labelsize=15)

for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η,
        ls="none", marker=".", ms=5, c="tab:blue"
    )
end

ax.xaxis.pane.fill = false
ax.yaxis.pane.fill = false
ax.zaxis.pane.fill = false
ax.set_xticks(())
ax.set_xlabel(L"x_1", labelpad=0, fontsize=15)
ax.set_yticks(())
ax.set_ylabel(L"x_2", labelpad=0, fontsize=15)
ax.set_zticks((0, 3))
ax.set_zlabel("y", labelpad=0, fontsize=15)
ax.grid(false)

ax.view_init(elev=14, azim=-70)
fig.savefig(
    "./examples/figures/exa_illustration_sine_data.png",
    bbox_inches=matplotlib.transforms.Bbox(bbox_figs), dpi=100
)

ϵ = 0.15
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
ax.tick_params(axis="both", which="major", labelsize=15)

for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η,
        ls="none", marker=".", ms=5, c="tab:blue"
    )
end

for (q, inodes) in enumerate(inodes_list_opt)
    local a = minimax_regression(nodes, inodes, BD, 3, solver)
    # plot
    local lb, ub = PWAR.compute_lims(xlist, inodes, 3)
    local x1_ = (lb[1], ub[1])
    local x2_ = (lb[2], ub[2])
    local Xt_ = Iterators.product(x1_, x2_)
    local X1_ = getindex.(Xt_, 1)
    local X2_ = getindex.(Xt_, 2)
    local Y_ = map(xt -> dot(a, (xt..., 1.0)), Xt_)
    ax.plot_surface(X1_, X2_, Y_, color=colors[q], lw=0, alpha=0.5)
end

ax.xaxis.pane.fill = false
ax.yaxis.pane.fill = false
ax.zaxis.pane.fill = false
ax.set_xticks(())
ax.set_xlabel(L"x_1", labelpad=0, fontsize=15)
ax.set_yticks(())
ax.set_ylabel(L"x_2", labelpad=0, fontsize=15)
ax.set_zticks((0, 3))
ax.set_zlabel("y", labelpad=0, fontsize=15)
ax.grid(false)

ax.view_init(elev=14, azim=-70)
fig.savefig(
    "./examples/figures/exa_illustration_sine_regr.png",
    bbox_inches=matplotlib.transforms.Bbox(bbox_figs), dpi=100
)