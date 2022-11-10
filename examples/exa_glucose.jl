using Random
using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

Random.seed!(0)

include("../src/PWARegression.jl")
PWAR = PWARegression

bbox_figs = ((1.7, 0.7), (5.2, 3.65))

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

UID(x1, x2) = (3.2667 + 0.0313*x1)*x2/(253.52 + x2)

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
# for xt in Iterators.product(0:300:4500, 0:100:1000, (1,))
#     local x = Float64.(collect(xt))
#     local η = (3.2667 + 0.0313*xt[1])*xt[2]/(253.52 + xt[2])
#     push!(nodes, PWAR.Node(x, η))
# end
nsample = 100
for isample = 1:nsample
    local xt = (rand()*50, rand()*200, 1)
    # local xt = (rand()*500, rand()*1000, 1) 
    local x = Float64.(collect(xt))
    local η = UID(xt[1], xt[2])
    push!(nodes, PWAR.Node(x, η))
end

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

ϵ = 0.05
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

# Plots

fig = figure()
ax = fig.add_subplot(projection="3d")

for inodes in inodes_list_opt
    # optim parameters
    local model = solver()
    local a = @variable(model, [1:3], lower_bound=-BD, upper_bound=BD)
    local r = @variable(model, lower_bound=-1)
    for inode in inodes
        local node = nodes[inode]
        @constraint(model, dot(a, node.x) ≤ node.η + r)
        @constraint(model, dot(a, node.x) ≥ node.η - r)
    end
    @objective(model, Min, r)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    local a_opt = value.(a)
    # plot
    local lb = fill(Inf, 2)
    local ub = fill(-Inf, 2)
    for inode in inodes
        node = nodes[inode]
        for k = 1:2
            if node.x[k] < lb[k]
                lb[k] = node.x[k]
            end
            if node.x[k] > ub[k]
                ub[k] = node.x[k]
            end
        end
    end
    display((lb, ub))
    display(a_opt)
    local x1rect = (lb[1], ub[1], ub[1], lb[1], lb[1])
    local x2rect = (lb[2], lb[2], ub[2], ub[2], lb[2])
    local p = ax.plot(x1rect, x2rect, 0)
    local x1_ = (lb[1], ub[1])
    local x2_ = (lb[2], ub[2])
    local Xt_ = Iterators.product(x1_, x2_)
    local X1_ = getindex.(Xt_, 1)
    local X2_ = getindex.(Xt_, 2)
    local Y_ = map(xt -> dot(a_opt, (xt..., 1.0)), Xt_)
    ax.plot_surface(X1_, X2_, Y_, color=p[1].get_c(), lw=0, alpha=0.5)
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

# 3580 iter, 67 secs