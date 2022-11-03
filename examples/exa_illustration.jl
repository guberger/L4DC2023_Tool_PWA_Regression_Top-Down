using Random
using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

Random.seed!(0)

include("../src/PWARegression.jl")
PWAR = PWARegression

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

colors = collect(keys(matplotlib.colors.TABLEAU_COLORS))

## Piecewise

fig = figure()
ax = fig.add_subplot(projection="3d")
ax.tick_params(axis="both", which="major", labelsize=15)

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
aref = [1, 2, 1]
ν = 0.5
for xt in Iterators.product(0:0.1:1, 0:0.2:2, (1.0,))
    x = collect(xt)
    local η = xt[1] > 0.5 && xt[2] > 1.0 ? dot(aref, x) :
        xt[1] ≤ 0.5 && xt[2] > 1.0 ? ν :
        xt[1] > 0.5 && xt[2] ≤ 1.0 ? -ν :
        (rand() - 0.5)*ν
    push!(nodes, PWAR.Node(x, η))
end

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
    bbox_inches=matplotlib.transforms.Bbox(((1.7, 0.75), (5.35, 3.65))),
    dpi=100
)

σ = 0.05
β = 1e-6
ϵ = 0.76*ν
BD = 100
γ = 0.01
δ = 1e-5
inodes_list = PWAR.optimal_covering(
    nodes, ϵ, BD, σ, β, γ, δ, 3, solver, solver, solver
)

bs = PWAR.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end
inodes_list_opt

fig = figure()
ax = fig.add_subplot(projection="3d")
ax.tick_params(axis="both", which="major", labelsize=15)

for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η,
        ls="none", marker=".", ms=5, c="tab:blue"
    )
end

nplot = 5

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
    local x1_ = range(lb[1], ub[1], length=nplot)
    local x2_ = range(lb[2], ub[2], length=nplot)
    local Xt_ = Iterators.product(x1_, x2_)
    local X1_ = getindex.(Xt_, 1)
    local X2_ = getindex.(Xt_, 2)
    local Y_ = map(xt -> dot(a_opt, (xt..., 1.0)), Xt_)
    ax.plot_surface(X1_, X2_, Y_)
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
    bbox_inches=matplotlib.transforms.Bbox(((1.7, 0.75), (5.35, 3.65))),
    dpi=100
)

## Sine

fig = figure()
ax = fig.add_subplot(projection="3d")
ax.tick_params(axis="both", which="major", labelsize=15)

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
for xt in Iterators.product(0:0.05:1, 0:0.2:2, (1.0,))
    x = collect(xt)
    local η = sin(xt[1]*7) + 2*xt[3]
    push!(nodes, PWAR.Node(x, η))
end

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
    bbox_inches=matplotlib.transforms.Bbox(((1.7, 0.75), (5.35, 3.65))),
    dpi=100
)

σ = 0.05
β = 1e-6
ϵ = 0.15
BD = 100
γ = 0.01
δ = 1e-5
inodes_list = PWAR.optimal_covering(
    nodes, ϵ, BD, σ, β, γ, δ, 3, solver, solver, solver
)

bs = PWAR.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end
inodes_list_opt

fig = figure()
ax = fig.add_subplot(projection="3d")
ax.tick_params(axis="both", which="major", labelsize=15)

for node in nodes
    ax.plot(
        node.x[1], node.x[2], node.η,
        ls="none", marker=".", ms=5, c="tab:blue"
    )
end

nplot = 5

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
    local x1_ = range(lb[1], ub[1], length=nplot)
    local x2_ = range(lb[2], ub[2], length=nplot)
    local Xt_ = Iterators.product(x1_, x2_)
    local X1_ = getindex.(Xt_, 1)
    local X2_ = getindex.(Xt_, 2)
    local Y_ = map(xt -> dot(a_opt, (xt..., 1.0)), Xt_)
    ax.plot_surface(X1_, X2_, Y_)
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
    bbox_inches=matplotlib.transforms.Bbox(((1.7, 0.75), (5.35, 3.65))),
    dpi=100
)