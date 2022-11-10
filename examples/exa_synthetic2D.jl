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

NT = PWAR.Node{Vector{Float64},Float64}
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
    push!(nodes, PWAR.Node(x, η))
end

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
inodes_list = PWAR.optimal_covering(
    nodes, ϵ, BD, γ, δ, 3, solver, solver, solver
)

bs = PWAR.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end

colors = collect(keys(matplotlib.colors.TABLEAU_COLORS))

for (k, inodes) in enumerate(inodes_list)
    local c = colors[mod(k - 1, length(colors)) + 1]
    local lb = fill(Inf, 2)
    local ub = fill(-Inf, 2)
    for inode in inodes
        local x = nodes[inode].x
        for k = 1:2
            if x[k] < lb[k]
                lb[k] = x[k]
            end
            if x[k] > ub[k]
                ub[k] = x[k]
            end
        end
    end
    local x1rect = (lb[1], ub[1], ub[1], lb[1], lb[1])
    local x2rect = (lb[2], lb[2], ub[2], ub[2], lb[2])
    ax.plot(x1rect, x2rect, -1, c=c)
end

# Plot planes

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

ax.view_init(elev=14, azim=-70)
fig.savefig(
    "./examples/figures/exa_synthetic2D.png", bbox_inches="tight", dpi=100
)