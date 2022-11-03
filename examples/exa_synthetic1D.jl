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

# colors = collect(keys(matplotlib.colors.TABLEAU_COLORS))
colors = ("tab:orange", "tab:green", "tab:purple")

F(x) = atan(10*x)*exp(-abs(x))

fig = figure(figsize=(4.5, 3))
ax = fig.add_subplot()

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
push!(nodes, PWAR.Node([0.0, 1.0], 0.0))
for xt in Iterators.product(0.2:0.1:1, (1.0,))
    for sg in (-1, 1)
        x = collect(xt.*(sg, 1))
        local η = F(x[1])
        push!(nodes, PWAR.Node(x, η))
    end
end

xplot = range(-1, 1, length=100)
yplot = map(x -> F(x), xplot)
ax.plot(xplot, yplot, ls="dashed", lw=3, c="k")

for node in nodes
    ax.plot(node.x[1], node.η, ls="none", marker=".", ms=15, c="tab:blue")
end

σ = 0.05
β = 1e-6
ϵ = 0.1
BD = 100
γ = 0.01
δ = 1e-5
inodes_list = PWAR.optimal_covering(
    nodes, ϵ, BD, σ, β, γ, δ, 2, solver, solver, solver
)

bs = PWAR.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end
inodes_list_opt

nplot = 5

for (s, inodes) in enumerate(inodes_list_opt)
    # optim parameters
    local model = solver()
    local a = @variable(model, [1:2], lower_bound=-BD, upper_bound=BD)
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
    local lb = Inf
    local ub = -Inf
    for inode in inodes
        node = nodes[inode]
        if node.x[1] < lb
            lb = node.x[1]
        end
        if node.x[1] > ub
            ub = node.x[1]
        end
    end
    local x1_ = range(lb, ub, length=nplot)
    local Xt_ = Iterators.product(x1_)
    local X1_ = getindex.(Xt_, 1)
    local Y_ = map(xt -> dot(a_opt, (xt..., 1.0)), Xt_)
    ax.plot(X1_, Y_, lw=3, c=colors[s])
end

ax.set_xlabel(L"x")
ax.set_ylabel(L"y")

fig.savefig(
    "./examples/figures/exa_synthetic1D.png",
    bbox_inches="tight", dpi=100
)