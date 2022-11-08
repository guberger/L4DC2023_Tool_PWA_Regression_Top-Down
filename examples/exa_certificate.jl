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
ax = fig.add_subplot()

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
for xt in Iterators.product(-1:0.05:1, -2:0.1:2, (1.0,))
    local x = collect(xt)
    local η = abs(xt[1] + 0.5)
    push!(nodes, PWAR.Node(x, η))
end

for node in nodes
    ax.plot(
        node.x[1], node.x[2],
        ls="none", marker=".", ms=5, c="tab:blue"
    )
end

ax.set_xlabel(L"x_1")
ax.set_ylabel(L"x_2")

for (k, (xb, yb)) in enumerate(Iterators.product((-1, 1), (-2, 2)))
    local x1rect = (xb, -0.5, -0.5, xb, xb)
    local x2rect = (yb, yb, -1, -1, yb)
    ax.plot(x1rect, x2rect, c="k")
end

ϵ = 0.01
BD = 100
γ = 0.01
inodes_all = BitSet(1:length(nodes))

inodes_cert = PWAR.find_infeasibility_certificate(
    nodes, inodes_all, ϵ, ϵ*(1 + γ), BD, 3, solver, solver
)

for inode in inodes_cert
    ax.plot(
        nodes[inode].x[1], nodes[inode].x[2],
        ls="none", marker=".", ms=7.5, c="tab:red"
    )
end