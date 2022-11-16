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

## Piecewise

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
for xt in Iterators.product(-1:0.2:1, -2:0.4:2, (1.0,))
    local x = collect(xt)
    local η = abs(xt[1] + 0.5)
    push!(nodes, PWAR.Node(x, η))
end

fig = figure()
ax = fig.add_subplot()

for node in nodes
    ax.plot(
        node.x[1], node.x[2],
        ls="none", marker=".", ms=5, c="tab:blue"
    )
end

ax.set_xlabel(L"x_1")
ax.set_ylabel(L"x_2")

for (xb, yb) in Iterators.product((-1, 1), (-2, 2))
    local x1rect = (xb, -0.5, -0.5, xb, xb)
    local x2rect = (yb, yb, -1, -1, yb)
    ax.plot(x1rect, x2rect, c="k")
end

ϵ = 0.01
θ = 0.1
BD = 100
γ = 0.01
inodes_all = BitSet(1:length(nodes))

xc = [-0.01, -0.01, 1.0]

obj, λus, λls = PWAR.infeasibility_certificate_LP(
    nodes, inodes_all, ϵ*(1 + γ), xc, 3, solver
)

obj, bins = PWAR.infeasibility_certificate_MILP(
    nodes, inodes_all, ϵ*(1 + γ), xc, 3, solver
)

inodes_cert_LP = PWAR.extract_infeasibility_certificate_LP(
    nodes, inodes_all, λus, λls, ϵ, θ, BD, 3, solver
)

inodes_cert_MILP = PWAR.extract_infeasibility_certificate_MILP(
    inodes_all, bins, θ
)

ax.plot(
    xc[1], xc[2], ls="none", marker=".", ms=20, c="tab:green"
)

for inode in inodes_cert_LP
    ax.plot(
        nodes[inode].x[1], nodes[inode].x[2],
        ls="none", marker="x", mew=2, ms=7.5, c="tab:red"
    )
end

for inode in inodes_cert_MILP
    ax.plot(
        nodes[inode].x[1], nodes[inode].x[2],
        ls="none", marker="+", mew=2, ms=7.5, c="tab:purple"
    )
end