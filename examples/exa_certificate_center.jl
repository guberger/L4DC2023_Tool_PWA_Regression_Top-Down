module ExampleCertificate

using Random
using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

Random.seed!(0)

include("../src/PWARegression.jl")
PWAR = PWARegression

include("./utils.jl")

solver() = Model(Gurobi.Optimizer)

colors = ("tab:orange", "tab:green", "tab:purple")

bbox_figs = ((1.7, 0.75), (5.35, 3.65))

## Piecewise

NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
for xt in Iterators.product(0:0.05:1, 0:0.05:1, (1.0,))
    local x = collect(xt)
    local η = xt[1] > 0.25 && xt[2] > 0.25 ? 1.0 :
        xt[1] ≤ 0.25 && xt[2] > 0.25 ? 2.0 :
        xt[1] > 0.25 && xt[2] ≤ 0.25 ? 3.0 : 4.0
    push!(nodes, PWAR.Node(x, η))
end
xlist = map(node -> node.x, nodes)

fig = figure()
ax = fig.add_subplot()

x1s = [node.x[1] for node in nodes]
x2s = [node.x[2] for node in nodes]
ηs = [node.η[1] for node in nodes]
h = ax.scatter(x1s, x2s, c=ηs, s=5)
fig.colorbar(h)

N = 3
xlist = map(node -> node.x, nodes)
inodes = BitSet(1:length(nodes))
ϵ = 0.01
meth = 4

xc = PWAR.compute_center(xlist, inodes, N)
if meth == 1
    val, λus, λls = PWAR.infeasibility_certificate_LP(
        nodes, inodes, ϵ, xc, N, solver
    )
elseif meth == 2
    val, bins = PWAR.infeasibility_certificate_MILP(
        nodes, inodes, ϵ, xc, N, solver
    )
elseif meth == 3
    μ = 0.01
    val, bins = PWAR.infeasibility_certificate_MILP_prox(
        nodes, inodes, ϵ, xc, μ, N, solver
    )
elseif meth == 4
    ρ = 0.2
    val, bins = PWAR.infeasibility_certificate_MILP_radius(
        nodes, inodes, ϵ, xc, ρ, N, solver
    )
end
display(val)
if meth == 1
    θ = 0.5
    BD = 100
    inodes_cert = PWAR.extract_infeasibility_certificate_LP(
        nodes, inodes, λus, λls, ϵ, θ, BD, N, solver
    )
elseif meth ∈ (2, 3, 4)
    θ = 0.5
    inodes_cert = PWAR.extract_infeasibility_certificate_MILP(inodes, bins, θ)
end
for inode in inodes_cert
    node = nodes[inode]
    ax.plot(node.x[1], node.x[2], marker="x", ms=10, c="red")
end

end # module