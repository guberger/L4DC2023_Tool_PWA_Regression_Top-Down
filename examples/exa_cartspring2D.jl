module Example

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../src/main.jl")
TK = ToolKit

include("./utils.jl")

const GUROBI_ENV = Gurobi.Env()
gurobi_() = Gurobi.Optimizer(GUROBI_ENV)
solver() = Model(optimizer_with_attributes(gurobi_, MOI.Silent()=>true))

colors = repeat(collect(keys(matplotlib.colors.TABLEAU_COLORS)), 100)

NT = TK.Node{Vector{Float64},Vector{Float64}}
nodes = NT[]
L = 5
k = 1
d = L/10
subs = range(0, L, length=20)
for xt in Iterators.product(subs, subs, (1.0,))
    x = collect(xt)
    u1l = max(0, k * (d - x[1]))
    δ = x[2] - x[1] - 2 * d
    u1r = min(0, +k * δ)
    u2l = max(0, -k * δ)
    u2r = min(0, k * (x[2] - d))
    y = [u1l + u1r]
    push!(nodes, TK.Node(x, y))
end
xlist = map(node -> node.x, nodes)

fig = figure()
ax = fig.add_subplot(projection="3d")
for node in nodes
    ax.plot(node.x[1], node.x[2], node.y[1],
            ls="none", marker=".", ms=5, c="tab:blue")
end

# regions

ϵ = 0.01
BD = 100
γ = 0.01
δ = 1e-5
Atemp = [1 0 0; 0 -1 0; -1 0 0; 0 1 0; 1 -1 0; -1 1 0; 1 1 0; -1 -1 0]
Atemp = [1 0 0; -1 0 0; 1 -1 0; -1 1 0]
# Atemp = [1 0 0; -1 0 0; 0 -1 0; 0 1 0]
inodes_list = @time TK.optimal_covering(nodes, ϵ, BD, γ, δ, 1, 3, Atemp,
                                          solver, solver, solver)

bs = TK.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end

fig = figure()
ax = fig.add_subplot()

A_list = [zeros(1, 3) for _ in inodes_list_opt]
ctemp_list = [zeros(size(Atemp, 1)) for _ in inodes_list_opt]
for (q, inodes) in enumerate(inodes_list_opt)
    A = minimax_regression(nodes, inodes, BD, 1, 3, solver)
    ctemp = TK.compute_lims(xlist, inodes, Atemp)
    A_list[q] = A
    ctemp_list[q] = ctemp
end

function model_val(x)
    q_opt = -1
    dist_opt = Inf
    for q in eachindex(inodes_list_opt)
        c = Atemp * x
        dist = maximum(c - ctemp_list[q])
        if dist < dist_opt
            dist_opt = dist
            q_opt = q
        end
    end
    return A_list[q_opt] * x
end

model_dist(x, q) = maximum(Atemp * x - ctemp_list[q])

subs = range(0, L, length=500)
Xt_ = Iterators.product(subs, subs)
X1_ = getindex.(Xt_, 1)
X2_ = getindex.(Xt_, 2)
Y_ = map(xt -> model_val([xt..., 1.0])[1], Xt_)
h = ax.contourf(X1_, X2_, Y_, levels=20, cmap=matplotlib.cm.coolwarm)
for q in eachindex(inodes_list_opt)
    Y_ = map(xt -> model_dist([xt..., 1.0], q), Xt_)
    ax.contour(X1_, X2_, Y_, levels=[0.1], colors="black")
end
fig.colorbar(h)


# for (q, inodes) in enumerate(inodes_list_opt)
#     a = minimax_regression(nodes, inodes, BD, 1, 3, solver)
#     # plot
#     Atemp = [1 0 0; 0 1 0; -1 0 0; 0 -1 0]
#     ctemp = TK.compute_lims(xlist, inodes, Atemp)
#     ub, lb = ctemp[1:2], -ctemp[3:4]
#     x1rect = (lb[1], ub[1], ub[1], lb[1], lb[1])
#     x2rect = (lb[2], lb[2], ub[2], ub[2], lb[2])
#     ax.plot(x1rect, x2rect, -1, c=colors[q])
#     x1_ = (lb[1], ub[1])
#     x2_ = (lb[2], ub[2])
#     Xt_ = Iterators.product(x1_, x2_)
#     X1_ = getindex.(Xt_, 1)
#     X2_ = getindex.(Xt_, 2)
#     Y_ = map(xt -> dot(a, (xt..., 1.0)), Xt_)
#     ax.plot_surface(X1_, X2_, Y_, color=colors[q], lw=0, alpha=0.5)
# end

fig.savefig(
    "./examples/figures/exa_cartspring.png", bbox_inches="tight", dpi=100
)

end # module