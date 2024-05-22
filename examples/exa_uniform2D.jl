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

colors = collect(keys(matplotlib.colors.TABLEAU_COLORS))

NT = TK.Node{Vector{Float64},Vector{Float64}}
nodes = NT[]
for xt in Iterators.product(-1:0.1:1, -1:0.1:1, (1.0,))
    x = collect(xt)
    y = [max(0, abs(xt[1]) - 0.5, ) + min(0, 0.5 - abs(xt[2]))]
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
Atemp = [1 0 0; 0 1 0; -1 0 0; 0 -1 0]
inodes_list = @time TK.optimal_covering(nodes, ϵ, BD, γ, δ, 1, 3, Atemp,
                                          solver, solver, solver)

bs = TK.optimal_set_cover(length(nodes), inodes_list, solver)
inodes_list_opt = BitSet[]
for (i, b) in enumerate(bs)
    if round(Int, b) == 1
        push!(inodes_list_opt, inodes_list[i])
    end
end

for (q, inodes) in enumerate(inodes_list_opt)
    a = minimax_regression(nodes, inodes, BD, 1, 3, solver)
    # plot
    ctemp = TK.compute_lims(xlist, inodes, Atemp)
    ub, lb = ctemp[1:2], -ctemp[3:4]
    x1rect = (lb[1], ub[1], ub[1], lb[1], lb[1])
    x2rect = (lb[2], lb[2], ub[2], ub[2], lb[2])
    ax.plot(x1rect, x2rect, -1, c=colors[q])
    x1_ = (lb[1], ub[1])
    x2_ = (lb[2], ub[2])
    Xt_ = Iterators.product(x1_, x2_)
    X1_ = getindex.(Xt_, 1)
    X2_ = getindex.(Xt_, 2)
    Y_ = map(xt -> dot(a, (xt..., 1.0)), Xt_)
    ax.plot_surface(X1_, X2_, Y_, color=colors[q], lw=0, alpha=0.5)
end

ax.view_init(elev=14, azim=-70)
fig.savefig(
    "./examples/figures/exa_uniform2D.png", bbox_inches="tight", dpi=100
)

end # module