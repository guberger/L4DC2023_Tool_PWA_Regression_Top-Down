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
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>true, "TimeLimit" => 1000
))

colors = collect(keys(matplotlib.colors.TABLEAU_COLORS))

str = readlines(string(@__DIR__, "/exa_acrobot_data.txt"))
NT = PWAR.Node{Vector{Float64},Float64}
nodes = NT[]
for ln in str
    local ln = replace(ln, r"[\(\),]"=>"")
    local words = split(ln)
    @assert length(words) == 4
    local x = parse.(Float64, [words[1], words[2], "1"])
    local η = parse(Float64, words[3])
    push!(nodes, PWAR.Node(x, η))
end
shuffle!(nodes)
# 50: 17 secs
# 51: 350 secs
# 52: > 1000 secs
nodes = nodes[1:52]
κ = maximum(node -> norm(node.x, 1), nodes)
display(κ)

# MILP

ϵ = 0.01
BD = 1e5
M = 9
Γ = 2*κ*BD
inodes_all = BitSet(eachindex(nodes))

as, bins = switched_bounded_regression(
    nodes, inodes_all, M, ϵ, BD, 3, Γ, solver
)

inodes_list_sw = [BitSet() for q in 1:M]
for (inode, bin) in bins
    for q in 1:M
        if round(Int, bin[q]) == 1
            push!(inodes_list_sw[q], inode)
        end
    end
end

fig = figure()
ax = fig.add_subplot()

for (inode, node) in enumerate(nodes)
    q = findfirst(q -> round(Int, bins[inode][q]) == 1, 1:M)
    ax.plot(
        node.x[1], node.x[2],
        ls="none", marker=".", ms=15, c=colors[q]
    )
end

ax.set_xlabel(L"x")
ax.set_ylabel(L"y")