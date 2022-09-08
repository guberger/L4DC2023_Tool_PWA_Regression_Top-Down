module PWARegression

using LinearAlgebra
using JuMP

struct Node{VTX<:AbstractVector,VTY<:AbstractVector}
    x::VTX
    y::VTY
end

struct Graph{NT<:Node}
    nodes::Vector{NT}
end

add_node!(graph::Graph, node::Node) = push!(graph.nodes, node)
Base.length(graph::Graph) = length(graph.nodes)

struct Subgraph{GT<:Graph,ST<:AbstractSet{Int}}
    graph::GT
    inodes::ST
end

add_inode!(subgraph::Subgraph, inode::Int) = push!(subgraph.inodes, inode)
Base.length(subgraph::Subgraph) = length(subgraph.inodes)

include("separator.jl")
include("verifier.jl")

end # module
