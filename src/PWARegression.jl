module PWARegression

using LinearAlgebra
using JuMP
using DataStructures

struct Node{VTX<:AbstractVector,TY}
    x::VTX
    η::TY
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

# Residuals

function LInf_residual(subgraph::Subgraph, BD, N, solver)
    model = solver()
    a = @variable(model, [1:N], lower_bound=-BD, upper_bound=BD)
    r = @variable(model, lower_bound=-1)
    for inode in subgraph.inodes
        node = subgraph.graph.nodes[inode]
        @constraint(model, dot(a, node.x) ≤ node.η + r)
        @constraint(model, dot(a, node.x) ≥ node.η - r)
    end
    @objective(model, Min, r)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return value(r)
end

function local_L2_residual(subgraph::Subgraph, xc, σ, β, N)
    nnode = length(subgraph)
    x_ = zeros(nnode + N, N)
    η_ = zeros(nnode + N)
    for (t, inode) in enumerate(subgraph.inodes)
        ρ = norm(subgraph.graph.nodes[inode].x - xc)
        ω = exp(-((ρ/σ)^2)/2)
        for k in 1:N
            x_[t, k] = subgraph.graph.nodes[inode].x[k]*ω
        end
        η_[t] = subgraph.graph.nodes[inode].η*ω
    end
    for k = 1:N
        x_[nnode + k, k] = β
    end
    a = x_ \ η_
    return norm(x_*a - η_)
    # try
    #     a = x_ \ η_
    #     return norm(x_*a - η_)
    # catch
    #     @warn("Catching SingularException")
    #     a = (x_'*x_ + 1e-6*I) \ (x_'*η_)
    #     return norm(x_*a - η_)
    # end
end

function max_local_L2_residual(subgraph::Subgraph, σ, ρ, N)
    res_max = -Inf
    inode_opt = 0
    for inode in subgraph.inodes
        xc = subgraph.graph.nodes[inode].x
        res = local_L2_residual(subgraph, xc, σ, ρ, N)
        if res > res_max
            res_max = res
            inode_opt = inode
        end
    end
    return res_max, inode_opt
end

# Certificate

function infeasibility_certificate(subgraph::Subgraph, ϵ, xc, solver)
    model = solver()
    λus = Dict(
        inode => @variable(model, lower_bound=0) for inode in subgraph.inodes
    ) # upper bound : a*x ≤ y + ϵ
    λls = Dict(
        inode => @variable(model, lower_bound=0) for inode in subgraph.inodes
    ) # lower bound : -a*x ≤ -y + ϵ
    @constraint(model, sum(
        (λus[inode] + λls[inode]) for inode in subgraph.inodes
    ) == 1)
    @constraint(model, sum(
        (λus[inode] - λls[inode])*subgraph.graph.nodes[inode].x
        for inode in subgraph.inodes
    ) .== 0)
    @constraint(model, sum(
        (λus[inode] - λls[inode])*subgraph.graph.nodes[inode].η
        for inode in subgraph.inodes
    ) ≤ -ϵ)
    @objective(model, Min, sum(
        (λus[inode] + λls[inode])*norm(subgraph.graph.nodes[inode].x - xc)^2
        for inode in subgraph.inodes
    ))
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return (
        Dict(inode => value(λ) for (inode, λ) in λus),
        Dict(inode => value(λ) for (inode, λ) in λls),
        objective_value(model)
    )
end

# Maximal regions

function _update_bounds!(lb, ub, x::AbstractVector, N)
    for k = 1:N
        if x[k] < lb[k]
            lb[k] = x[k]
        end
        if x[k] > ub[k]
            ub[k] = x[k]
        end
    end
end

function _update_bounds!(lb, ub, subgraph::Subgraph, N)
    for inode in subgraph.inodes
        _update_bounds!(lb, ub, subgraph.graph.nodes[inode].x, N)
    end
end

"""
    all_maximal_regions(graph, ϵ, BD, σ, β, γ, θ, δ, N, solver)

`ϵ`: max error;
`BD`: max value of system parameters;
`γ`: 'gap' for max error;
`σ`: radius for local residual;
`β`: regularization parameter;
`θ`: threshold for certificate multipliers;
`δ`: distance for certificate;
`N`: variable dimension;
`solver`: LP solver.
"""
function all_maximal_regions(graph::Graph, ϵ, BD, σ, β, γ, θ, δ, N, solver)
    subgraph = Subgraph(graph, BitSet(1:length(graph)))
    subgraphs_remain = [subgraph]
    subgraphs_found = typeof(subgraph)[]
    iter = 0
    lb = fill(NaN, N)
    ub = fill(NaN, N)
    rect_list = Tuple{Vector{Float64},Vector{Float64}}[]
    while !isempty(subgraphs_remain)
        iter += 1
        println(
            "iter: ", iter,
            " - rects remain: ", length(subgraphs_remain),
            " - rects found: ", length(subgraphs_found)
        )
        subgraph = pop!(subgraphs_remain)
        isempty(subgraph.inodes) && continue
        # compute lims of subgraph
        fill!(lb, Inf)
        fill!(ub, -Inf)
        _update_bounds!(lb, ub, subgraph, N)
        any(k -> lb[k] > ub[k], 1:N) && continue
        # check if contained in another rect (+`δ`)
        is_sub = false
        for (lbr, ubr) in rect_list
            is_sub = all(
                k -> lbr[k] - δ < lb[k] && ub[k] < ubr[k] + δ, 1:N
            )
            is_sub && break
        end
        is_sub && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph and associated rectangle to "found"
        res = LInf_residual(subgraph, BD, N, solver)
        if res < ϵ*(1 + γ)
            push!(subgraphs_found, subgraph)
            lbr, ubr = fill(Inf, N), fill(-Inf, N)
            _update_bounds!(lbr, ubr, subgraph, N)
            push!(rect_list, (lbr, ubr))
            continue
        end
        # if no, find an infeasibility certificate
        inode = max_local_L2_residual(subgraph, σ, β, N)[2]
        @assert !iszero(inode)
        xc = subgraph.graph.nodes[inode].x
        λus, λls, obj = infeasibility_certificate(subgraph, ϵ, xc, solver)
        fill!(lb, Inf)
        fill!(ub, -Inf)
        for inode in subgraph.inodes
            max(λus[inode], λls[inode]) < θ && continue
            _update_bounds!(lb, ub, subgraph.graph.nodes[inode].x, N)
        end
        # add maximal sub-rectangles breaking the infeasibility certificate
        subgraphs_left = [Subgraph(subgraph.graph, BitSet()) for k = 1:N]
        subgraphs_right = [Subgraph(subgraph.graph, BitSet()) for k = 1:N]
        for inode in subgraph.inodes
            for k = 1:N
                if subgraph.graph.nodes[inode].x[k] < ub[k] - δ
                    add_inode!(subgraphs_left[k], inode)
                end
                if subgraph.graph.nodes[inode].x[k] > lb[k] + δ
                    add_inode!(subgraphs_right[k], inode)
                end
            end
        end
        append!(subgraphs_remain, subgraphs_left, subgraphs_right)
    end
    # post-processing
    println("Post-processing")
    subgraphs_final = eltype(subgraphs_remain)[]
    for subgraph in subgraphs_found
        fill!(lb, Inf)
        fill!(ub, -Inf)
        _update_bounds!(lb, ub, subgraph, N)
        any(k -> lb[k] > ub[k], 1:N) && continue
        is_subneq = false
        for (lbr, ubr) in rect_list
            any(
                k -> lbr[k] - δ > lb[k] || ub[k] > ubr[k] + δ, 1:N
            ) && continue
            # here is_sub, i.e., subgraph ⊆ rect
            is_subneq = any(
                k -> lbr[k] + δ < lb[k] || ub[k] < ubr[k] - δ, 1:N
            )
            is_subneq && break
        end
        is_subneq && continue
        push!(subgraphs_final, subgraph)
    end
    println("rect founds: ", length(subgraphs_final))
    return subgraphs_final
end

"""
    greedy_maximal_regions(graph, ϵ, BD, σ, β, γ, θ, δ, N, solver)

`ϵ`: max error;
`BD`: max value of system parameters;
`γ`: 'gap' for max error;
`σ`: radius for local residual;
`β`: regularization parameter;
`θ`: threshold for certificate multipliers;
`δ`: distance for certificate;
`N`: variable dimension;
`solver`: LP solver.
"""
function greedy_maximal_regions(graph::Graph, ϵ, BD, σ, β, γ, θ, δ, N, solver)
    subgraph = Subgraph(graph, BitSet(1:length(graph)))
    subgraphs_remain = PriorityQueue(subgraph => -length(graph))
    subgraphs_found = typeof(subgraph)[]
    iter = 0
    lb = fill(NaN, N)
    ub = fill(NaN, N)
    while !isempty(subgraphs_remain)
        iter += 1
        println(
            "iter: ", iter,
            " - rects remain: ", length(subgraphs_remain),
            " - rects found: ", length(subgraphs_found)
        )
        subgraph = dequeue!(subgraphs_remain)
        isempty(subgraph.inodes) && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph and remove nodes for remaining subgraphs
        res = LInf_residual(subgraph, BD, N, solver)
        if res < ϵ*(1 + γ)
            push!(subgraphs_found, subgraph)
            for subgraph_remain in keys(subgraphs_remain)
                setdiff!(subgraph_remain.inodes, subgraph.inodes)
                subgraphs_remain[subgraph_remain] = -length(subgraph_remain)
            end
            continue
        end
        # if no, find an infeasibility certificate
        inode = max_local_L2_residual(subgraph, σ, β, N)[2]
        @assert !iszero(inode)
        xc = subgraph.graph.nodes[inode].x
        λus, λls, obj = infeasibility_certificate(subgraph, ϵ, xc, solver)
        fill!(lb, Inf)
        fill!(ub, -Inf)
        for inode in subgraph.inodes
            max(λus[inode], λls[inode]) < θ && continue
            _update_bounds!(lb, ub, subgraph.graph.nodes[inode].x, N)
        end
        # add maximal sub-rectangles breaking the infeasibility certificate
        subgraphs_left = [Subgraph(subgraph.graph, BitSet()) for k = 1:N]
        subgraphs_right = [Subgraph(subgraph.graph, BitSet()) for k = 1:N]
        for inode in subgraph.inodes
            for k = 1:N
                if subgraph.graph.nodes[inode].x[k] < ub[k] - δ
                    add_inode!(subgraphs_left[k], inode)
                end
                if subgraph.graph.nodes[inode].x[k] > lb[k] + δ
                    add_inode!(subgraphs_right[k], inode)
                end
            end
        end
        for subgraph in Iterators.flatten((subgraphs_left, subgraphs_right))
            enqueue!(subgraphs_remain, subgraph => -length(subgraph))
        end
    end
    # Finished
    println("rect founds: ", length(subgraphs_found))
    return subgraphs_found
end

end # module
