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

# MILP set cover

function optimal_set_cover(N, sets, solver)
    model = solver()
    bs = [@variable(model, binary=true) for k in eachindex(sets)]
    cons = [AffExpr(0.0) for i = 1:N]
    for (k, set) in enumerate(sets)
        for i in set
            cons[i] += bs[k]
        end
    end
    for con in cons
        @constraint(model, con ≥ 1)
    end
    @objective(model, Min, sum(bs))
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return value.(bs)    
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

"""
    maximal_regions(graph, ϵ, BD, σ, β, γ, θ, δ, N, solvers...)

`ϵ`: max error;
`BD`: max value of system parameters;
`γ`: 'gap' for max error;
`σ`: radius for local residual;
`β`: regularization parameter;
`θ`: threshold for certificate multipliers;
`δ`: distance for certificate;
`N`: variable dimension;
`solver_F`: LP solver for feasibility
`solver_I`: LP solver for infeasibility
"""
function maximal_regions(
        graph::Graph, ϵ, BD, σ, β, γ, θ, δ, N, solver_F, solver_I
    )
    subgraph = Subgraph(graph, BitSet(1:length(graph)))
    subgraphs_remain = PriorityQueue(subgraph => -length(graph))
    subgraphs_found = typeof(subgraph)[]
    subgraphs_bis = typeof(subgraph)[]
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
        # check if contained in a found subgraph
        # if yes, discard subgraph because not maximal
        is_sub = false
        for subgraph_found in subgraphs_found
            is_sub = subgraph.inodes ⊆ subgraph_found.inodes
            is_sub && break
        end
        is_sub && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph to "found" and remove all previously found
        # subgraphs that are strictly contained in subgraph
        res = LInf_residual(subgraph, BD, N, solver_F)
        if res < ϵ*(1 + γ)
            empty!(subgraphs_bis)
            for subgraph_found in subgraphs_found
                subgraph_found.inodes ⊊ subgraph.inodes && continue
                push!(subgraphs_bis, subgraph_found)
            end
            subgraphs_found, subgraphs_bis = subgraphs_bis, subgraphs_found
            push!(subgraphs_found, subgraph)
            continue
        end
        # if no, find an infeasibility certificate
        inode = max_local_L2_residual(subgraph, σ, β, N)[2]
        @assert !iszero(inode)
        xc = subgraph.graph.nodes[inode].x
        λus, λls, obj = infeasibility_certificate(subgraph, ϵ, xc, solver_I)
        # compute lims of infeasibility certificate
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

"""
    greedy_covering(graph, ϵ, BD, σ, β, γ, θ, δ, N, solvers...)

`ϵ`: max error;
`BD`: max value of system parameters;
`γ`: 'gap' for max error;
`σ`: radius for local residual;
`β`: regularization parameter;
`θ`: threshold for certificate multipliers;
`δ`: distance for certificate;
`N`: variable dimension;
`solver_F`: LP solver for feasibility
`solver_I`: LP solver for infeasibility
"""
function greedy_covering(
        graph::Graph, ϵ, BD, σ, β, γ, θ, δ, N, solver_F, solver_I
    )
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
        res = LInf_residual(subgraph, BD, N, solver_F)
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
        λus, λls, obj = infeasibility_certificate(subgraph, ϵ, xc, solver_I)
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

"""
    optimal_covering(graph, ϵ, BD, σ, β, γ, θ, δ, N, solvers...)

`ϵ`: max error;
`BD`: max value of system parameters;
`γ`: 'gap' for max error;
`σ`: radius for local residual;
`β`: regularization parameter;
`θ`: threshold for certificate multipliers;
`δ`: distance for certificate;
`N`: variable dimension;
`solver_F`: LP solver for feasibility
`solver_I`: LP solver for infeasibility
`solver_C`: MILP solver for cover
"""
function optimal_covering(
        graph::Graph, ϵ, BD, σ, β, γ, θ, δ, N, solver_F, solver_I, solver_C
    )
    nnode = length(graph)
    subgraph = Subgraph(graph, BitSet(1:length(graph)))
    subgraphs_remain = PriorityQueue(subgraph => -length(graph))
    subgraphs_found = typeof(subgraph)[]
    subgraphs_bis = typeof(subgraph)[]
    iter = 0
    lb = fill(NaN, N)
    ub = fill(NaN, N)
    inodes_list = BitSet[]
    inodes_found = BitSet()
    ncov_lower::Int = 1
    ncov_upper::Int = nnode
    all_nodes = BitSet(1:nnode)
    while !isempty(subgraphs_remain) && ncov_lower < ncov_upper
        iter += 1
        println(
            "iter: ", iter,
            " - rects remain: ", length(subgraphs_remain),
            " - rects found: ", length(subgraphs_found),
            " - ncov_lower: ", ncov_lower,
            " - ncov_upper: ", ncov_upper
        )
        subgraph = dequeue!(subgraphs_remain)
        isempty(subgraph.inodes) && continue
        # check if contained in a found subgraph
        # if yes, discard subgraph because not maximal
        is_sub = false
        for subgraph_found in subgraphs_found
            is_sub = subgraph.inodes ⊆ subgraph_found.inodes
            is_sub && break
        end
        is_sub && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph to "found" and remove all previously found
        # subgraphs that are strictly contained in subgraph
        # finally, update upper bound on covering number
        res = LInf_residual(subgraph, BD, N, solver_F)
        if res < ϵ*(1 + γ)
            empty!(subgraphs_bis)
            for subgraph_found in subgraphs_found
                subgraph_found.inodes ⊊ subgraph.inodes && continue
                push!(subgraphs_bis, subgraph_found)
            end
            subgraphs_found, subgraphs_bis = subgraphs_bis, subgraphs_found
            push!(subgraphs_found, subgraph)
            # update upper bound on covering number
            empty!(inodes_list)
            empty!(inodes_found)
            for subgraph in subgraphs_found
                push!(inodes_list, subgraph.inodes)
                union!(inodes_found, subgraph.inodes)
            end
            !(all_nodes == inodes_found) && continue
            bs = optimal_set_cover(nnode, inodes_list, solver_C)
            ncov_upper = sum(b -> round(Int, b), bs)
            continue
        end
        # if no, find an infeasibility certificate
        inode = max_local_L2_residual(subgraph, σ, β, N)[2]
        @assert !iszero(inode)
        xc = subgraph.graph.nodes[inode].x
        λus, λls, obj = infeasibility_certificate(subgraph, ϵ, xc, solver_I)
        # compute lims of infeasibility certificate
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
        # update lower_bound on covering number
        empty!(inodes_list)
        for subgraph in Iterators.flatten((subgraphs_found, keys(subgraphs_remain)))
            push!(inodes_list, subgraph.inodes)
        end
        bs = optimal_set_cover(nnode, inodes_list, solver_C)
        ncov_lower = sum(b -> round(Int, b), bs)
    end
    # Finished
    println("rect founds: ", length(subgraphs_found))
    return subgraphs_found
end

end # module
