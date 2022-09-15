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

# Residuals

function LInf_residual(graph, inodes, BD, N, solver)
    model = solver()
    a = @variable(model, [1:N], lower_bound=-BD, upper_bound=BD)
    r = @variable(model, lower_bound=-1)
    for inode in inodes
        node = graph.nodes[inode]
        @constraint(model, dot(a, node.x) ≤ node.η + r)
        @constraint(model, dot(a, node.x) ≥ node.η - r)
    end
    @objective(model, Min, r)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return value(r)
end

function local_L2_residual(graph, inodes, xc, σ, β, N)
    nnode = length(inodes)
    x_ = zeros(nnode + N, N)
    η_ = zeros(nnode + N)
    for (t, inode) in enumerate(inodes)
        ρ = norm(graph.nodes[inode].x - xc)
        ω = exp(-((ρ/σ)^2)/2)
        for k in 1:N
            x_[t, k] = graph.nodes[inode].x[k]*ω
        end
        η_[t] = graph.nodes[inode].η*ω
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

function max_local_L2_residual(graph, inodes, σ, ρ, N)
    res_max = -Inf
    inode_opt = 0
    for inode in inodes
        xc = graph.nodes[inode].x
        res = local_L2_residual(graph, inodes, xc, σ, ρ, N)
        if res > res_max
            res_max = res
            inode_opt = inode
        end
    end
    return res_max, inode_opt
end

# Certificate

function infeasibility_certificate(graph, inodes, ϵ, xc, solver)
    model = solver()
    λus = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # upper bound : a*x ≤ y + ϵ
    λls = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # lower bound : -a*x ≤ -y + ϵ
    @constraint(model, sum(
        (λus[inode] + λls[inode]) for inode in inodes
    ) == 1)
    @constraint(model, sum(
        (λus[inode] - λls[inode])*graph.nodes[inode].x
        for inode in inodes
    ) .== 0)
    @constraint(model, sum(
        (λus[inode] - λls[inode])*graph.nodes[inode].η
        for inode in inodes
    ) ≤ -ϵ)
    @objective(model, Min, sum(
        (λus[inode] + λls[inode])*norm(graph.nodes[inode].x - xc)^2
        for inode in inodes
    ))
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return (
        Dict(inode => value(λ) for (inode, λ) in λus),
        Dict(inode => value(λ) for (inode, λ) in λls)
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

struct KeyPQ
    inodes::BitSet
end

"""
    maximal_regions(graph, ϵ, BD, σ, β, γ, δ, N, solvers...)

`ϵ`: max error;
`BD`: max value of system parameters;
`γ`: 'gap' for max error;
`σ`: radius for local residual;
`β`: regularization parameter;
`δ`: distance for certificate;
`N`: variable dimension;
`solver_F`: LP solver for feasibility
`solver_I`: LP solver for infeasibility
"""
function maximal_regions(
        graph::Graph, ϵ, BD, σ, β, γ, δ, N, solver_F, solver_I
    )
    inodes = BitSet(1:length(graph))
    inodes_list_remain = PriorityQueue(KeyPQ(inodes) => -length(inodes))
    inodes_list_found = BitSet[]
    inodes_list_bis = BitSet[]
    inodes_cert = BitSet()
    iter = 0
    lb = fill(NaN, N)
    ub = fill(NaN, N)
    while !isempty(inodes_list_remain)
        iter += 1
        println(
            "iter: ", iter,
            " - rects remain: ", length(inodes_list_remain),
            " - rects found: ", length(inodes_list_found)
        )
        inodes = dequeue!(inodes_list_remain).inodes
        isempty(inodes) && continue
        # check if contained in a found subgraph
        # if yes, discard subgraph because not maximal
        is_sub = false
        for inodes_found in inodes_list_found
            is_sub = inodes ⊆ inodes_found
            is_sub && break
        end
        is_sub && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph to "found" and remove all previously found
        # subgraphs that are strictly contained in subgraph
        res = LInf_residual(graph, inodes, BD, N, solver_F)
        if res < ϵ*(1 + γ)
            empty!(inodes_list_bis)
            for inodes_found in inodes_list_found
                inodes_found ⊊ inodes && continue
                push!(inodes_list_bis, inodes_found)
            end
            inodes_list_found, inodes_list_bis = inodes_list_bis, inodes_list_found
            push!(inodes_list_found, inodes)
            continue
        end
        # if no, find an infeasibility certificate
        inode = max_local_L2_residual(graph, inodes, σ, β, N)[2]
        @assert !iszero(inode)
        xc = graph.nodes[inode].x
        λus, λls = infeasibility_certificate(
            graph, inodes, ϵ*(1 + γ/2), xc, solver_I
        )
        # select certificate nodes (with `θ`)
        θ = 1/2
        empty!(inodes_cert)
        while true
            for inode in inodes
                max(λus[inode], λls[inode]) < θ && continue
                push!(inodes_cert, inode)
            end
            # verify certificate
            res = LInf_residual(graph, inodes_cert, BD, N, solver_F)
            res > ϵ && break
            θ /= 2
        end
        # compute lims of infeasibility certificate
        fill!(lb, Inf)
        fill!(ub, -Inf)
        for inode in inodes_cert
            x = graph.nodes[inode].x
            for k = 1:N
                if x[k] < lb[k]
                    lb[k] = x[k]
                end
                if x[k] > ub[k]
                    ub[k] = x[k]
                end
            end
        end
        # add maximal sub-rectangles breaking the infeasibility certificate
        inodes_list_left = [BitSet() for k = 1:N]
        inodes_list_right = [BitSet() for k = 1:N]
        for inode in inodes
            for k = 1:N
                if graph.nodes[inode].x[k] < ub[k] - δ
                    push!(inodes_list_left[k], inode)
                end
                if graph.nodes[inode].x[k] > lb[k] + δ
                    push!(inodes_list_right[k], inode)
                end
            end
        end
        for inodes in Iterators.flatten((inodes_list_left, inodes_list_right))
            enqueue!(inodes_list_remain, KeyPQ(inodes) => -length(inodes))
        end
    end
    # Finished
    println("rects found: ", length(inodes_list_found))
    return inodes_list_found
end

"""
    greedy_covering(graph, ϵ, BD, σ, β, γ, δ, N, solvers...)

`ϵ`: max error;
`BD`: max value of system parameters;
`γ`: 'gap' for max error;
`σ`: radius for local residual;
`β`: regularization parameter;
`δ`: distance for certificate;
`N`: variable dimension;
`solver_F`: LP solver for feasibility
`solver_I`: LP solver for infeasibility
"""
function greedy_covering(
        graph::Graph, ϵ, BD, σ, β, γ, δ, N, solver_F, solver_I
    )
    inodes = BitSet(1:length(graph))
    inodes_list_remain = PriorityQueue(KeyPQ(inodes) => -length(inodes))
    inodes_list_found = BitSet[]
    inodes_cert = BitSet()
    iter = 0
    lb = fill(NaN, N)
    ub = fill(NaN, N)
    while !isempty(inodes_list_remain)
        iter += 1
        println(
            "iter: ", iter,
            " - rects remain: ", length(inodes_list_remain),
            " - rects found: ", length(inodes_list_found)
        )
        inodes = dequeue!(inodes_list_remain).inodes
        isempty(inodes) && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph and remove nodes for remaining subgraphs
        res = LInf_residual(graph, inodes, BD, N, solver_F)
        if res < ϵ*(1 + γ)
            push!(inodes_list_found, inodes)
            for k_remain in keys(inodes_list_remain)
                setdiff!(k_remain.inodes, inodes)
                inodes_list_remain[k_remain] = -length(k_remain.inodes)
            end
            continue
        end
        # if no, find an infeasibility certificate
        inode = max_local_L2_residual(graph, inodes, σ, β, N)[2]
        @assert !iszero(inode)
        xc = graph.nodes[inode].x
        λus, λls = infeasibility_certificate(
            graph, inodes, ϵ*(1 + γ/2), xc, solver_I
        )
        # select certificate nodes (with `θ`)
        θ = 1/2
        empty!(inodes_cert)
        while true
            for inode in inodes
                max(λus[inode], λls[inode]) < θ && continue
                push!(inodes_cert, inode)
            end
            # verify certificate
            res = LInf_residual(graph, inodes_cert, BD, N, solver_F)
            res > ϵ && break
            θ /= 2
        end
        # compute lims of infeasibility certificate
        fill!(lb, Inf)
        fill!(ub, -Inf)
        for inode in inodes_cert
            x = graph.nodes[inode].x
            for k = 1:N
                if x[k] < lb[k]
                    lb[k] = x[k]
                end
                if x[k] > ub[k]
                    ub[k] = x[k]
                end
            end
        end
        # add maximal sub-rectangles breaking the infeasibility certificate
        inodes_list_left = [BitSet() for k = 1:N]
        inodes_list_right = [BitSet() for k = 1:N]
        for inode in inodes
            for k = 1:N
                if graph.nodes[inode].x[k] < ub[k] - δ
                    push!(inodes_list_left[k], inode)
                end
                if graph.nodes[inode].x[k] > lb[k] + δ
                    push!(inodes_list_right[k], inode)
                end
            end
        end
        for inodes in Iterators.flatten((inodes_list_left, inodes_list_right))
            enqueue!(inodes_list_remain, KeyPQ(inodes) => -length(inodes))
        end
    end
    # Finished
    println("rects found: ", length(inodes_list_found))
    return inodes_list_found
end

"""
    optimal_covering(graph, ϵ, BD, σ, β, γ, δ, N, solvers...)

`ϵ`: max error;
`BD`: max value of system parameters;
`γ`: 'gap' for max error;
`σ`: radius for local residual;
`β`: regularization parameter;
`δ`: distance for certificate;
`N`: variable dimension;
`solver_F`: LP solver for feasibility
`solver_I`: LP solver for infeasibility
`solver_C`: MILP solver for cover
"""
function optimal_covering(
        graph::Graph, ϵ, BD, σ, β, γ, δ, N, solver_F, solver_I, solver_C
    )
    nnode = length(graph)
    inodes = BitSet(1:length(graph))
    inodes_list_remain = PriorityQueue(KeyPQ(inodes) => -length(inodes))
    inodes_list_found = BitSet[]
    inodes_list_bis = BitSet[]
    inodes_cert = BitSet()
    inodes_list_all = BitSet[]
    inodes_found_union = BitSet()
    ncov_lower::Int = 1
    ncov_upper::Int = nnode
    all_nodes = BitSet(1:nnode)
    iter = 0
    lb = fill(NaN, N)
    ub = fill(NaN, N)
    while !isempty(inodes_list_remain) && ncov_lower < ncov_upper
        iter += 1
        println(
            "iter: ", iter,
            " - rects remain: ", length(inodes_list_remain),
            " - rects found: ", length(inodes_list_found),
            " - ncov_lower: ", ncov_lower,
            " - ncov_upper: ", ncov_upper
        )
        inodes = dequeue!(inodes_list_remain).inodes
        isempty(inodes) && continue
        # check if contained in a found subgraph
        # if yes, discard subgraph because not maximal
        is_sub = false
        for inodes_found in inodes_list_found
            is_sub = inodes ⊆ inodes_found
            is_sub && break
        end
        is_sub && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph to "found" and remove all previously found
        # subgraphs that are strictly contained in subgraph
        # finally, update upper bound on covering number
        res = LInf_residual(graph, inodes, BD, N, solver_F)
        if res < ϵ*(1 + γ)
            empty!(inodes_list_bis)
            for inodes_found in inodes_list_found
                inodes_found ⊊ inodes && continue
                push!(inodes_list_bis, inodes_found)
            end
            inodes_list_found, inodes_list_bis = inodes_list_bis, inodes_list_found
            push!(inodes_list_found, inodes)
            # update upper bound on covering number
            empty!(inodes_found_union)
            for inodes_found in inodes_list_found
                union!(inodes_found_union, inodes_found)
            end
            !(all_nodes == inodes_found_union) && continue
            bs = optimal_set_cover(nnode, inodes_list_found, solver_C)
            ncov_upper = sum(b -> round(Int, b), bs)
            continue
        end
        # if no, find an infeasibility certificate
        inode = max_local_L2_residual(graph, inodes, σ, β, N)[2]
        @assert !iszero(inode)
        xc = graph.nodes[inode].x
        λus, λls = infeasibility_certificate(
            graph, inodes, ϵ*(1 + γ/2), xc, solver_I
        )
        # select certificate nodes (with `θ`)
        θ = 1/2
        empty!(inodes_cert)
        while true
            for inode in inodes
                max(λus[inode], λls[inode]) < θ && continue
                push!(inodes_cert, inode)
            end
            # verify certificate
            res = LInf_residual(graph, inodes_cert, BD, N, solver_F)
            res > ϵ && break
            θ /= 2
        end
        # verify certificate
        res = LInf_residual(graph, inodes, BD, N, solver_F)
        @assert res > ϵ
        # compute lims of infeasibility certificate
        fill!(lb, Inf)
        fill!(ub, -Inf)
        for inode in inodes_cert
            x = graph.nodes[inode].x
            for k = 1:N
                if x[k] < lb[k]
                    lb[k] = x[k]
                end
                if x[k] > ub[k]
                    ub[k] = x[k]
                end
            end
        end
        # add maximal sub-rectangles breaking the infeasibility certificate
        inodes_list_left = [BitSet() for k = 1:N]
        inodes_list_right = [BitSet() for k = 1:N]
        for inode in inodes
            for k = 1:N
                if graph.nodes[inode].x[k] < ub[k] - δ
                    push!(inodes_list_left[k], inode)
                end
                if graph.nodes[inode].x[k] > lb[k] + δ
                    push!(inodes_list_right[k], inode)
                end
            end
        end
        for inodes in Iterators.flatten((inodes_list_left, inodes_list_right))
            enqueue!(inodes_list_remain, KeyPQ(inodes) => -length(inodes))
        end
        # update lower_bound on covering number
        empty!(inodes_list_all)
        for inodes in Iterators.flatten((
                inodes_list_found,
                k.inodes for k in keys(inodes_list_remain)
            ))
            push!(inodes_list_all, inodes)
        end
        bs = optimal_set_cover(nnode, inodes_list_all, solver_C)
        ncov_lower = sum(b -> round(Int, b), bs)
    end
    # Finished
    println("rects found: ", length(inodes_list_found))
    return inodes_list_found
end

end # module
