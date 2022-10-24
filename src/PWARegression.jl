module PWARegression

using LinearAlgebra
using JuMP
using DataStructures

struct Node{XT<:AbstractVector,HT}
    x::XT
    η::HT
end

# Residuals

function LInf_residual(nodes, inodes, BD, N, solver)
    model = solver()
    a = @variable(model, [1:N], lower_bound=-BD, upper_bound=BD)
    r = @variable(model, lower_bound=-1)
    for inode in inodes
        node = nodes[inode]
        @constraint(model, dot(a, node.x) ≤ node.η + r)
        @constraint(model, dot(a, node.x) ≥ node.η - r)
    end
    @objective(model, Min, r)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return value(r)
end

function local_L2_residual(nodes, inodes, xc, σ, β, N)
    nnode = length(inodes)
    xMat = zeros(nnode + N, N)
    ηMat = zeros(nnode + N)
    for (t, inode) in enumerate(inodes)
        ρ = norm(nodes[inode].x - xc)
        ω = exp(-((ρ/σ)^2)/2)
        for k in 1:N
            xMat[t, k] = nodes[inode].x[k]*ω
        end
        ηMat[t] = nodes[inode].η*ω
    end
    for k = 1:N
        xMat[nnode + k, k] = β
    end
    a = xMat \ ηMat
    return norm(xMat*a - ηMat)
    # try
    #     a = xMat \ ηMat
    #     return norm(xMat*a - ηMat)
    # catch
    #     @warn("Catching SingularException")
    #     a = (xMat'*xMat + 1e-6*I) \ (xMat'*ηMat)
    #     return norm(xMat*a - ηMat)
    # end
end

function max_local_L2_residual(nodes, inodes, σ, ρ, N)
    res_max = -Inf
    inode_opt = 0
    for inode in inodes
        xc = nodes[inode].x
        res = local_L2_residual(nodes, inodes, xc, σ, ρ, N)
        if res > res_max
            res_max = res
            inode_opt = inode
        end
    end
    return inode_opt
end

# Certificate

function infeasibility_certificate(nodes, inodes, ϵ, xc, solver)
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
        (λus[inode] - λls[inode])*nodes[inode].x
        for inode in inodes
    ) .== 0)
    @constraint(model, sum(
        (λus[inode] - λls[inode])*nodes[inode].η
        for inode in inodes
    ) ≤ -ϵ)
    @objective(model, Min, sum(
        (λus[inode] + λls[inode])*norm(nodes[inode].x - xc)^2
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
    @constraint(model, cons .≥ 1)
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

PairPQ(inodes) = KeyPQ(inodes) => -length(inodes)

function find_infeasibility_certificate(
        nodes, inodes, ϵ, ϵ_up, BD, σ, β, N, solver_F, solver_I
    )
    inode = max_local_L2_residual(nodes, inodes, σ, β, N)
    @assert !iszero(inode)
    xc = nodes[inode].x
    λus, λls = infeasibility_certificate(nodes, inodes, ϵ_up, xc, solver_I)
    # select certificate nodes (with `θ`)
    θ = 1/2
    inodes_cert = BitSet()
    while true
        for inode in inodes
            max(λus[inode], λls[inode]) ≥ θ && push!(inodes_cert, inode)
        end
        # verify certificate
        LInf_residual(
            nodes, inodes_cert, BD, N, solver_F
        ) > ϵ && return inodes_cert
        θ /= 2
    end
end

function compute_lims(nodes, inodes, N)
    lb = fill(Inf, N)
    ub = fill(-Inf, N)
    for inode in inodes
        x = nodes[inode].x
        for k = 1:N
            if x[k] < lb[k]
                lb[k] = x[k]
            end
            if x[k] > ub[k]
                ub[k] = x[k]
            end
        end
    end
    return lb, ub
end

"""
    maximal_regions(nodes, ϵ, BD, σ, β, γ, δ, N, solvers...)

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
        nodes::Vector{<:Node}, ϵ, BD, σ, β, γ, δ, N, solver_F, solver_I
    )
    inodes_list_remain = PriorityQueue(PairPQ(BitSet(1:length(nodes))))
    inodes_list_found = BitSet[]
    iter = 0
    while !isempty(inodes_list_remain)
        iter += 1
        println(
            "iter: ", iter,
            ". remain: ", length(inodes_list_remain),
            ". found: ", length(inodes_list_found)
        )
        inodes = dequeue!(inodes_list_remain).inodes
        isempty(inodes) && continue
        # check if contained in a found subgraph
        any(x -> inodes ⊆ x, inodes_list_found) && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph to "found" and remove all previously found
        # subgraphs that are strictly contained in subgraph
        res = LInf_residual(nodes, inodes, BD, N, solver_F)
        if res < ϵ*(1 + γ)
            println("--> New found!")
            filter!(x -> !(x ⊆ inodes), inodes_list_found)
            push!(inodes_list_found, inodes)
            continue
        end
        # if no, find an infeasibility certificate
        inodes_cert = find_infeasibility_certificate(
            nodes, inodes, ϵ, ϵ*(1 + γ/2), BD, σ, β, N, solver_F, solver_I
        )
        # compute lims of infeasibility certificate
        lb, ub = compute_lims(nodes, inodes_cert, N)
        # add maximal sub-rectangles breaking the infeasibility certificate
        for k = 1:N
            enqueue!(inodes_list_remain,
                PairPQ(filter(y -> nodes[y].x[k] < ub[k] - δ, inodes)))
            enqueue!(inodes_list_remain,
                PairPQ(filter(y -> nodes[y].x[k] > lb[k] + δ, inodes)))
        end
    end
    # Finished
    println("rects found: ", length(inodes_list_found))
    return inodes_list_found
end

"""
    greedy_covering(nodes, ϵ, BD, σ, β, γ, δ, N, solvers...)

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
        nodes::Vector{<:Node}, ϵ, BD, σ, β, γ, δ, N, solver_F, solver_I
    )
    inodes_list_remain = PriorityQueue(PairPQ(BitSet(1:length(nodes))))
    inodes_list_found = BitSet[]
    iter = 0
    while !isempty(inodes_list_remain)
        iter += 1
        println(
            "iter: ", iter,
            ". remain: ", length(inodes_list_remain),
            ". found: ", length(inodes_list_found)
        )
        inodes = dequeue!(inodes_list_remain).inodes
        isempty(inodes) && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph and remove nodes for remaining subgraphs
        res = LInf_residual(nodes, inodes, BD, N, solver_F)
        if res < ϵ*(1 + γ)
            println("--> New found!")
            push!(inodes_list_found, inodes)
            for k_remain in keys(inodes_list_remain)
                setdiff!(k_remain.inodes, inodes)
                inodes_list_remain[k_remain] = -length(k_remain.inodes)
            end
            continue
        end
        # if no, find an infeasibility certificate
        inodes_cert = find_infeasibility_certificate(
            nodes, inodes, ϵ, ϵ*(1 + γ/2), BD, σ, β, N, solver_F, solver_I
        )
        # compute lims of infeasibility certificate
        lb, ub = compute_lims(nodes, inodes_cert, N)
        # add maximal sub-rectangles breaking the infeasibility certificate
        for k = 1:N
            enqueue!(inodes_list_remain,
                PairPQ(filter(y -> nodes[y].x[k] < ub[k] - δ, inodes)))
            enqueue!(inodes_list_remain,
                PairPQ(filter(y -> nodes[y].x[k] > lb[k] + δ, inodes)))
        end
    end
    # Finished
    println("rects found: ", length(inodes_list_found))
    return inodes_list_found
end

"""
    optimal_covering(nodes, ϵ, BD, σ, β, γ, δ, N, solvers...)

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
        nodes::Vector{<:Node}, ϵ, BD, σ, β, γ, δ, N,
        solver_F, solver_I, solver_C
    )
    nnode = length(nodes)
    inodes_list_remain = PriorityQueue(PairPQ(BitSet(1:nnode)))
    inodes_list_found = BitSet[]
    inodes_list_all = BitSet[]
    inodes_found_union = BitSet()
    ncov_lower::Int = 1
    ncov_upper::Int = nnode
    iter = 0
    while !isempty(inodes_list_remain) && ncov_lower < ncov_upper
        iter += 1
        println(
            "iter: ", iter,
            ". remain: ", length(inodes_list_remain),
            ". found: ", length(inodes_list_found),
            ". ncov_lower: ", ncov_lower,
            ". ncov_upper: ", ncov_upper
        )
        inodes = dequeue!(inodes_list_remain).inodes
        isempty(inodes) && continue
        # check if contained in a found subgraph
        any(x -> inodes ⊆ x, inodes_list_found) && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph to "found" and remove all previously found
        # subgraphs that are strictly contained in subgraph
        # finally, update upper bound on covering number
        res = LInf_residual(nodes, inodes, BD, N, solver_F)
        if res < ϵ*(1 + γ)
            println("--> New found!")
            filter!(x -> !(x ⊆ inodes), inodes_list_found)
            push!(inodes_list_found, inodes)
            # update upper bound on covering number
            empty!(inodes_found_union)
            for inodes_found in inodes_list_found
                union!(inodes_found_union, inodes_found)
            end
            !(length(inodes_found_union) == nnode) && continue
            bs = optimal_set_cover(nnode, inodes_list_found, solver_C)
            ncov_upper = sum(b -> round(Int, b), bs)
            println("--> ncov_upper: ", ncov_upper)
            continue
        end
        # if no, find an infeasibility certificate
        inodes_cert = find_infeasibility_certificate(
            nodes, inodes, ϵ, ϵ*(1 + γ/2), BD, σ, β, N, solver_F, solver_I
        )
        # compute lims of infeasibility certificate
        lb, ub = compute_lims(nodes, inodes_cert, N)
        # add maximal sub-rectangles breaking the infeasibility certificate
        for k = 1:N
            enqueue!(inodes_list_remain,
                PairPQ(filter(y -> nodes[y].x[k] < ub[k] - δ, inodes)))
            enqueue!(inodes_list_remain,
                PairPQ(filter(y -> nodes[y].x[k] > lb[k] + δ, inodes)))
        end
        # update lower_bound on covering number
        empty!(inodes_list_all)
        append!(inodes_list_all, inodes_list_found)
        foreach(
            k -> push!(inodes_list_all, k.inodes), keys(inodes_list_remain)
        )
        bs = optimal_set_cover(nnode, inodes_list_all, solver_C)
        ncov_lower = sum(b -> round(Int, b), bs)
        println("--> ncov_lower: ", ncov_lower)
    end
    # Finished
    println("rects found: ", length(inodes_list_found))
    return inodes_list_found
end

end # module
