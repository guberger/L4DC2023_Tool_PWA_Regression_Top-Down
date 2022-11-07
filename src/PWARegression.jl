module PWARegression

using LinearAlgebra
using JuMP

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

# Extract certificate

function find_infeasibility_certificate(
        nodes, inodes, ϵ, ϵ_up, BD, N, solver_F, solver_I
    )
    xc = zeros(N)
    for inode in inodes
        xc += nodes[inode].x
    end
    xc /= length(inodes)
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

struct Elem
    inodes::BitSet
    length::Int
end

function _add_elem!(elem_list, inodes_list_out, inodes)
    any(
        inodes_out -> inodes_out == inodes, inodes_list_out
    ) && return nothing
    any(
        elem -> elem.inodes == inodes, elem_list
    ) && return nothing
    push!(elem_list, Elem(inodes, length(inodes)))
    return nothing
end

"""
    optimal_covering(nodes, ϵ, BD, γ, δ, N, solvers...)

`ϵ`: max error;
`BD`: max value of system parameters;
`γ`: 'gap' for max error;
`δ`: distance for certificate;
`N`: variable dimension;
`solver_F`: LP solver for feasibility
`solver_I`: LP solver for infeasibility
`solver_C`: MILP solver for cover
"""
function optimal_covering(
        nodes::Vector{<:Node}, ϵ, BD, γ, δ, N,
        solver_F, solver_I, solver_C
    )
    nnode = length(nodes)
    elem_list_remain = [Elem(BitSet(1:nnode), nnode)]
    inodes_list_ko = BitSet[]
    inodes_list_ok = BitSet[]
    inodes_list_all = BitSet[]
    inodes_full = BitSet()
    ncov_lower::Int = 1
    ncov_upper::Int = nnode
    iter = 0
    cover_with_ok = false
    lower_bound_current = true
    while !isempty(elem_list_remain) && ncov_lower < ncov_upper
        iter += 1
        nremain = length(elem_list_remain)
        println(
            "iter: ", iter,
            ". rem: ", nremain,
            ". ok: ", length(inodes_list_ok),
            ". ko: ", length(inodes_list_ko),
            ". ↑: ", ncov_lower,
            ". ↓: ", ncov_upper
        )
        ic = argmax(i -> elem_list_remain[i].length, 1:nremain)
        inodes = elem_list_remain[ic].inodes
        deleteat!(elem_list_remain, ic)
        # check if contained in a found subgraph
        any(x -> inodes ⊆ x, inodes_list_ok) && continue
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph to "ok" and remove all previously ok
        # subgraphs that are strictly contained in subgraph
        # finally, update upper bound on covering number
        res = LInf_residual(nodes, inodes, BD, N, solver_F)
        if res < ϵ*(1 + γ)
            println("--> New ok!")
            filter!(x -> !(x ⊆ inodes), inodes_list_ok)
            push!(inodes_list_ok, inodes)
            # update upper bound on covering number
            union!(inodes_full, 1:nnode)
            for inodes_ok in inodes_list_ok
                setdiff!(inodes_full, inodes_ok)
            end
            cover_with_ok = isempty(inodes_full)
            if cover_with_ok
                bs = optimal_set_cover(nnode, inodes_list_ok, solver_C)
                ncov_upper = sum(b -> round(Int, b), bs)
                println("--> ↓: ", ncov_upper)
            end
        else
            push!(inodes_list_ko, inodes)
            # if no, find an infeasibility certificate
            inodes_cert = find_infeasibility_certificate(
                nodes, inodes, ϵ, ϵ*(1 + γ/2), BD, N, solver_F, solver_I
            )
            # compute lims of infeasibility certificate
            lb, ub = compute_lims(nodes, inodes_cert, N)
            # add maximal sub-rectangles breaking the infeasibility certificate
            for k = 1:N
                inodes_new = filter(y -> nodes[y].x[k] < ub[k] - δ, inodes)
                _add_elem!(elem_list_remain, inodes_list_ko, inodes_new)
                inodes_new = filter(y -> nodes[y].x[k] > lb[k] + δ, inodes)
                _add_elem!(elem_list_remain, inodes_list_ko, inodes_new)
            end
            lower_bound_current = false
        end
        if cover_with_ok && !lower_bound_current
            # update lower_bound on covering number
            empty!(inodes_list_all)
            append!(inodes_list_all, inodes_list_ok)
            foreach(
                elem -> push!(inodes_list_all, elem.inodes), elem_list_remain
            )
            bs = optimal_set_cover(nnode, inodes_list_all, solver_C)
            ncov_lower = sum(b -> round(Int, b), bs)
            println("--> ↑: ", ncov_lower)
            lower_bound_current = true
        end
    end
    # Finished
    println("rects found: ", length(inodes_list_ok))
    return inodes_list_ok
end

end # module
