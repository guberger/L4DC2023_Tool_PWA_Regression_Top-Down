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

function infeasibility_certificate(nodes, inodes, ϵ, xc, N, solver)
    model = solver()
    λus = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # upper bound : a*x ≤ y + ϵ
    λls = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # lower bound : -a*x ≤ -y + ϵ
    con_x::Vector{AffExpr} = [AffExpr(0) for k in 1:N]
    con_1::AffExpr = AffExpr(0)
    con_η::AffExpr = AffExpr(0)
    obj::AffExpr = AffExpr(0)
    for inode in inodes
        con_x += (λus[inode] - λls[inode])*nodes[inode].x
        con_1 += λus[inode] + λls[inode]
        con_η += (λus[inode] - λls[inode])*nodes[inode].η
        obj += (λus[inode] + λls[inode])*norm(nodes[inode].x - xc)^2
    end
    @constraint(model, con_x .== 0)
    @constraint(model, con_1 == 1)
    @constraint(model, con_η ≤ -ϵ)
    @objective(model, Min, obj)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return (
        objective_value(model),
        Dict(inode => value(λ) for (inode, λ) in λus),
        Dict(inode => value(λ) for (inode, λ) in λls)
    )
end

# Extract certificate

function find_infeasibility_certificate(nodes, inodes, ϵ, N, solver)
    obj_opt = Inf
    inode_opt = 0
    λus_opt = Dict{Int,Float64}()
    λls_opt = Dict{Int,Float64}()
    for inode in inodes
        xc = nodes[inode].x
        obj, λus, λls = infeasibility_certificate(
            nodes, inodes, ϵ, xc, N, solver
        )
        if obj < obj_opt
            inode_opt = inode
            obj_opt = obj
            λus_opt = λus
            λls_opt = λls
        end
    end
    return inode_opt, λus_opt, λls_opt
end

function extract_infeasibility_certificate(
        nodes, inodes, λus, λls, ϵ, θ, BD, N, solver
    )
    # select certificate nodes (with `θ`)
    inodes_cert = BitSet()
    while true
        for inode in inodes
            max(λus[inode], λls[inode]) ≥ θ && push!(inodes_cert, inode)
        end
        # verify certificate
        LInf_residual(
            nodes, inodes_cert, BD, N, solver
        ) > ϵ && return inodes_cert
        θ /= 2
    end
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
