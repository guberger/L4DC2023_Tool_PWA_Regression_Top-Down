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

function infeasibility_certificate_LP(nodes, inodes, ϵ, xc, N, solver)
    model = solver()
    λus = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # upper bound : a*x ≤ y + ϵ
    λls = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # lower bound : -a*x ≤ -y + ϵ
    # con_x::Vector{AffExpr} = [AffExpr(0) for k in 1:N]
    # con_1::AffExpr = AffExpr(0)
    # con_η::AffExpr = AffExpr(0)
    # obj::AffExpr = AffExpr(0)
    # for inode in inodes
    #     con_x += (λus[inode] - λls[inode])*nodes[inode].x
    #     con_1 += λus[inode] + λls[inode]
    #     con_η += (λus[inode] - λls[inode])*nodes[inode].η
    #     obj += (λus[inode] + λls[inode])*norm(nodes[inode].x - xc)^2
    # end
    con_x::Vector{AffExpr} = [AffExpr(0) for k in 1:N]
    con_1::AffExpr = AffExpr(0)
    con_η::AffExpr = AffExpr(0)
    obj::AffExpr = AffExpr(0)
    for inode in inodes
        λsum = λus[inode] + λls[inode]
        λdif = λus[inode] - λls[inode]
        for k = 1:N
            add_to_expression!(con_x[k], λdif, nodes[inode].x[k])
        end
        add_to_expression!(con_1, λsum)
        add_to_expression!(con_η, λdif, nodes[inode].η)
        add_to_expression!(obj, λsum, norm(nodes[inode].x - xc)^2)
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

function infeasibility_certificate_MILP(nodes, inodes, ϵ, xc, N, solver)
    model = solver()
    λus = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # upper bound : a*x ≤ y + ϵ
    λls = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # lower bound : -a*x ≤ -y + ϵ
    bins = Dict(
        inode => @variable(model, binary=true) for inode in inodes
    )
    con_x::Vector{AffExpr} = [AffExpr(0) for k in 1:N]
    con_1::AffExpr = AffExpr(0)
    con_η::AffExpr = AffExpr(0)
    con_bins::AffExpr = AffExpr(0)
    obj::AffExpr = AffExpr(0)
    for inode in inodes
        λsum = λus[inode] + λls[inode]
        λdif = λus[inode] - λls[inode]
        for k = 1:N
            add_to_expression!(con_x[k], λdif, nodes[inode].x[k])
        end
        add_to_expression!(con_1, λsum)
        add_to_expression!(con_η, λdif, nodes[inode].η)
        add_to_expression!(con_bins, bins[inode])
        @constraint(model, bins[inode] ≥ λsum/2)
        add_to_expression!(obj, bins[inode], norm(nodes[inode].x - xc)^2)
    end
    @constraint(model, con_x .== 0)
    @constraint(model, con_1 == 1)
    @constraint(model, con_η ≤ -ϵ)
    @constraint(model, con_bins ≤ N + 1)
    @objective(model, Min, obj)
    optimize!(model)
    if termination_status(model) == OPTIMAL
        @assert primal_status(model) == FEASIBLE_POINT
        return (
            objective_value(model),
            Dict(inode => value(bin) for (inode, bin) in bins)
        )
    end
    @warn(termination_status(model))
    return Inf, Dict(inode => 1 for (inode, bin) in bins)
end

# For tests: see example certificate center
function infeasibility_certificate_MILP_prox(
        nodes, inodes, ϵ, xc, μ, N, solver
    )
    model = solver()
    λus = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # upper bound : a*x ≤ y + ϵ
    λls = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # lower bound : -a*x ≤ -y + ϵ
    bins = Dict(
        inode => @variable(model, binary=true) for inode in inodes
    )
    con_x::Vector{AffExpr} = [AffExpr(0) for k in 1:N]
    con_1::AffExpr = AffExpr(0)
    con_η::AffExpr = AffExpr(0)
    con_bins::AffExpr = AffExpr(0)
    obj_c::AffExpr = AffExpr(0) # center
    obj_p::AffExpr = AffExpr(0) # proximity
    jnodes = copy(inodes)
    for inode in inodes
        delete!(jnodes, inode)
        λsum = λus[inode] + λls[inode]
        λdif = λus[inode] - λls[inode]
        for k = 1:N
            add_to_expression!(con_x[k], λdif, nodes[inode].x[k])
        end
        add_to_expression!(con_1, λsum)
        add_to_expression!(con_η, λdif, nodes[inode].η)
        add_to_expression!(con_bins, bins[inode])
        @constraint(model, bins[inode] ≥ λsum/2)
        add_to_expression!(obj_c, bins[inode], norm(nodes[inode].x - xc)^2)
        for jnode in jnodes
            t = @variable(model, lower_bound=0)
            @constraint(model, t ≥ bins[inode] + bins[jnode] - 1)
            add_to_expression!(
                obj_p, t, norm(nodes[inode].x - nodes[jnode].x)^2
            )
        end
    end
    @constraint(model, con_x .== 0)
    @constraint(model, con_1 == 1)
    @constraint(model, con_η ≤ -ϵ)
    @constraint(model, con_bins ≤ N + 1)
    @objective(model, Min, obj_c*μ + obj_p)
    optimize!(model)
    if termination_status(model) == OPTIMAL
        @assert primal_status(model) == FEASIBLE_POINT
        return (
            objective_value(model),
            Dict(inode => value(bin) for (inode, bin) in bins)
        )
    end
    @warn(termination_status(model))
    return Inf, Dict(inode => 1 for (inode, bin) in bins)
end

# For tests: see example certificate center
function infeasibility_certificate_MILP_radius(
        nodes, inodes, ϵ, xc, ρ, N, solver
    )
    model = solver()
    λus = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # upper bound : a*x ≤ y + ϵ
    λls = Dict(
        inode => @variable(model, lower_bound=0) for inode in inodes
    ) # lower bound : -a*x ≤ -y + ϵ
    bins = Dict(
        inode => @variable(model, binary=true) for inode in inodes
    )
    con_x::Vector{AffExpr} = [AffExpr(0) for k in 1:N]
    con_1::AffExpr = AffExpr(0)
    con_η::AffExpr = AffExpr(0)
    con_bins::AffExpr = AffExpr(0)
    obj::AffExpr = AffExpr(0)
    jnodes = copy(inodes)
    for inode in inodes
        delete!(jnodes, inode)
        λsum = λus[inode] + λls[inode]
        λdif = λus[inode] - λls[inode]
        for k = 1:N
            add_to_expression!(con_x[k], λdif, nodes[inode].x[k])
        end
        add_to_expression!(con_1, λsum)
        add_to_expression!(con_η, λdif, nodes[inode].η)
        add_to_expression!(con_bins, bins[inode])
        @constraint(model, bins[inode] ≥ λsum/2)
        add_to_expression!(obj, bins[inode], norm(nodes[inode].x - xc)^2)
        for jnode in jnodes
            norm(nodes[inode].x - nodes[jnode].x) ≤ ρ && continue
            @constraint(model, bins[inode] + bins[jnode] ≤ 1)
        end
    end
    @constraint(model, con_x .== 0)
    @constraint(model, con_1 == 1)
    @constraint(model, con_η ≤ -ϵ)
    @constraint(model, con_bins ≤ N + 1)
    @objective(model, Min, obj)
    optimize!(model)
    if termination_status(model) == OPTIMAL
        @assert primal_status(model) == FEASIBLE_POINT
        return (
            objective_value(model),
            Dict(inode => value(bin) for (inode, bin) in bins)
        )
    end
    @warn(termination_status(model))
    return Inf, Dict(inode => 1 for (inode, bin) in bins)
end

# Extract certificate

function extract_infeasibility_certificate_LP(
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

function extract_infeasibility_certificate_MILP(inodes, bins, θ)
    # select certificate nodes (with `θ`)
    inodes_cert = BitSet()
    for inode in inodes
        bins[inode] ≥ θ && push!(inodes_cert, inode)
    end
    return inodes_cert
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

# Utils

function compute_center(xlist, indices, N)
    xc = zeros(N)
    for index in indices
        xc += xlist[index]
    end
    return xc/length(indices)
end

function compute_lims(xlist, indices, N)
    lb = fill(Inf, N)
    ub = fill(-Inf, N)
    for index in indices
        for k = 1:N
            if xlist[index][k] < lb[k]
                lb[k] = xlist[index][k]
            end
            if xlist[index][k] > ub[k]
                ub[k] = xlist[index][k]
            end
        end
    end
    return lb, ub
end

# Regions

struct Elem
    inodes::BitSet
    length::Int
end

function _add_elem!(elem_list, inodes_list_ko, inodes_list_ok, inodes)
    any(x -> x == inodes, inodes_list_ko) && return nothing
    any(x -> x ⊇ inodes, inodes_list_ok) && return nothing
    any(x -> x.inodes == inodes, elem_list) && return nothing
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
`solver_F`: solver for feasibility
`solver_I`: solver for infeasibility
`solver_C`: solver for cover
"""
function optimal_covering(
        nodes::Vector{<:Node}, ϵ, BD, γ, δ, N,
        solver_F, solver_I, solver_C
    )
    nnode = length(nodes)
    xlist = map(node -> node.x, nodes)
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
        # compute L∞ residual and check if ≤ relaxed tolerance `ϵ*(1 + γ)`
        # if yes, add subgraph to "ok" and remove all previously ok
        # subgraphs that are strictly contained in subgraph
        # finally, update upper bound on covering number
        res = Inf
        if all(x -> !(x ⊆ inodes), inodes_list_ko)
            res = LInf_residual(nodes, inodes, BD, N, solver_F)
        end
        if res < ϵ*(1 + γ)
            println("--> New ok!")
            filter!(x -> !(x ⊆ inodes), inodes_list_ok)
            filter!(x -> !(x.inodes ⊆ inodes), elem_list_remain)
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
            xc = compute_center(xlist, inodes, N)
            # TODO: add bound on affine function parameters in infeasibility
            # certificate to ensure feasibility when res > ϵ
            bins = infeasibility_certificate_MILP(
                nodes, inodes, ϵ*(1 + γ/2), xc, N, solver_I
            )[2]
            inodes_cert = extract_infeasibility_certificate_MILP(
                inodes, bins, 0.5
            )
            # compute lims of infeasibility certificate
            lb, ub = compute_lims(xlist, inodes_cert, N)
            # add maximal sub-rectangles breaking the infeasibility certificate
            for k = 1:N
                for (s, b) in ((1, ub), (-1, lb))
                    inodes_new = filter(
                        y -> s*(nodes[y].x[k] - b[k]) + δ < 0, inodes
                    )
                    _add_elem!(
                        elem_list_remain, inodes_list_ko,
                        inodes_list_ok, inodes_new
                    )
                end
            end
            lower_bound_current = false
        end
        if cover_with_ok && !lower_bound_current
            # update lower_bound on covering number
            empty!(inodes_list_all)
            append!(inodes_list_all, inodes_list_ok)
            foreach(x -> push!(inodes_list_all, x.inodes), elem_list_remain)
            print("--> # node = ", length(inodes_list_all))
            for elem in elem_list_remain
                filter!(x -> !(x ⊊ elem.inodes), inodes_list_all)
            end
            println(" - # node = ", length(inodes_list_all))
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
