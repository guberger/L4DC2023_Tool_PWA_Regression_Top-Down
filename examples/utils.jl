function minimax_regression(nodes, inodes, BD, N, solver)
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
    return value.(a)
end

function switched_bounded_regression(nodes, inodes, M, ϵ, BD, N, Γ, solver)
    model = solver()
    as = [@variable(model, [1:N], lower_bound=-BD, upper_bound=BD) for q in 1:M]
    bins = Dict(inode => @variable(model, [1:M], binary=true) for inode in inodes)
    for inode in inodes
        node = nodes[inode]
        @constraint(model, sum(bins[inode]) == 1)
        for q in 1:M
            Δ = Γ*(1 - bins[inode][q])
            @constraint(model, dot(as[q], node.x) ≤ node.η + ϵ + Δ)
            @constraint(model, dot(as[q], node.x) ≥ node.η - ϵ - Δ)
        end
    end
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    as_opt = map(a -> value.(a), as)
    bins_opt = Dict(inode => value.(bin) for (inode, bin) in bins)
    return as_opt, bins_opt
end