function minimax_regression(nodes, inodes, BD, M, N, solver)
    model = solver()
    A = @variable(model, [1:M, 1:N], lower_bound=-BD, upper_bound=BD)
    r = @variable(model, lower_bound=-1)
    for inode in inodes
        node = nodes[inode]
        @constraint(model, A * node.x .≤ node.y .+ r)
        @constraint(model, A * node.x .≥ node.y .- r)
    end
    @objective(model, Min, r)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return value.(A)
end

function switched_bounded_regression(nodes, inodes, Q, ϵ, BD, M, N, Γ, solver)
    model = solver()
    var() = @variable(model, [1:M, 1:N], lower_bound=-BD, upper_bound=BD)
    As = [var() for _ in 1:Q]
    bins = Dict(inode => @variable(model, [1:M], binary=true) for inode in inodes)
    for inode in inodes
        node = nodes[inode]
        @constraint(model, sum(bins[inode]) == 1)
        for q in 1:M
            Δ = Γ*(1 - bins[inode][q])
            @constraint(model, As[q] * node.x .≤ node.η .+ (ϵ + Δ))
            @constraint(model, As[q] * node.x .≥ node.η .- (ϵ + Δ))
        end
    end
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    As_opt = map(a -> value.(a), As)
    bins_opt = Dict(inode => value.(bin) for (inode, bin) in bins)
    return As_opt, bins_opt
end