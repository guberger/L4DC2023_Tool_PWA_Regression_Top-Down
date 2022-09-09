function local_residual!(res, subgraph::Subgraph, xc, σ, nx, ny)
    M = fill(NaN, nx, length(subgraph))
    N = fill(NaN, ny, length(subgraph))
    ω_tot = 0.0
    for (t, inode) in enumerate(subgraph.inodes)
        ρ = norm(subgraph.graph.nodes[inode].x - xc)
        ω = exp(-((ρ/σ)^2)/2)
        for k in 1:nx
            M[k, t] = subgraph.graph.nodes[inode].x[k]*ω
        end
        for k in 1:ny
            N[k, t] = subgraph.graph.nodes[inode].y[k]*ω
        end
        ω_tot += ω
    end
    try
        A = N / M
        for (k, r) in enumerate(eachrow(A*M - N))
            res[k] = norm(r)
        end
    catch
        @warn("Catching SingularException")
        A = (N*M') / (M*M' + 1e-6*I)
        for (k, r) in enumerate(eachrow(A*M - N))
            res[k] = norm(r)
        end
    end
end

function max_local_residual!(
        res_max, inodes_opt, res, subgraph::Subgraph, σ, nx, ny
    )
    for inode in subgraph.inodes
        xc = subgraph.graph.nodes[inode].x
        local_residual!(res, subgraph, xc, σ, nx, ny)
        for k = 1:ny
            if res[k] > res_max[k]
                res_max[k] = res[k]
                inodes_opt[k] = inode
            end
        end
    end
end

function find_infeasible(subgraph::Subgraph, ϵ, xc, ky, solver)
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
        (λus[inode] - λls[inode])*subgraph.graph.nodes[inode].y[ky]
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