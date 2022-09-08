function verify(subgraph::Subgraph, arad, nx, ny, solver)
    model = solver()
    A = @variable(model, [1:ny, 1:nx], lower_bound=-arad, upper_bound=arad)
    r = @variable(model, lower_bound=-1)
    for inode in subgraph.inodes
        node = subgraph.graph.nodes[inode]
        @constraint(model, A*node.x .≤ node.y .+ r)
        @constraint(model, A*node.x .≥ node.y .- r)
    end
    @objective(model, Min, r)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT
    return value(r)
end