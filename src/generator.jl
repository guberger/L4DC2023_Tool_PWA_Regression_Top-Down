function generate(subgraph::Subgraph, xc, σ, nx, ny)
    M = fill(NaN, nx, length(subgraph))
    N = fill(NaN, ny, length(subgraph))
    ω_tot = 0.0
    for (t, inode) in enumerate(subgraph.inodes)
        ρ = norm(subgraph.graph.nodes[inode].x - xc)
        ω = exp(-ρ/σ)
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
        return A, norm(A*M - N)^2, ω_tot
    catch
        @warn("Catching SingularException")
        A = (N*M') / (M*M' + 1e-6*I)
        return A, norm(A*M - N)^2, ω_tot
    end    
end