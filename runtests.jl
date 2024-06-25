using Test
using PoliBoA
using DifferentialEquations
using Distributed

@testset "cell to u and vice-versa" begin
    u1range = [0.0, 1.0]
    u2range = [0.0, 1.0]
    u1cells = 2
    u2cells = 2
    u1extendedrange = [-1.0, 2.0]
    u2extendedrange = [-1.0, 2.0]
    region = BasinRegion(u1range, u2range, u1cells, u2cells,
        u1extendedrange, u2extendedrange)

    @test PoliBoA.cell_from_u([0.7, 0.2], region) == 4
    @test PoliBoA.u_center_from_cell(1, region) == [0.25, 0.75]
end

@testset "search_for_attractor! regular, size 1 attractor" begin
    # see if seatch_for_attractor! can find an already known attractor
    prob = PoliBoA.sample_helmduff_prob(0.020)
    reg = BasinRegion([-1.2, 1.5], [-1.5, 1.5], 100, 100, [-2.0, 2.5], [-2.0, 2.0])

    trajectory = zeros(Int64, prob.maxcycles + 1)
    attractors = zeros(Int64, 20)
    grid = zeros(Int64, reg.u1cells, reg.u2cells)
    initialcell = 100

    trajlen, attrsz, attrnum, attrkind = PoliBoA.search_for_attractor!(trajectory, attractors, initialcell, 
        grid, prob, reg)

    @test attrsz == 1
    @test attrnum == initialcell*2
    @test attrkind == PoliBoA.regular
    #@test attractors[1] == PoliBoA.cell_from_u([-0.736364, -0.0151515], reg)
end

@testset "search_for_attractor! in extended region" begin
    #= same as previous test, but u2range was cut in half and attractor is no
       longer in main region =#
    prob = PoliBoA.sample_helmduff_prob(0.020)
    reg = BasinRegion([-1.2, 1.5], [-1.5, -0.1], 100, 100, [-2.0, 2.5], [-2.0, 2.0])

    trajectory = zeros(Int64, prob.maxcycles + 1)
    attractors = zeros(Int64, 20)
    grid = zeros(Int64, reg.u1cells, reg.u2cells)
    initialcell = 100

    trajlen, attrsz, attrnum, attrkind = PoliBoA.search_for_attractor!(trajectory, attractors, initialcell, 
        grid, prob, reg)

    @test attrsz == 0
    @test attrnum == 0
    @test attrkind == PoliBoA.divergent
end

@testset "search_for_attractor! outside extended region" begin
    #= same as previous test, but u2range and u2extendedrange are smaller and attractor
       is not even inside extended region =#
    prob = PoliBoA.sample_helmduff_prob(0.020)
    reg = BasinRegion([-1.2, 1.5], [-1.5, -1.0], 100, 100, [-2.0, -0.5], [-2.0, 2.0])

    trajectory = zeros(Int64, prob.maxcycles + 1)
    attractors = zeros(Int64, 20)
    grid = zeros(Int64, reg.u1cells, reg.u2cells)
    initialcell = 100

    trajlen, attrsz, attrnum, attrkind = PoliBoA.search_for_attractor!(trajectory, attractors, initialcell, 
        grid, prob, reg)

    @test attrsz == 0
    @test attrnum == 0
    @test attrkind == PoliBoA.divergent
end

@testset "search_for_attractor! but exceed maxcycles" begin
    # prob with really small maxcycles
    prob = BasinProblem(
        PoliBoA.make_helmholtz_duffing_integrator([0.1,-1.2,-0.3, 2,0.020,1.17]),
        2*pi/1.17, 100, 4, 20,5)
    reg = BasinRegion([-1.2, 1.5], [-1.5, 1.5], 100, 100, [-2.0, 2.5], [-2.0, 2.0])

    trajectory = zeros(Int64, prob.maxcycles + 1)
    attractors = zeros(Int64, 20)
    grid = zeros(Int64, reg.u1cells, reg.u2cells)
    initialcell = 100

    trajlen, attrsz, attrnum, attrkind  = PoliBoA.search_for_attractor!(trajectory, attractors, initialcell, 
        grid, prob, reg)

    @test attrnum == -200
    @test attrkind == PoliBoA.quasiperiodic
end

addprocs(2, exeflags="--project=$(Base.active_project())")
@everywhere using PoliBoA

@testset "next_cell(jobs) with non-empty and then empty jobs" begin
    jobs = RemoteChannel(()->Channel{Int64}(32))    
    worker1 = rand(workers())
    worker2 = rand(filter(w->w!=worker1, workers()))

    put!(jobs, 1)
    fut1 = remotecall(PoliBoA.next_cell, worker1, jobs)
    fut2 = remotecall(PoliBoA.next_cell, worker2, jobs)
    close(jobs)

    result1 = fetch(fut1)
    result2 = fetch(fut2)

    @test (result1 == 0 && result2 == 1) || (result1 == 1 && result2 == 0)
end

@testset "worker_populate_basins!" begin
    jobs = RemoteChannel(()->Channel{Int64}(32))    
    results = RemoteChannel(()->Channel{PoliBoA.BasinResult{Float64,Int64}}(32))    
    prob = PoliBoA.sample_helmduff_prob(0.020)
    reg = BasinRegion([-1.2, 1.5], [-1.5, 1.5], 100, 100, [-2.0, 2.5], [-2.0, 2.0])
    worker = rand(workers())

    put!(jobs, 10)
    future = @spawnat worker PoliBoA.worker_populate_basins!(prob, reg, jobs, results)
    close(jobs)

    wait(future) # sync with worker
    @test isready(results)
end

@testset "dispatch_work" begin
    prob = PoliBoA.sample_helmduff_prob(0.020)
    reg = BasinRegion([-1.2, 1.5], [-1.5, 1.5], 100, 100, [-2.0, 2.5], [-2.0, 2.0])
    results = PoliBoA.dispatch_work(prob, reg)

    s = sum(map(r -> r.grid, results))

    @test length(results) == nworkers()
    @test isempty(filter(iszero, s)) # make sure every cell was filled
end

@testset "join_results!" begin 
    prob = PoliBoA.sample_helmduff_prob(0.020)
    reg = BasinRegion([-1.2, 1.5], [-1.5, 1.5], 100, 100, [-2.0, 2.5], [-2.0, 2.0])
    results = PoliBoA.dispatch_work(prob, reg)
    joined = foldl(PoliBoA.join_results!, results)

    @test length(joined.attractors_dict) == 2
    @test isempty(filter(iszero, joined.grid))
    @test length(Set(joined.grid)) == 4 # traj number and attr number for each attractor
end

@testset "populate_basins" begin
    prob = PoliBoA.sample_helmduff_prob(0.020)
    reg = BasinRegion([-1.2, 1.5], [-1.5, 1.5], 100, 100, [-2.0, 2.5], [-2.0, 2.0])
    result = populate_basins(prob,reg)
end
