using Accessors
using DifferentialEquations
using Plots
using PlotThemes
using Plots.PlotMeasures
using DelimitedFiles
using LaTeXStrings
using Printf
using Colors
using Distributed
using ProgressMeter

# Test function : Helmholtz duffing
function helmholtz_duffing!(du, u, (a,b,c,d,e,ω), t)
    du[1] = u[2] 
    du[2] = -a*u[2] - b*u[1] - c*u[1]^2 - d*u[1]^3 + e*sin(ω*t)
end

function spring_column!(dx , x , (c, p, alpha, q, q1, omega), t)
    # State space description of the problem ∂X = A.X
    dx[1] = x[2]
    dx[2] = -c * x[2] + p * sin(x[1]) - (1 - 1 / sqrt(1 + alpha * sin(x[1])) - (q + q1 * sin(omega * t))) * cos(x[1])
end

function dynamics_nva!(dx, x , (delta, r, beta1, beta2, beta3, n), t )
    # Variables
    y       = x[1]
    theta   = x[2]
    yp      = x[3]
    thetap  = x[4]

    # Equations
    dx[1] = yp
    dx[2] = thetap
    dx[3] =  -r * (y + yp * beta2 - beta1 * thetap ^ 2 * cos(theta)  + beta1 * beta3 * thetap * sin(theta) + y * delta * sin(n * t)) / (r - beta1 * sin(theta)^2)    
    dx[4] = (-r * beta3 * thetap - y * sin(theta) - yp * beta2 * sin(theta) + beta1 * thetap ^2 * cos(theta) * sin(theta) - y * delta * sin(theta) * sin(n * t)) / (r - beta1 * sin(theta)^2)    
end

function double_helmholtz_duffing!(du, u, (a1,b1,c1,d1,e1,ω1,a2,b2,c2,d2,e2,ω2), t)
    du[1] = u[2] 
    du[2] = -a1*u[2] - b1*u[1] - c1*u[1]^2 - d1*u[1]^3 + e1*sin(ω1*t)
    du[3] = u[4] 
    du[4] = -a2*u[4] - b2*u[3] - c2*u[3]^2 - d2*u[3]^3 + e2*sin(ω2*t)
end

mutable struct BasinRegion
    range :: Vector{Vector{Float64}}
    elements :: Vector{Int64}
    is_cyclic :: Vector{Bool}
    extended_range :: Vector{Vector{Float64}}
end

mutable struct BasinProblem{F <: Function, P <: Union{Vector, Array}}
    f :: F
    params :: P
    period :: Float64
    transient_cycles :: Int64
    maximum_cycles :: Int64
    maximum_extended_cycles :: Int64
    attractor_tolerance :: Int64
    region :: BasinRegion
    threads :: Int64
end

@enum AttractorType begin
    none
    regular
    quasi_periodic
    divergent
end

mutable struct Attractor
    number :: Int64
    kind :: AttractorType
    points :: Vector{Vector{Float64}}
    finder_thread_id :: Int64
end

mutable struct BasinResult
    cells
    attractors :: Vector{Attractor}
    region :: BasinRegion
end

function populate_basins(bp :: BasinProblem) :: BasinResult
    # Total cells in the basin
    total_cells = prod(bp.region.elements)

    p = Progress(total_cells)
    # Single thread
    if bp.threads == 1
        # Map the cells for a single thread - no split necessary
        results = map_cells(bp, 1:total_cells, p)
        # Return the results - no join necessary
        return results
    # Multiple threads
    else
        # Divide the ranges for the especified number of threads
        cell_ranges = Iterators.partition(1:total_cells, ceil(Int64, total_cells / bp.threads))
        # Creates the tasks to be used (based on the number of threads)
        tasks = map(cr -> Threads.@spawn(map_cells(bp, cr, p)), cell_ranges)
        # Fetch the results of each thread
        results = map(t -> fetch(t), tasks)
        # Join the results basins up to one result
        return foldl(join_results!, results)
    end
end

function get_cell_number(u, region::BasinRegion) :: Int64
    # Initialize cell
    cell = 0

    # cell dimensions in each direction
    for i in eachindex(region.range)
        length = (region.range[i][2] - region.range[i][1]) / region.elements[i]
        index = clamp(ceil(Int64, (u[i] - region.range[i][1]) / length), 1, region.elements[i])
        cell = i == 1 ? index : cell + (index - 1) * prod(region.elements[1:i-1])
    end

    # Return cell
    return cell
end

function store_cell_center!(u :: Vector{Float64}, cell::Int64, region::BasinRegion)
    indexes = Vector{Int64}()

    pos = cell
    # First position
    pos_i = pos % region.elements[1] == 0 ? region.elements[1] : pos % region.elements[1]        
    push!(indexes, pos_i)

    for i = 2:lastindex(region.range)
        # Get position of generalized coordinate
        qi = pos - pos_i == 0 ? 1 : pos - pos_i
        # Update pos
        pos :: Int64 = floor(qi / region.elements[i - 1]) + 1
        pos_i = pos % region.elements[i] == 0 ? region.elements[i] : pos % region.elements[i]
        push!(indexes, pos_i)
    end

    # cell dimensions in each direction
    for i in eachindex(region.range)
        length = (region.range[i][2] - region.range[i][1]) / region.elements[i]
        u[i] = (indexes[i] - 0.5) * length + region.range[i][1]
    end
end

function is_inside_range(pos, range)
    # Declare in-line function for single dimension check
    is_inside(pos_i, range_i) = range_i[1] <= pos_i <= range_i[2]
    # Join the range checks. If all inside, returns true; returns false otherwise.
    return reduce(&, [is_inside(p, r) for (p, r) in zip(pos, range)])
end

function set_integrator!(integrator; u::Vector{Float64}=nothing, t::Float64=nothing)
    if u !== nothing
        set_u!(integrator, u)
        u_modified!(integrator, true)
    end

    t === nothing || set_t!(integrator, t)
end

function map_cells(bp :: BasinProblem, cell_range :: UnitRange{Int64}, p) :: BasinResult
    # Create the grid that will be populated in the function. Can be accessed by a tuple / list of values or single value
    grid = zeros(Int64, Tuple(x for x in bp.region.elements))

    # Create the attractors list for filling
    attractors = []
    attractors_found = 1 # we already know of the divergent one

    # temporary vector for storing indices of cells we visited in each iteration
    T = zeros(Int64, bp.maximum_cycles + 1) # + 1 to accommodate initial cell
    # temporary vector for cells of attractors found in a iteration
    A = zeros(Int64, 20)

    u = zeros(Float64, length(bp.region.elements))

    # initialize problem and integrator
    ode_problem = ODEProblem(
        bp.f,
        zero(u),
        (0.0, bp.period * bp.maximum_cycles),
        bp.params,
    )
    integrator = init(
        ode_problem;
        dense=false,
        save_everystep=false,
        save_start=false,
        maxiters=1e10,
    )

    for c in cell_range
        # Trajectory already filled
        if (grid[c] != 0)
            next!(p)
            continue
        end

        # # Set the initial value as the current cell
        store_cell_center!(u, c, bp.region)

        # Set integrator's initial condition. 2 options: first line requires
        # more memory but maxiters can be way smaller; second line uses half
        # of the memory but requires a large maxiters.
        # reinit!(integrator, u; t0=0.0)
        set_integrator!(integrator; u, t=0.0)
        
        # Search for the attractor
        attractor = search_for_attractor!(;
            bp,
            grid,
            integrator,
            next_attr_num=attractors_found,
            T_prealloc=T,
            A_prealloc=A,
            u=u
            #u=u0
        )

        # If attractor is found, save it in list for the results exporting
        if !isnothing(attractor)
            attractors_found += 1
            push!(attractors, attractor)
        end

        next!(p)

    end

    # Returns result
    BasinResult(grid, attractors, bp.region)
end

function make_attractor_from_cells(cells; number::Int64, type::AttractorType, region::BasinRegion)
    points = map(cells) do c
        u =  zeros(Float64, length(region.elements))
        store_cell_center!(u, c, region)
        return u
    end

    return Attractor(number, type, points, Threads.threadid())
end

function search_for_attractor!(; bp :: BasinProblem, grid, integrator, next_attr_num::Int64, 
                                T_prealloc::Vector{Int64}, A_prealloc::Vector{Int64}, u :: Vector{Float64})

    T = T_prealloc
    A = A_prealloc
    u0 = u

    T_len = A_len = 0   # how much we've filled the preallocated T and A vectors
    cycles = 0          # total number of cycles spent integrating
    extended_cycles = 0 # number of consecutive cycles spent in extended region
    
    T[T_len += 1] = get_cell_number(u0, bp.region)

    # skip transient 
    step!(integrator, bp.period * bp.transient_cycles, true)
    # Adjust in case of cyclic conditions
    adjust_cyclic(u, bp.region.range, bp.region.is_cyclic)
    
    set_integrator!(integrator; u, t=0.0)

    attr_number = -1 # number of the attractor we've found    
    attr_type = none # type of the attractor we've found

    found_new_attractor = false

    while cycles < bp.maximum_cycles
        if !is_inside_range(integrator.u, bp.region.range) # Is outside basin region
            if (!is_inside_range(integrator.u, bp.region.extended_range) # Is outside extended region
                || cycles > bp.maximum_cycles)
                attr_number = -1
                break
            end
            # Is in extended region
            extended_cycles += 1
        else # Is in the basin
            extended_cycles = 0
            cell = get_cell_number(integrator.u, bp.region)

            if grid[cell] != 0
                # bumped into a basin/attractor found in a previous call to search_for_attractor!
                attr_number = grid[cell]
                break
            end

            T[T_len += 1] = cell # push cell into trajectory

            # check previous occurrences of this cell 
            occurrences = count(c -> c == cell, @view T[1:(T_len - 1)])
            if occurrences == bp.attractor_tolerance
                # found a cell that's part of a pontential attractor
                if length(A) < A_len + 1
                    push!(A, cell)
                else
                    A[A_len+=1] = cell
                end
            elseif occurrences > bp.attractor_tolerance
                # assume trajectory started looping through the cells in A
                found_new_attractor = true
                attr_number = next_attr_num
                attr_type = regular
                break
            end
        end

        #u = step(bp.f, bp.params, u, bp.period)        
        step!(integrator, bp.period, true)
        # Adjust in case of cyclic conditions
        adjust_cyclic(u, bp.region.range, bp.region.is_cyclic)
        cycles += 1
    end

    if cycles == bp.maximum_cycles
        found_new_attractor = true
        attr_type = quasi_periodic
        attr_number = next_attr_num
        # in this case, the attractor is the whole trajectory:
        A = @view(T[ceil(Int64, T_len/2):T_len])
        A_len = ceil(Int64, T_len/2)
    end

    foreach(c -> grid[c] = attr_number, @view T[1:T_len])

    if found_new_attractor
        return make_attractor_from_cells(@view(A[1:A_len]), type=attr_type, number=attr_number, region=bp.region)
    else
        return nothing
    end
end

function join_results!(resultA::BasinResult, resultB::BasinResult)
    # maps attractor numbers in resultB to attractor numbers in resultA
    convert_dict = Dict{eltype(resultA.cells),eltype(resultA.cells)}(-1 => -1)

    # attractors that are not yet in resultA will start to be numbered from
    # `next_attr_num`
    next_attr_num = maximum(map(a -> a.number, resultA.attractors))

    for attr in resultB.attractors
        other_attr = findfirst(a -> Set(a.points) == Set(attr.points),
            resultA.attractors)

        if isnothing(other_attr) # resultA doesn't have this attractor
            convert_dict[attr.number] = (next_attr_num += 1)
            new_attr = @set attr.number = next_attr_num
            push!(resultA.attractors, new_attr)
        else
            convert_dict[attr.number] = resultA.attractors[other_attr].number
        end
    end

    for i in 1:length(resultA.cells)
        if iszero(resultA.cells[i]) && haskey(convert_dict, resultB.cells[i])
            resultA.cells[i] = convert_dict[resultB.cells[i]]
        end
    end

    return resultA
end

function adjust_cyclic(u :: Vector{Float64}, range:: Vector{Vector{Float64}}, is_cyclic :: Vector{Bool})

    function adjust_cyclic_single(u :: Float64; maximum:: Float64 = pi, minimum :: Float64 = pi)
        # Number of turns
        turns = fld(u, 2 * pi)

        # Calculates the new value removing the number of turns
        adjusted_u = u - turns * 2 * pi

        # Checks if it is within the region and returns the corrected value
        if adjusted_u > maximum
            return adjusted_u - 2 * pi
        end

        # Checks if it is within the region and returns the corrected value
        if adjusted_u < minimum
            return adjusted_u + 2 * pi
        end

        # Returns the corrected value
        return adjusted_u
    end

    for i in eachindex(u)
        if is_cyclic[i]
            u[i] = adjust_cyclic_single(u[i], minimum = range[i][1], maximum = range[i][2])
        end
    end
end

function plotbasins(result::BasinResult)
    theme(:juno)

    colors = [colorant"black", colorant"white", colorant"red",
    colorant"yellow", colorant"blue", colorant"green", 
    colorant"cyan", colorant"purple", colorant"brown", 
    colorant"pink", colorant"aquamarine4", colorant"goldenrod",
    colorant"ivory4", colorant"salmon2", colorant"midnightblue",
    colorant"magenta", colorant"rosybrown4", colorant"teal"]

    x1 = collect(LinRange(result.region.range[1][1], result.region.range[1][2], result.region.elements[1]))
    x2 = collect(LinRange(result.region.range[2][1], result.region.range[2][2], result.region.elements[2]))

    basin = transpose(reshape(result.cells,(result.region.elements[1], result.region.elements[2])))

    colorgrid = map(c -> colors[c + 2], basin)

    # plot heatmap
    xformatter(x) = string(round(x, digits=2))
    yformatter(y) = string(round(y, digits=2))

    plt = heatmap(
        x1,
        x2,
        colorgrid,
        size = (1100, 900),
        legend = :false,
        left_margin = [5mm 0mm],
        tickfontsize = 18, 
        framestyle = :box,
        xformatter = xformatter,
        yformatter = yformatter,
        yflip=false,
        aspect_ratio = :none,
    )

    # plot attractors
    for attr in result.attractors
        # Points
        u1 = map(u -> u[1], attr.points)
        u2 = map(u -> u[2], attr.points)

        # Add as crosses
        plot!(plt, u1, u2, seriestype = :scatter, 
                markershape = :cross, markercolor = :white, markersize = 16, legend = false)
    end

    # Set labels
    xlabel!(L"x_1(0)",xguidefontsize = 18)
    ylabel!(L"x_2(0)",yguidefontsize = 18)

    return plt
end

function plotbasins_slice(result::BasinResult, basin, i1 :: Int64, i2 :: Int64)
    theme(:juno)

    colors = [colorant"black", colorant"white", colorant"red",
    colorant"yellow", colorant"blue", colorant"green", 
    colorant"cyan", colorant"purple", colorant"brown", 
    colorant"pink", colorant"aquamarine4", colorant"goldenrod",
    colorant"ivory4", colorant"salmon2", colorant"midnightblue",
    colorant"magenta", colorant"rosybrown4", colorant"teal"]

    x1 = collect(LinRange(result.region.range[i1][1], result.region.range[i1][2], result.region.elements[i1]))
    x2 = collect(LinRange(result.region.range[i2][1], result.region.range[i2][2], result.region.elements[i2]))

    basin = transpose(reshape(basin,(result.region.elements[i1], result.region.elements[i2])))

    colorgrid = map(c -> colors[c + 2], basin)

    # plot heatmap
    xformatter(x) = string(round(x, digits=2))
    yformatter(y) = string(round(y, digits=2))

    plt = heatmap(
        x1,
        x2,
        colorgrid,
        size = (1100, 900),
        legend = :false,
        left_margin = [5mm 0mm],
        tickfontsize = 18, 
        framestyle = :box,
        xformatter = xformatter,
        yformatter = yformatter,
        yflip=false,
        aspect_ratio = :none,
    )

    # plot attractors
    for attr in result.attractors
        # Points
        u1 = map(u -> u[i1], attr.points)
        u2 = map(u -> u[i2], attr.points)

        # Add as crosses
        plot!(plt, u1, u2, seriestype = :scatter, 
                markershape = :cross, markercolor = :white, markersize = 16, legend = false)
    end

    # Set labels
    xlabel!(L"x_1(0)",xguidefontsize = 18)
    ylabel!(L"x_2(0)",yguidefontsize = 18)

    return plt
end

function bench_spring_column(
    divs; 
    omega = 0.0,
    nthreads=Threads.nthreads(),
    params = [0.01, 0.05, 0.8, 0.01, 0.05, omega],
)
    region = BasinRegion(
        [[-3.14, 3.14], [-1.3, 1.3]], 
        [divs,  divs],
        [true, false],
        [[-3.14, 3.14],[-2.0, 2.0]],
    )

    bp = BasinProblem(
        spring_column!, 
        params,
        7.853981634, 
        10, 
        1000, 
        20, 
        80, 
        region,
        nthreads,
    )

    println(">>> Running with $(nthreads) threads and $(bp.region.elements[1])x$(bp.region.elements[2]) grid")
    @time result = populate_basins(bp);
    return result
end

function bench_helmduff_simple(
    divs; 
    e=0.077, 
    #e = 0.02,
    nthreads=Threads.nthreads(),
    params = [0.1, -1.2, -0.3, 2.0, e, 1.17],
)
    region = BasinRegion(
        [[-1.2, 1.5], [-1.5, 1.5]], 
        [divs,  floor(UInt32,(3.0/2.7)*divs)],
        [false, false],
        [[-2.0, 2.5],[-2.0, 2.0]],
    )

    bp = BasinProblem(
        helmholtz_duffing!, 
        params,
        2*pi/1.17, 
        10, 
        1000, 
        20, 
        80, 
        region,
        nthreads,
    )

    println(">>> Running with $(nthreads) threads and $(bp.region.elements[1])x$(bp.region.elements[2]) grid")
    @time result = populate_basins(bp);
    return result
end

function bench_dynamics_nva(
    divs; 
    nthreads=Threads.nthreads(),
    params = [0.1, 0.35, 0.0375, 0.02, 0.2, 2],
)
    region = BasinRegion(
        [[-0.15, 0.15], [-0.25, 0.25], [-3.14, 3.14], [-0.5, 0.5]], 
        divs,
        [false, false, true, false],
        [[-0.30, 0.3],[-2.0, 2.0], [-3.14, 3.14], [-1.0, 1.0]],
    )

    bp = BasinProblem(
        dynamics_nva!, 
        params,
        3.1415926535897, 
        10, 
        1000, 
        20, 
        80, 
        region,
        nthreads,
    )

    println(">>> Running with $(nthreads) threads and $(bp.region.elements[1])x$(bp.region.elements[2])x$(bp.region.elements[3])x$(bp.region.elements[4]) grid")
    @time result = populate_basins(bp);
    return result
end

function bench_double_helmduff(
    divs; 
    e1=0.02, 
    e2 = 0.02,
    nthreads=Threads.nthreads(),
    params = [0.1, -1.2, -0.3, 2.0, e1, 1.17, 0.1, -1.2, -0.3, 2.0, e2, 1.17],
)
    region = BasinRegion(
        [[-1.2, 1.5], [-1.5, 1.5],[-1.2, 1.5], [-1.5, 1.5]], 
        [divs,  floor(UInt32,(3.0/2.7)*divs), divs,  floor(UInt32,(3.0/2.7)*divs)],
        [false, false, false, false],
        [[-2.0, 2.5],[-2.0, 2.0],[-2.0, 2.5],[-2.0, 2.0]],
    )

    bp = BasinProblem(
        double_helmholtz_duffing!, 
        params,
        2*pi/1.17, 
        10, 
        1000, 
        20, 
        80, 
        region,
        nthreads,
    )

    println(">>> Running with $(nthreads) threads and $(bp.region.elements[1])x$(bp.region.elements[2])x$(bp.region.elements[3])x$(bp.region.elements[4]) grid")
    @time result = populate_basins(bp);
    return result
end

function test_helmduff(divs)
    # Force garbage collector
    GC.gc()
    bench_helmduff_simple(10) # pre-compile run
    # Force garbage collector
    GC.gc()
    result = bench_helmduff_simple(divs; nthreads=Threads.nthreads())
    plt = plotbasins(result)
    display(plt)
    writedlm("basin_hd.csv", result.cells, ',')
end

function test_spring_column(divs; omega=0.0)
    # Force garbage collector
    GC.gc()
    bench_spring_column(10) # pre-compile run
    # Force garbage collector
    GC.gc()
    result = bench_spring_column(divs; nthreads=Threads.nthreads(), omega=omega)
    plt = plotbasins(result)
    display(plt)
    writedlm("basin_sc.csv", result.cells, ',')
end

function test_dynamics_nva(divs)
    # Force garbage collector
    GC.gc()
    bench_dynamics_nva([10, 10, 10, 10]) # pre-compile run
    # Force garbage collector
    GC.gc()
    result = bench_dynamics_nva([divs, divs, divs, divs]; nthreads=Threads.nthreads())
    plt = plotbasins_slice(result, result.cells[1,1,:,:], 3, 4)
    display(plt)
    writedlm("basin_dyn_nva.csv", result.cells, ',')
end

function test_double_helmduff(divs)
    # Force garbage collector
    GC.gc()
    bench_double_helmduff(10) # pre-compile run
    # Force garbage collector
    GC.gc()
    result = bench_double_helmduff(divs; nthreads=Threads.nthreads())
    # plt = plotbasins_slice(result, result.cells[1,1,:,:], 3, 4)
    plt = plotbasins_slice(result, result.cells[:,:,1,1], 1, 2)
    display(plt)
    writedlm("basin_double_hd.csv", result.cells, ',')
end

function assert_cells(divs; 
    e=0.077, 
    #e = 0.02,
    nthreads=Threads.nthreads(),
    params = [0.1, -1.2, -0.3, 2.0, e, 1.17],
)
    region = BasinRegion(
        [[-1.2, 1.5], [-1.5, 1.5]], 
        [divs,  floor(UInt32,(3.0/2.7)*divs)],
        [false, false],
        [[-2.0, 2.5],[-2.0, 2.0]],
    )

    bp = BasinProblem(
        helmholtz_duffing!, 
        params,
        2*pi/1.17, 
        10, 
        1000, 
        20, 
        80, 
        region,
        nthreads,
    )

    for c in 1:prod(bp.region.elements)
        u0 = zeros(Float64, length(bp.region.elements))
        store_cell_center!(u0, c, bp.region)
        cell = get_cell_number(u0, region)
        if cell != c 
            print("Difference! $(cell - c)")
        end
    end       
end

function run(; maxthreads=Threads.nthreads(), divrange=1000:1000:4000)
    bench_helmduff_simple(10) # pre-compile run

    for divs in divrange
        for t in maxthreads:-1:1
            bench_helmduff_simple(divs; nthreads=t)
        end
    end
end

# assert_cells(200)
#test_helmduff(200)
#test_spring_column(1000, omega=0.0)
#test_dynamics_nva(10)
#test_double_helmduff(50)
#run()
