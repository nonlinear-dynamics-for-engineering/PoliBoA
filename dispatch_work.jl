using Accessors
using DifferentialEquations

"""
    BasinRegion

Specifies the rectangle in which cell mapping will be executed

# Fields
- `u1range`: vector [a, b] where a is the min x coordinate and b is the max x
  coordinate of a point inside the region
- `u2range`: same as u1range but for y coordinate
- `u1cells`: number of columns (i.e. number of cells in the x direction)
- `u2cells`: number of rows (i.e. number of rows in the y direction)
- `u1extendedrange`: vector [a, b] where a is the min x coordinate and b is the
  max x coordinate of a point in the extended region
- `u2extendedrange`: same as u1extendedrange but for y coordinate
"""
struct BasinRegion
    u_range::Vector{Vector{Float64}}
    u_cells::Vector{Int64}
    u_extendedrange::Vector{Vector{Float64}}
end

"""
    BasinProblem

Describes a basin/attractor computation problem

# Fields
- `f`: differential equation that models the system. Will be called as 
  `f(du, u, params, t)` and should mutate du.
- `params`: parameters that will be passed as the third argument to `f` (see 
  above).
- `period`: period of the forcing component of f.
- `transient_cycles`: number of periods that are skipped at the beginning of each 
  cell mapping integration. Increasing this parameter too much **heavily** 
  affects performance and should be avoided (see also `att_tolerance`).
- `maxcycles`: max number of iterations before a trajectory is considered to be 
  attracted to a quasi-periodic attractor.
- `max_extended_cycles`: number of cycles a trajectory can spend in the extended 
  region before being considered part of the divergent attractor.
- `att_tolerance`: number of times a trajectory must repeatedly fall inside a 
  cell before that cell is considered to be part of an attractor. Increasing 
  this parameter helps to eliminate "duplicate" attractors that are actually 
  the same attractor found by different threads. Increasing it does not affect 
  performance noticeably.
- `region`: the region that will be populated.
- `nthreads`: number of threads the program will spawn to split the work.
"""
struct BasinProblem{F<:Function, P}
    f::F
    params::P 
    period::Float64
    transient_cycles::Int64
    maxcycles::Int64
    max_extended_cycles::Int64
    att_tolerance::Int64
    region::BasinRegion
    nthreads::Int64
end

"""
    AttractorType

Enum of attractor types

# Values
- `none`: not initalized. Only used internally.
- `regular`: regular periodic attractor.
- `quasiperiodic`: attractor that's not periodic (i.e. has a period greater than `bp.maxcycles`).
- `divergent`: attractor that escapes the main region.
"""
@enum AttractorType begin
    none
    regular
    quasiperiodic
    divergent
end

"""
    Attractor

Holds information about an attractor

# Fields
- `number`: number of the attractor.
- `kind`: `AttractorType` of this attractor.
- `points`: points that make up the attractor.
- `finder_id`: id of the thread that found this attractor
"""
struct Attractor
    number::Int64
    kind::AttractorType
    points::Vector{Vector{Float64}}
    finder_id::Int64
end

"""
    BasinResult

The result of a successful cell mapping

# Fields
- `cells`: vector containing the resulting basins
- `attractors`: list of the attractors that were discovered
- `region`: reference to the region that was originally used in the problem
"""
struct BasinResult
    cells::Vector{Int64}
    attractors::Vector{Attractor}
    region::BasinRegion
end

"""
    populate_basins(bp::BasinProblem)::BasinResult

Executes cell mapping for some basin problem.
"""
function populate_basins(bp::BasinProblem)::BasinResult
    #Number of total cells
    totalcells = prod(bp.region.u_cells)

    if bp.nthreads == 1
        results = mapcells(bp, 1:totalcells)
        return results
    else
        cellranges = Iterators.partition(1:totalcells, ceil(Int64, totalcells / bp.nthreads))
        tasks = map(cr -> Threads.@spawn(mapcells(bp, cr)), cellranges)
        results = map(t -> fetch(t), tasks)
        return foldl(join_results!, results)
    end
end

function is_inside_range(pos, u_range)
    for i in eachindex(u_range)
        if !(u_range[i][1] < pos[i] < u_range[i][2])
            return false
        end
    end

    return true
end

function get_cell_number(u, region::BasinRegion)
    #get_cell_number_original(u, region)
    get_cell_number_multi(u, region)
end
function store_cell_center!(u, cell::Int64, region::BasinRegion)
    #store_cell_center_original!(u, cell, region)
    store_cell_center_multi!(u, cell, region)
end

function get_cell_number_original(u, region::BasinRegion)
    # cell dimensions in each direction
    u1length = (region.u_range[1][2] - region.u_range[1][1]) / region.u_cells[1]
    u2length = (region.u_range[2][2] - region.u_range[2][1]) / region.u_cells[2]

    # indexes in the grid (clamp to avoid precision errors)
    u1index = clamp(ceil(Int64, (u[1] - region.u_range[1][1]) / u1length), 1, region.u_cells[1])
    u2index = clamp(ceil(Int64, (region.u_range[2][2] - u[2]) / u2length), 1, region.u_cells[2])

    return (u1index - 1) * region.u_cells[2] + u2index
end

function get_cell_number_multi(u, region::BasinRegion)
    # cell dimensions in each direction
    length = Vector{Float64}()
    for i in eachindex(region.u_range)
        length_i = (region.u_range[i][2] - region.u_range[i][1]) / region.u_cells[i]
        push!(length, length_i)
    end
    
    # Indexes array
    indexes = Vector{Int64}()

    # Collect the index for each generalized coordinate
    for i in eachindex(region.u_range)
        if u[i] < region.u_extendedrange[i][1] || u[i] > region.u_extendedrange[i][2]
            # Outside extended region
            return -1
        elseif u[i] < region.u_range[i][1] || u[i] > region.u_range[i][2]
            # Inside extended region but outside central region
            return 0    
        else
            # Index in the generalizec coordinate vector
            index = clamp(ceil(Int64, (u[i] - region.u_range[i][1]) / length[i]), 1, region.u_cells[i])
        end
        
        # Add to index vector
        push!(indexes, index)
    end

    # Position
    position = indexes[1]

    # Calculates position
    for i = 2:lastindex(region.u_range)
        position = position + (indexes[i] - 1) * prod(region.u_cells[1:i-1])
    end

    # Return calculated position
    return position
end

function store_cell_center_original!(u, cell::Int64, region::BasinRegion)
    # cell dimensions in each direction
    u1length = (region.u_range[1][2] - region.u_range[1][1]) / region.u_cells[1]
    u2length = (region.u_range[2][2] - region.u_range[2][1]) / region.u_cells[2]

    # indexes in the grid
    u1index = ceil(cell / region.u_cells[2])
    u2index = cell - (u1index - 1) * region.u_cells[2]

    u[1] = (u1index - 0.5) * u1length + region.u_range[1][1]
    u[2] = region.u_range[2][2] - (u2index - 0.5) * u2length
end

function store_cell_center_multi!(u, cell::Int64, region::BasinRegion)
    # cell dimensions in each direction
    length = Vector{Float64}()
    for i in eachindex(region.u_range)
        length_i = (region.u_range[i][2] - region.u_range[i][1]) / region.u_cells[i]
        push!(length, length_i)
    end

    # First position
    pos_i = cell % region.u_cells[1] == 0 ? region.u_cells[1] : cell % region.u_cells[1]    
    u[1] = (pos_i - 0.5) * length[1] + region.u_range[1][1]
    
    pos = cell
    
    for i = 2:lastindex(region.u_range)
        # Get position of generalized coordinate
        qi = pos - pos_i == 0 ? 1 : pos - pos_i
        # Update pos
        pos :: Int64 = floor(qi / region.u_cells[i - 1]) + 1
        pos_i = pos % region.u_cells[i] == 0 ? region.u_cells[i] : pos % region.u_cells[i]
        u[i] = (pos_i - 0.5) * length[i] + region.u_range[i][1]
    end
end

function set_integrator!(integrator; u::Vector{Float64}=nothing, t::Float64=nothing)
    if u !== nothing
        set_u!(integrator, u)
        u_modified!(integrator, true)
    end

    t === nothing || set_t!(integrator, t)
end

function mapcells(bp::BasinProblem, cellrange::UnitRange{Int64})::BasinResult
    # grid that will be populated 
    grid = zeros(Int64, prod(bp.region.u_cells))

    # temporary vector for storing indices of cells we visited in each iteration
    T = zeros(Int64, bp.maxcycles + 1) # + 1 to accommodate initial cell
    # temporary vector for cells of attractors found in a iteration
    A = zeros(Int64, 20)

    attractors_found = 1 # we already know of the divergent one
    u = Array{Float64}(undef, length(bp.region.u_cells))

    # initialize problem and integrator
    ode_problem = ODEProblem(
        bp.f,
        zero(u),
        (0.0, bp.period * bp.maxcycles),
        bp.params,
    )
    integrator = init(
        ode_problem;
        dense=false,
        save_everystep=false,
        save_start=false,
        maxiters=1e10,
    )

    result = BasinResult(grid, [], bp.region)

    for c in cellrange
        if (grid[c] != 0)
            continue
        end

        store_cell_center!(u, c, bp.region)

        # Set integrator's initial condition. 2 options: first line requires
        # more memory but maxiters can be way smaller; second line uses half
        # of the memory but requires a large maxiters.
        # reinit!(integrator, u; t0=0.0)
        set_integrator!(integrator; u, t=0.0)

        attractor = search_for_attractor!(;
            grid,
            integrator,
            next_attr_num=attractors_found,
            bp,
            T_prealloc=T,
            A_prealloc=A,
            u_prealloc=u
        )

        if !isnothing(attractor)
            attractors_found += 1
            push!(result.attractors, attractor)
        end
    end

    return result
end

function make_attractor_from_cells(
    cells;
    number::Int64,
    type::AttractorType,
    region::BasinRegion
)
    points = map(cells) do c
        u = Array{Float64}(undef, length(region.u_cells))
        store_cell_center!(u, c, region)
        return u
    end

    return Attractor(number, type, points, Threads.threadid())
end

function search_for_attractor!(;
    grid::Vector{Int64},
    integrator,
    next_attr_num::Int64,
    bp::BasinProblem,
    T_prealloc::Vector{Int64},
    A_prealloc::Vector{Int64},
    u_prealloc::Vector{Float64}
)
    T = T_prealloc
    A = A_prealloc
    u = u_prealloc

    T_len = A_len = 0   # how much we've filled the preallocated T and A vectors
    cycles = 0          # total number of cycles spent integrating
    extended_cycles = 0 # number of consecutive cycles spent in extended region

    T[T_len+=1] = get_cell_number(integrator.u, bp.region)

    # skip transient 
    step!(integrator, bp.period * bp.transient_cycles, true)

    # start from the center of the cell we ended at 
    store_cell_center!(u, get_cell_number(integrator.u, bp.region), bp.region)
    set_integrator!(integrator; u, t=0.0)

    attr_number = -1 # number of the attractor we've found
    attr_type = none # type of the attractor we've found

    found_new_attractor = false

    while cycles < bp.maxcycles
        if !is_inside_range(integrator.u, bp.region.u_range)
            if (!is_inside_range(integrator.u, bp.region.u_extendedrange)
                ||
                cycles > bp.maxcycles)
                attr_number = -1
                break
            end
            extended_cycles += 1
        else # u is inside main region
            extended_cycles = 0
            cell = get_cell_number(integrator.u, bp.region)

            if grid[cell] != 0
                # bumped into a basin/attractor found in a previous call to search_for_attractor!
                attr_number = grid[cell]
                break
            end

            T[T_len+=1] = cell # push cell into trajectory

            # check previous occurrences of this cell 
            occurrences = count(c -> c == cell, @view T[1:(T_len-1)])
            if occurrences == bp.att_tolerance
                # found a cell that's part of a pontential attractor
                if length(A) < A_len + 1
                    push!(A, cell)
                else
                    A[A_len+=1] = cell
                end
            elseif occurrences > bp.att_tolerance
                # assume trajectory started looping through the cells in A
                found_new_attractor = true
                attr_number = next_attr_num
                attr_type = regular
                break
            end
        end

        step!(integrator, bp.period, true)
        cycles += 1
    end

    if cycles == bp.maxcycles
        found_new_attractor = true
        attr_type = quasiperiodic
        attr_number = next_attr_num
        # in this case, the attractor is the whole trajectory:
        A = @view(T[floor(Int64, T_len/2):T_len])
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