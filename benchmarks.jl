include("plotbasins.jl")
include("dispatch_work.jl")

function helmduff!(du, u, (a,b,c,d,e,ω), t)
    du[1] = u[2] 
    du[2] = -a*u[2] - b*u[1] - c*u[1]^2 - d*u[1]^3 + e*sin(ω*t)
end

function bench_helmduff_simple(
    divs; 
    #e=0.077, 
    e = 0.02,
    nthreads=Threads.nthreads(),
    params = [0.1, -1.2, -0.3, 2.0, e, 1.17],
)
    region = BasinRegion([[-1.2, 1.5], [-1.5, 1.5]], 
        [divs,floor(UInt32,(3.0/2.7)*divs)],
        [[-2.0, 2.5], [-2.0, 2.0]],
    )

    bp = BasinProblem(
        helmduff!, 
        params,
        2*pi/1.17, 
        10, 
        1000, 
        20, 
        80, 
        region,
        nthreads,
    )

    println(">>> Running with $(nthreads) threads and $(bp.region.u_cells[1])x$(bp.region.u_cells[2]) grid")
    #@time result = populate_basins(bp);
    result = populate_basins(bp)
    return result
end

function run(; maxthreads=Threads.nthreads(), divrange=1000:1000:4000)
    bench_helmduff_simple(10) # pre-compile run

    for divs in divrange
        for t in maxthreads:-1:1
            bench_helmduff_simple(divs; nthreads=t)
        end
    end
end

function test(divs)
    result = bench_helmduff_simple(divs; nthreads=Threads.nthreads())
    println(result.cells)
    plotbasins(result)
end

# run()
test(100)