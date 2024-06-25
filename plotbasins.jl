using Plots
using PlotThemes
using Plots.PlotMeasures
using DelimitedFiles
using LaTeXStrings
using Printf
using Colors

include("dispatch_work.jl")

function plotbasins(result::BasinResult)
    theme(:juno)
    colors = [colorant"black", colorant"white", colorant"red",
    colorant"yellow", colorant"blue", colorant"green", colorant"cyan",
    colorant"purple", colorant"brown"]

    colorgrid = map(c -> colors[c + 2], reshape(result.cells, (result.region.u_cells[2], result.region.u_cells[1])))

    # plot heatmap
    function xformatter(x)
        tmp = [0.0,0.0]
        store_cell_center!(tmp, Int64(result.region.u_cells[2]*x), result.region)
        xcenter = tmp[1]
        return string(round(xcenter, digits=1))
    end

    function yformatter(y)
        tmp = [0.0,0.0]
        store_cell_center!(tmp, Int64(y), result.region)
        ycenter = tmp[2]
        return string(round(ycenter, digits=1))
    end

    plt = heatmap(
        colorgrid,
        size = (1100, 900),
        legend = :false,
        left_margin = [5mm 0mm],
        tickfontsize = 18, 
        framestyle = :box,
        xformatter = xformatter,
        yformatter = yformatter,
        yflip=true)

    # plot attractors
    for attr in result.attractors
        attr_points = attr.points

        # cell dimensions in each direction
        u_length =
        [
            (result.region.u_range[1][2]-result.region.u_range[1][1])/result.region.u_cells[1],
            (result.region.u_range[2][2]-result.region.u_range[2][1])/result.region.u_cells[2]
        ]

        # indexes in the grid
        u1indices = map(u->ceil(Int64, (u[1] - result.region.u_range[1][1])/u_length[1]),
                        attr_points)
        u2indices = map(u->ceil(Int64, (result.region.u_range[2][2] - u[2])/u_length[2]),
                        attr_points)

        foreach((x,y) -> annotate!(plt, x, y, text("$(attr.number)", 8)),
                                   u1indices, u2indices)
        #plot!(plt, u1indices, u2indices, seriestype = :scatter, markershape = :cross, markercolor = :white, markersize = 12, legend = false)
   end
    
    # axis labels
    xlabel!(L"x_1(0)",xguidefontsize = 18)
    ylabel!(L"x_2(0)",yguidefontsize = 18)

    return plt
end
