function plot_reconstruction_2d(
    X::AbstractMatrix,
    X̃::AbstractMatrix;
    X_test = X[end+1:end,:],
    X̃_test = X̃[end+1:end,:]
)
    xlimit = axis_rescale(vcat(X[:, 1], X_test[:, 1]), vcat(X̃[:, 1], X̃_test[:, 1]))
    ylimit = axis_rescale(vcat(X[:, 2], X_test[:, 2]), vcat(X̃[:, 2], X̃_test[:, 2]))
    standard_colors = theme_palette(:auto)
    @assert size(X, 2) == size(X̃, 2) == 2
    fig = plot(X[:, 1], X[:, 2], label = "true", legend = true, xlims = xlimit, ylims = ylimit)
    plot!(fig, X̃[:, 1], X̃[:, 2], label = "generated", xlabel = "x", ylabel = "y")
    isempty(X_test) ? nothing : plot!(fig, X_test[:, 1], X_test[:, 2], label = false, color=standard_colors[1], linestyle=:dash)
    isempty(X̃_test) ? nothing : plot!(fig, X̃_test[:, 1], X̃_test[:, 2], label = false, color=standard_colors[2], linestyle=:dash)
    return fig
end

function plot_reconstruction_3d(
    X::AbstractMatrix,
    X̃::AbstractMatrix;
    X_test = X[end+1:end,:],
    X̃_test = X̃[end+1:end,:]
)
    xlimit = axis_rescale(vcat(X[:, 1], X_test[:, 1]), vcat(X̃[:, 1], X̃_test[:, 1]))
    ylimit = axis_rescale(vcat(X[:, 2], X_test[:, 2]), vcat(X̃[:, 2], X̃_test[:, 2]))
    zlimit = axis_rescale(vcat(X[:, 3], X_test[:, 3]), vcat(X̃[:, 3], X̃_test[:, 3]))
    standard_colors = theme_palette(:auto)
    @assert size(X, 2) == size(X̃, 2) == 3
    fig = plot(X[:, 1], X[:, 2], X[:, 3], label = "true", legend = true, xlims = xlimit, ylims = ylimit, zlims = zlimit)
    plot!(
        fig,
        X̃[:, 1],
        X̃[:, 2],
        X̃[:, 3],
        label = "generated",
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
    )
    isempty(X_test) ? nothing : plot!(fig, X_test[:, 1], X_test[:, 2], X_test[:, 3], label = false, color=standard_colors[1], linestyle=:dash)
    isempty(X̃_test) ? nothing : plot!(fig, X̃_test[:, 1], X̃_test[:, 2], X̃_test[:, 3], label = false, color=standard_colors[2], linestyle=:dash)
    return fig
end

function plot_reconstruction_series(
    X::AbstractMatrix,
    X̃::AbstractMatrix;
    X_test = X[end+1:end,:],
    X̃_test = X̃[end+1:end,:],
    cutl = 0,
    cutr = 0
)
    @assert size(X, 2) == size(X̃, 2) == 1
    t = 1:size(X, 1)
    t_test = (size(X, 1) + 1):(size(X, 1) + size(X_test,1))
    standard_colors = theme_palette(:auto)
    fig = plot(t, X[:, 1], label = "true", legend = true)
    plot!(fig, t, X̃[:, 1], label = "generated", xlabel = "t", ylabel = "a.u.")
    isempty(X_test) ? nothing : plot!(fig, t_test, X_test[:, 1], label = false, color=standard_colors[1], linestyle=:dash)
    isempty(X̃_test) ? nothing : plot!(fig, t_test, X̃_test[:, 1], label = false, color=standard_colors[2], linestyle=:dash)
    return fig
end

function plot_reconstruction_multiple_series(
    X::AbstractMatrix,
    X̃::AbstractMatrix,
    n_plots::Int;
    X_test = X[end+1:end,:],
    X̃_test = X̃[end+1:end,:],
    cutl = 0,
    cutr = 0
)
    @assert size(X, 2) == size(X̃, 2) >= n_plots
    t_deconv = (1 + cutl):(size(X̃, 1) + cutl)
    t_test = (size(X, 1) + 1):(size(X, 1) + size(X_test,1))
    t_deconv_test = (1 + size(X̃, 1) + cutl - cutr):(size(X̃, 1) + cutl - cutr + size(X̃_test, 1))
    ps = []
    standard_colors = theme_palette(:auto)
    for i = 1:n_plots
        ticks = i == n_plots ? true : false
        legend = i == 1 ? true : false
        ylimit = axis_rescale(vcat(X[:, i], X_test[:, i]), vcat(X̃[:, i], X̃_test[:, i]))
        p = plot(
            X[:, i],
            label = "true",
            legend = legend,
            xticks = ticks,
            yticks = false,
            ylims = ylimit
        )
        plot!(p, t_deconv, X̃[:, i], label = "generated")
        isempty(X_test) ? nothing : plot!(p, t_test, X_test[:, i], label = false, color=standard_colors[1], linestyle=:dash)
        isempty(X̃_test) ? nothing : plot!(p, t_deconv_test, X̃_test[:, i], label = false, color=standard_colors[2], linestyle=:dash)
        push!(ps, p)
    end
    plts = (ps[i] for i = 1:n_plots)
    fig = plot(plts..., layout = (n_plots, 1), link = :all)
    return fig
end

function plot_reconstruction(
    X_gen_cpu::AbstractMatrix,
    X_cpu::AbstractMatrix,
    save_path::String;
    X̃_test = X_gen_cpu[end+1:end,:],
    X_test = X_cpu[end+1:end,:],
    cutl = 0,
    cutr = 0 
) 
    if size(X_cpu, 2) == 3
        fig = plot_reconstruction_3d(X_cpu, X_gen_cpu; X_test, X̃_test)
    elseif size(X_cpu, 2) == 2
        fig = plot_reconstruction_2d(X_cpu, X_gen_cpu; X_test, X̃_test)
    elseif size(X_cpu, 2) == 1
        fig = plot_reconstruction_series(X_cpu, X_gen_cpu; X_test, X̃_test, cutl, cutr)
    elseif size(X_cpu, 2) >= 3
        n_plots = size(X_cpu, 2) > 5 ? 5 : size(X_cpu, 2)
        fig = plot_reconstruction_multiple_series(X_cpu, X_gen_cpu, n_plots; X_test, X̃_test, cutl, cutr)
    end
    savefig(fig, save_path)
end

function axis_rescale(X::AbstractVector, X̃::AbstractVector; factor = 0.5, bound = 0.05)
    ymin1, ymax1 = extrema(filter!(!isnan, X))
    ymin2, ymax2 = extrema(filter!(!isnan, X̃))
    diff1 = ymax1 - ymin1
    ymin = (ymin1 < ymin2 ? ymin1 : maximum([ymin2, ymin1 - factor*diff1]))
    ymax = (ymax1 > ymax2 ? ymax1 : minimum([ymax2, ymax1 + factor*diff1]))
    diff = ymax - ymin
    ymin_bound = ymin - bound*diff
    ymax_bound = ymax + bound*diff
    return ymin_bound, ymax_bound
end

function plot_loss(data::AbstractDataFrame, save_path::String)
    CSV.write(joinpath(save_path, "LossMetrics.csv"), data, delim = "\t")
    p1 = plot(data[:,:Epoch], data[:, :Loss], title = "Loss", yscale=:log10)
    p2 = plot(data[:,:Epoch], data[:, :PE_train], title = "Prediction error (train)", yscale=:log10)
    p3 = plot(data[:,:Epoch], data[:, :PE_test], title = "Prediction error (test)", yscale=:log10)
    p4 = plot(data[:,:Epoch], data[:, :D_stsp], title = "State space divergence")
    p5 = plot(data[:,:Epoch], data[:, :PSE], title = "Power spectrum error")
    fig = plot(p1, p2, p3, p4, p5, layout=(5,1), ylabel="loss", size = (500,800), legend=false)
    savefig(fig, joinpath(save_path, "Loss.png")) 
end
