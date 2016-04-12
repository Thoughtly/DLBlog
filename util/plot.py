import matplotlib.pyplot as plot

# Simple 2d scatter plot of samples vs y OR samples[0] vs samples[1]
def scatter(samples, y=None, color=None):

    # if samples is 1D then we assume there is a valid y and set x to samples
    if samples.shape[1] == 1:
        x = samples

    # a 2D sample means x is the first dimension and y is the second dimension
    elif samples.shape[1] == 2:
        x = [v for (v,_) in samples]
        y = [v for (_,v) in samples]

    # Force the plot to be square or the aspect ratio of plotted content looks wrong
    plot_max = max(max(x), max(y))
    plot_min = min(min(x), min(y))

    plot.xlim(plot_min - .1*plot_max, 1.1*plot_max)
    plot.ylim(plot_min - .1*plot_max, 1.1*plot_max)

    plot.scatter(x, y, color=color)
    plot.show()
