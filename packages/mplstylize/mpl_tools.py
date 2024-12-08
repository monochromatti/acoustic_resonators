def ylim_breathe(ax, factor=0.1):
    ylim = ax.get_ylim()
    yticks = ax.get_yticks()
    s = (ylim[1] - ylim[0]) * factor
    ax.set_yticks(yticks)
    ax.set_ylim(ylim[0] - s, ylim[1] + s)


def xlim_breathe(ax, factor=0.1):
    xlim = ax.get_xlim()
    xticks = ax.get_xticks()
    s = (xlim[1] - xlim[0]) * factor
    ax.set_xticks(xticks)
    ax.set_xlim(xlim[0] - s, xlim[1] + s)


def make_iterable(x):
    try:
        return list(iter(x))
    except TypeError:
        return (x,)


def breathe_axes(axes, axis="both", factor=0.1):
    for ax in make_iterable(axes):
        match axis:
            case "both":
                xlim_breathe(ax, factor=factor)
                ylim_breathe(ax, factor=factor)
            case "x":
                xlim_breathe(ax, factor=factor)
            case "y":
                ylim_breathe(ax, factor=factor)
            case _:
                raise ValueError(f"Unknown axis {axis}")


def enumerate_axes(axes, loc=(0, 1), offset=(-10, 10), **kwargs):
    for i, ax in enumerate(make_iterable(axes)):
        ax.annotate(
            f"{chr(97 + i)}",
            loc,
            xytext=offset,
            textcoords="offset points",
            xycoords="axes fraction",
            va="center",
            ha="center",
            weight="bold",
            **kwargs,
        )

def square_axes(axes):
    for ax in make_iterable(axes):
        ax.set_box_aspect(1)