import json
from importlib.resources import files
from pathlib import Path

from IPython.display import display
from matplotlib import colormaps as cmaps
from matplotlib import font_manager, style
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.colors import to_rgb, to_rgba

# Add fonts
for fontpath in (Path(__file__).parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(fontpath)

style.use(Path(__file__).parent / "stylesheet.mplstylize")


# Define classes
class Color:
    def __init__(self, name, hex):
        self.name = name
        self.hex = hex

    @property
    def rgb(self):
        return to_rgb(self.hex)

    @property
    def rgba(self):
        return to_rgba(self.hex)

    def _repr_html_(self):
        html = "<table><tr>"
        html += (
            f'<td style="background-color:{self.hex}; width:20px; height:20px;"></td>'
        )
        html += f"<td>{self.name}</td>"
        html += "</tr></table>"
        return html


class ColorCollection:
    def __init__(self, colors: dict):
        self.colors = colors

    def _repr_html_(self):
        html = "<table>"
        for color in self.colors.values():
            html += "<tr>"
            html += f'<td style="background-color:{color.hex}; width:20px; height:20px;"></td>'
            html += f"<td>{color.name}</td>"
            html += "</tr>"
        html += "</table>"
        return html

    def __call__(self, *args):
        if len(args) > 1:
            return [self.colors[arg].hex for arg in args]
        return self.colors[args[0]].hex

    def __getattr__(self, name):
        if name in self.colors:
            return self.colors[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class ColorMap:
    def __init__(self, name, hex_values):
        self.name = name
        self.hex_values = hex_values

    @property
    def cmap(self):
        return LSC.from_list(self.name, self.hex_values)

    def register(self):
        try:
            cmaps.register(name=self.cmap.name, cmap=self.cmap)
        except ValueError:
            pass

    def _repr_html_(self):
        return self.cmap._repr_html_()


class ColorMapCollection:
    def __init__(self, color_maps: dict):
        self.color_maps = color_maps

    def _repr_html_(self):
        # Show the cmap._repr_html_() for each cmap in the collection, in a single-column table
        html = "<table>"
        for cmap in self.color_maps.values():
            html += "<tr>"
            html += f"<td>{cmap._repr_html_()}</td>"
            html += "</tr>"
        html += "</table>"
        return html

    def __call__(self, *args):
        if len(args) > 1:
            return [self.color_maps[arg].cmap for arg in args]
        return self.color_maps[args[0]].cmap

    def __getattr__(self, name):
        if name in self.color_maps:
            return self.color_maps[name].hex
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


# Load resources
def load_resource(file_name):
    resource_path = files("mplstylize").joinpath(file_name)
    with resource_path.open("r", encoding="utf-8") as file:
        return json.load(file)


# Load color maps
color_data = load_resource("cmaps.json")
color_maps = {
    color["name"]: ColorMap(color["name"], color["hex_values"]) for color in color_data
}

for cmap in color_maps.values():
    cmap.register()

    cmap_reverse = ColorMap(cmap.name + "_r", cmap.hex_values[::-1])
    cmap_reverse.register()

color_maps = ColorMapCollection(color_maps)

# Load colors
color_data = load_resource("colors.json")
colors = ColorCollection(
    {color["name"]: Color(color["name"], color["hex"]) for color in color_data}
)


def cmap_options():
    """Prints name of color, and displays color swatch, in a table."""
    for cmap in color_maps.values():
        display(cmap)


def get_cmap(name):
    """Returns a color object from its name."""
    if name.endswith("_r"):
        return color_maps[name[:-2]].cmap_r
    return color_maps[name].cmap


def color_options():
    """Prints name of color, and displays color swatch, in a table."""
    for color in colors.values():
        display(color)
