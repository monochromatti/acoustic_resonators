from pathlib import Path

from matplotlib.cm import register_cmap
from matplotlib.colors import LinearSegmentedColormap


class ProjectPaths:
    def __init__(self, root_path: str):
        self.root: Path = Path(root_path)

    @property
    def raw_data(self):
        return self.root / "raw_data"

    @property
    def processed_data(self):
        return self.root / "processed_data"

    @property
    def figures(self):
        return self.root / "figures"
    
    @property 
    def file_lists(self):
        return self.root / "file_lists"


paths = ProjectPaths(
    "/Users/monochromatti/University/acoustic_resonators/src/acoustic_resonators/"
)


def register_colors():
    colors_dict = {
        "RdOr": [
            "#7f1919",
            "#b12927",
            "#d55245",
            "#ee7f66",
            "#fca37a",
            "#fdc69a",
            "#fddbb4",
            "#feecd2",
            "#fff9f0",
        ],
        "OrBu": [
            "#d0b29d",
            "#c58770",
            "#b25d58",
            "#98384f",
            "#6f204c",
            "#5b1b64",
            "#562a87",
            "#5e52ab",
            "#677aba",
            "#7e9ec3",
            "#a6bfcb",
        ],
    }
    for name, colors in colors_dict.items():
        cmap = LinearSegmentedColormap.from_list(name, colors)
        cmap_r = LinearSegmentedColormap.from_list(name + "_r", colors[::-1])
        try:
            register_cmap(name=cmap.name, cmap=cmap)
            register_cmap(name=cmap_r.name, cmap=cmap_r)
        except ValueError:
            pass
