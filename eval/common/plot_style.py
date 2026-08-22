import matplotlib.pyplot as plt

# IEEE two-column paper: 3.5 in wide single-col, 7.16 in double-col
IEEE_SINGLE = (3.5, 2.6)
IEEE_DOUBLE = (7.16, 3.0)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.usetex": True,           # needs texlive-latex-extra
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Colorblind-safe palette (Wong 2011) — reviewers will flag red/green
PALETTE = {
    "baseline":   "#000000",   # paper's 30 s fixed / star topology
    "enhanced":   "#0072B2",   # your adaptive / mesh
    "flood":      "#D55E00",
    "aodv":       "#009E73",
    "gradient":   "#CC79A7",
    "sos":        "#E69F00",
    "ctrl":       "#56B4E9",
    "telem":      "#0072B2",
    "audio":      "#009E73",
    "bulk":       "#999999",
}
