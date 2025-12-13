import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from io import StringIO

DATA = """resize,epochs,batch_size,best_val_acc,best_val_loss,best_epoch
64,10,64,0.2660919540229885,2.089479509441332,9
64,10,128,0.2655172413793103,2.069231123759829,9
64,10,256,0.26091954022988506,2.0863036791483562,8
64,30,64,0.2655172413793103,2.1449227283740866,21
64,30,128,0.27873563218390807,2.1084044653793863,17
64,30,256,0.27298850574712646,2.0707324904957036,8
64,50,64,0.2649425287356322,2.0767648548915467,8
64,50,128,0.25862068965517243,2.080995068604919,9
64,50,256,0.28045977011494255,2.096523918502632,18
96,10,64,0.27011494252873564,2.09696971630228,9
96,10,128,0.25,2.073631817170943,7
96,10,256,0.2574712643678161,2.0797795651972977,9
96,30,64,0.26436781609195403,2.0948885166782074,11
96,30,128,0.267816091954023,2.11940429347685,14
96,30,256,0.2752873563218391,2.1549230871529415,24
96,50,64,0.2689655172413793,2.3467276512891395,31
96,50,128,0.25862068965517243,2.11188253643869,12
96,50,256,0.26091954022988506,2.264453822716899,34
128,10,64,0.23850574712643677,2.109774117634214,9
128,10,128,0.2540229885057471,2.096012789353557,9
128,10,256,0.2517241379310345,2.0751135278022153,9
128,30,64,0.2689655172413793,2.2314386784345253,23
128,30,128,0.26264367816091955,2.098473955570966,17
128,30,256,0.25689655172413794,2.078461936972607,13
128,50,64,0.26091954022988506,2.161126708984375,15
128,50,128,0.27298850574712646,2.081014743344537,11
128,50,256,0.2574712643678161,2.267714433560426,26
"""

def plot1():
    # Load from string (you can also read from CSV file)
    from io import StringIO
    df = pd.read_csv(StringIO(DATA))

    # Ensure types
    df["resize"] = df["resize"].astype(int)
    df["epochs"] = df["epochs"].astype(int)
    df["batch_size"] = df["batch_size"].astype(int)
    df["best_val_acc"] = df["best_val_acc"].astype(float)

    # Bubble size mapping (tune these numbers if you want bigger/smaller bubbles)
    size_map = {64: 80, 96: 160, 128: 260}
    df["bubble_size"] = df["resize"].map(size_map).fillna(140)

    # Color mapping for batch_size (discrete blues like your example)
    bs_levels = sorted(df["batch_size"].unique())
    cmap = plt.cm.Blues
    color_map = {bs: cmap(i / max(1, len(bs_levels)-1)) for i, bs in enumerate(bs_levels)}
    df["color"] = df["batch_size"].map(color_map)

    plt.figure(figsize=(9, 6))

    plt.scatter(
        df["epochs"],                 # X
        df["best_val_acc"],           # Y
        s=df["bubble_size"],          # size by resize
        c=df["color"].tolist(),       # color by batch_size
        alpha=0.75,
        edgecolors="k",
        linewidths=0.4
    )

    plt.title("Validation accuracy vs Hyperparameter config (Coarse)")
    plt.xlabel("Epochs")
    plt.ylabel("Best val accuracy")
    plt.grid(True, alpha=0.3)

    # Legend for colors (batch_size)
    color_handles = []
    for bs in bs_levels:
        color_handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       label=str(bs),
                       markerfacecolor=color_map[bs],
                       markeredgecolor='k',
                       markersize=9)
        )
    leg1 = plt.legend(handles=color_handles, title="Batch size", loc="upper right")
    plt.gca().add_artist(leg1)

    # Legend for sizes (resize)
    resize_levels = sorted(df["resize"].unique())
    size_handles = []
    for r in resize_levels:
        size_handles.append(
            plt.scatter([], [], s=size_map.get(r, 140),
                        facecolors="none", edgecolors="k", label=str(r))
        )
    plt.legend(handles=size_handles, title="Image size (resize)", loc="center right")

    plt.tight_layout()
    plt.savefig("coarse_bubble_plot.png", dpi=200)
    plt.show()



DATA2 = """resize,batch_size,epochs,hidden_dims,best_val_acc,best_val_loss,best_epoch
64,256,50,[128],0.2649425287356322,2.079698524803951,19
64,256,50,[256],0.26666666666666666,2.0770859526491714,9
64,256,50,[300],0.26436781609195403,2.1078053638852876,20
64,256,50,[512],0.2637931034482759,2.122025049143824,24
64,256,50,"[128, 128]",0.28045977011494255,2.0574665546417235,19
64,256,50,"[256, 256]",0.2689655172413793,2.1771490645134586,31
64,256,50,"[300, 300]",0.2637931034482759,2.206629656101095,33
64,256,50,"[512, 512]",0.27241379310344827,2.1157307137017964,20
64,256,50,"[128, 128, 128]",0.2695402298850575,2.086174488067627,12
64,256,50,"[256, 256, 256]",0.27298850574712646,2.0794131947659897,15
64,256,50,"[300, 300, 300]",0.26666666666666666,2.191772546987424,26
64,256,50,"[512, 512, 512]",0.2689655172413793,2.0782511152070144,14
64,256,50,"[128, 128, 128, 128, 128]",0.2695402298850575,2.0946593443552652,21
64,256,50,"[256, 256, 256, 256, 256]",0.2672413793103448,2.1655847006830675,24
64,256,50,"[300, 300, 300, 300, 300]",0.2793103448275862,2.079887111159577,25
64,256,50,"[512, 512, 512, 512, 512]",0.2655172413793103,2.0911035175981194,15
64,256,50,"[512, 256]",0.26264367816091955,2.088873458182675,7
64,256,50,"[512, 256, 128]",0.2689655172413793,2.190585408265563,29
64,256,50,"[300, 200, 100]",0.27241379310344827,2.098589711901785,25
"""

def parse_hidden_dims(x):
    # hidden_dims comes sometimes as [128] and sometimes as string "[128, 128]"
    if isinstance(x, str):
        x = x.strip()
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return x

def is_uniform_list(lst):
    return isinstance(lst, list) and len(lst) >= 1 and all(v == lst[0] for v in lst)

def plot2():
    df = pd.read_csv(StringIO(DATA2))
    df["hidden_dims"] = df["hidden_dims"].apply(parse_hidden_dims)

    # Keep only uniform hidden_dims like [w]*d (so neurons_per_layer is well-defined)
    df_u = df[df["hidden_dims"].apply(is_uniform_list)].copy()

    df_u["hidden_layers"] = df_u["hidden_dims"].apply(len)
    df_u["neurons_per_layer"] = df_u["hidden_dims"].apply(lambda l: int(l[0]))
    df_u["best_val_acc"] = df_u["best_val_acc"].astype(float)

    # Pivot to matrix: rows=layers, cols=neurons, values=acc
    pivot = df_u.pivot_table(
        index="hidden_layers",
        columns="neurons_per_layer",
        values="best_val_acc",
        aggfunc="max"   # in case of duplicates, keep best
    ).sort_index().sort_index(axis=1)

    # Plot heatmap with matplotlib (no seaborn)
    plt.figure(figsize=(8, 5))
    im = plt.imshow(pivot.values, aspect="auto")

    plt.title("Heatmap: Best Val Accuracy vs (Hidden Layers, Neurons/Layer)")
    plt.xlabel("Neurons per layer")
    plt.ylabel("Number of hidden layers")

    # ticks
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    # annotate each cell with accuracy (optional but nice)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, label="Best val accuracy")

    plt.tight_layout()
    plt.savefig("fine_heatmap_layers_vs_neurons.png", dpi=200)
    plt.show()



if __name__ == "__main__":
    plot1()
    plot2()



