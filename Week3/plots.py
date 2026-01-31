import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def heatmap(csv_file: str, output_plot: str, x: str, y: str, value: str, title: str, x_label: str = None, y_label: str = None, view: bool = False):
    df = pd.read_csv(csv_file)
    df_piv = df.pivot_table(
        index=x,
        columns=y,
        values=value,
        aggfunc="max"
    ).sort_index().sort_index(axis=1)

    fig = plt.figure(figsize=(9, 6))
    
    if not x_label:
        x_label = x
    if not y_label:
        y_label = y

    ax = sns.heatmap(df_piv, annot=True, fmt=".3g", cmap='viridis', cbar_kws={'label': value} )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig.add_axes(ax)
    fig.savefig(output_plot)

    if view:
        fig.show()
        input()


def barplot(csv_file: str, output_plot: str, x: str, y: str, title: str, x_label: str = None, y_label: str = None, view: bool = False):
    df = pd.read_csv(csv_file)

    fig = plt.figure(figsize=(9, 6))
    
    if not x_label:
        x_label = x
    if not y_label:
        y_label = y

    ax = sns.barplot(df, x=x, y=y, palette='Set1', order=df.sort_values(y, ascending=False)[x])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig.add_axes(ax)
    fig.savefig(output_plot)

    if view:
        fig.show()
        input()
