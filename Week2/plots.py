import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

def plot1():
    df = pd.read_csv("results/coarse_results.csv")
    df_piv = df.pivot_table(
        index="resize",
        columns="batch_size",
        values="best_val_acc",
        aggfunc="max"
    ).sort_index().sort_index(axis=1)
    
    fig = plt.figure(figsize=(9, 6))
    
    ax = sns.heatmap(df_piv, annot=True, fmt=".3g", cmap='viridis', cbar_kws={'label': 'Best val accuracy'} )
    ax.set_title("Heatmap: Best Val Accuracy vs (Image Resize, Batch Size)")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Image size (resize)")
    
    fig.add_axes(ax)
    fig.savefig("results/coarse_heatmap.png")

def plot2():
    df = pd.read_csv("results/fine_results.csv")
    df["num_hidden_dims"] = df.apply(lambda x: len(ast.literal_eval(x.hidden_dims)), axis=1)
    df["num_neurons"] = df.apply(lambda x: ast.literal_eval(x.hidden_dims)[0], axis=1)
    df_piv = df.pivot_table(
        index="num_hidden_dims",
        columns="num_neurons",
        values="best_val_acc",
        aggfunc="max"
    ).sort_index().sort_index(axis=1)
    
    fig = plt.figure(figsize=(9, 6))
    
    ax = sns.heatmap(df_piv, annot=True, fmt=".3g", cmap='viridis', cbar_kws={'label': 'Best val accuracy'} )
    ax.set_title("Heatmap: Best Val Accuracy vs (Num Hidden Dims, Num Neurons)")
    ax.set_xlabel("Num neurons")
    ax.set_ylabel("Num hidden dims")
    
    fig.add_axes(ax)
    fig.savefig("results/fine_heatmap.png")

def plot3():
    df = pd.read_csv("results/svm_results.csv")
    
    fig = plt.figure(figsize=(9, 6))
    
    ax = sns.barplot(df, x='kernel', y='val_acc', palette='Set1', order=df.sort_values('val_acc', ascending=False).kernel)
    ax.set_axisbelow(True)
    ax.grid(True, axis='both')
    ax.set_title("Best Val Accuracy vs Kernel Type")
    ax.set_xlabel("Kernel Type")
    ax.set_ylabel("Val Accuracy")
    
    fig.add_axes(ax)
    fig.savefig("results/svm_barplot.png")

def plot4():
    df = pd.read_csv("results/patch_results.csv")
    df['run_name'] = df.apply(lambda x: f"patch{x.patch_size}_agg{x.agg_method}", axis=1)

    fig = plt.figure(figsize=(9, 6))
    
    ax = sns.barplot(df, x='run_name', y='best_val_acc', palette='Set1', order=df.sort_values('best_val_acc', ascending=False).run_name)
    ax.set_axisbelow(True)
    ax.grid(True, axis='both')
    ax.set_title("Best Val Accuracy by Size of patch and Aggregation method")
    ax.set_xlabel("Run name")
    ax.set_ylabel("Best Val Accuracy")
    ax.tick_params('x', rotation=45)
    fig.add_axes(ax)
    fig.tight_layout()
    fig.savefig("results/patch_barplot.png")

if __name__ == "__main__":
    plot4()
