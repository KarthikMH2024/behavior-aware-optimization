import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# -------------------------------------------------
# Load training artifacts
# -------------------------------------------------
ckpt = torch.load("training_artifacts.pt", weights_only=False)
history = ckpt["history"]
ga_fitness = ckpt["ga_fitness"]
y_true = ckpt["y_true"]
y_pred = ckpt["y_pred"]
route_encoding = ckpt["route_encoding"]
route_names = list(route_encoding.keys())

print("\nLoaded artifacts successfully")
print("Routes:", route_names)

# -------------------------------------------------
# Prepare all plots as a list of functions
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(bottom=0.2)

plot_idx = 0  # current plot index

def plot_train_val_loss():
    ax.clear()
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Validation Loss")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

def plot_train_val_acc():
    ax.clear()
    ax.plot(history["train_acc"], label="Train Accuracy")
    ax.plot(history["val_acc"], label="Validation Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

def plot_ga_fitness():
    ax.clear()
    ax.plot(ga_fitness, marker="o")
    ax.set_title("Evolutionary Search Fitness Progress")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Behavioral Fitness (Worst-case + Variance)")
    ax.grid(True)

def plot_confusion_matrix():
    ax.clear()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=route_names, yticklabels=route_names, ax=ax)
    ax.set_title("Route Classification Confusion Matrix")
    ax.set_xlabel("Predicted Route")
    ax.set_ylabel("Actual Route")

# list of plotting functions in order
plot_funcs = [plot_train_val_loss, plot_train_val_acc, plot_ga_fitness, plot_confusion_matrix]

# -------------------------------------------------
# Button callbacks
# -------------------------------------------------
def next_plot(event):
    global plot_idx
    plot_idx = (plot_idx + 1) % len(plot_funcs)
    plot_funcs[plot_idx]()
    fig.canvas.draw_idle()

def prev_plot(event):
    global plot_idx
    plot_idx = (plot_idx - 1) % len(plot_funcs)
    plot_funcs[plot_idx]()
    fig.canvas.draw_idle()

# -------------------------------------------------
# Add buttons
# -------------------------------------------------
axprev = plt.axes([0.2, 0.05, 0.1, 0.075])
axnext = plt.axes([0.7, 0.05, 0.1, 0.075])
btn_prev = Button(axprev, 'Previous')
btn_next = Button(axnext, 'Next')
btn_prev.on_clicked(prev_plot)
btn_next.on_clicked(next_plot)

# -------------------------------------------------
# Initial plot
# -------------------------------------------------
plot_funcs[plot_idx]()
plt.show()

# -------------------------------------------------
# Classification report (text)
# -------------------------------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=route_names))
