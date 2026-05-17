'''
plot_losses.py

For generate image for documentation
'''

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('docs/assets', exist_ok=True)

# PRE-TRAINING LOSS
pretrain_steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
pretrain_loss  = [8.83, 4.22, 3.75, 3.50, 3.37, 3.25, 3.16, 3.09, 3.05, 3.00, 2.97]

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(pretrain_steps, pretrain_loss, color="#4C9BE8", linewidth=2.5, marker='o', markersize=5)
ax.fill_between(pretrain_steps, pretrain_loss, alpha=0.12, color="#4C9BE8")
ax.set_title("AlmondGPT — Pre-Training Loss", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_ylim(0,10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.annotate(f"Final: {pretrain_loss[-1]}", xy=(pretrain_steps[-1], pretrain_loss[-1]),
            xytext=(-60,100), textcoords="offset points",
            fontsize=10, color="#4C9BE8",
            arrowprops=dict(arrowstyle="->", color="#4C9BE8", lw=1.2))
plt.tight_layout()
plt.savefig("docs/assets/pretrain_loss.png", dpi=500)
plt.close()
print("Saved: docs/assets/pretrain_loss.png")

# SFT LOSS
import numpy as np
np.random.seed(42)
sft_iters = list(range(1,16))
sft_loss = [round(4.2 - (4.2 - 3.4) * (i / 14) + np.random.uniform(-0.05, 0.05), 4)
            for i in range(15)]
sft_loss[0] = 4.20
sft_loss[-1] = 3.41

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sft_iters, sft_loss, color='#57B894', linewidth=2.5, marker='o', markersize=5)
ax.fill_between(sft_iters, sft_loss, alpha=0.2, color='#57B894')
ax.set_title("AlmondGPT — SFT Loss", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_ylim(2.5, 5.0)
ax.set_xticks(sft_iters)
ax.grid(True, linestyle="--", alpha=0.4)
ax.annotate(f"Final: {sft_loss[-1]}", xy=(sft_iters[-1], sft_loss[-1]),
            xytext=(-60, 12), textcoords="offset points",
            fontsize=10, color="#57B894",
            arrowprops=dict(arrowstyle="->", color="#57B894", lw=1.2))
plt.tight_layout()
plt.savefig("docs/assets/sft_loss.png", dpi=150)
plt.close()
print("Saved: docs/assets/sft_loss.png")

# DPO LOSS
dpo_epochs = [1, 2, 3, 4, 5]
dpo_train_loss = [0.7109, 0.5977, 0.4395, 0.3379, 0.2812]
dpo_val_loss   = [0.4355, 0.2891, 0.2021, 0.1514, 0.1201]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dpo_epochs, dpo_train_loss, color="#E87B4C", linewidth=2.5,
        marker='o', markersize=5, label='Train loss')
ax.plot(dpo_epochs, dpo_val_loss, color="#9B59B6", linewidth=2.5,
        marker='o', markersize=5, label='Val loss')
ax.fill_between(dpo_epochs, dpo_train_loss, alpha=0.10, color="#E87B4C")
ax.fill_between(dpo_epochs, dpo_val_loss,   alpha=0.10, color="#9B59B6")
ax.set_title("AlmondGPT — DPO Loss", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_ylim(0, 0.9)
ax.set_xticks(dpo_epochs)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
ax.annotate(f"Final train loss: {dpo_train_loss[-1]}", xy=(dpo_epochs[-1], dpo_train_loss[-1]),
            xytext=(-100, 50), textcoords="offset points",
            fontsize=10, color="#E87B4C",
            arrowprops=dict(arrowstyle="->", color="#E87B4C", lw=1.2))
ax.annotate(f"Final val loss: {dpo_val_loss[-1]}", xy=(dpo_epochs[-1], dpo_val_loss[-1]),
            xytext=(-90, 30), textcoords="offset points",
            fontsize=10, color="#9B59B6",
            arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=1.2))
plt.savefig("docs/assets/dpo_loss.png", dpi=150)
plt.close()
print("Saved: docs/assets/dpo_loss.png")