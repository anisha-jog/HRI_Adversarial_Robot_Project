import os
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIG ===
base_dir = "./saved_images/2/"

visual_levels = ["similar", "neutral", "different"]   # x-axis (columns)
semantic_levels = ["different", "neutral", "similar"] # y-axis (rows, top -> bottom)

img_name = "turn_2_robot.png"
output_path = base_dir+"grid_visual_semantic.png"


def get_img_path(base, v_level, s_level):
    """Build path like:
       base/custom_visual-<v_level>_semantic-<s_level>/turn_2_robot.png
    """
    folder = f"custom_visual-{v_level}_semantic-{s_level}"
    return os.path.join(base, folder, img_name)


# === PLOT ===
n_rows = len(semantic_levels)
n_cols = len(visual_levels)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))

for row, sem in enumerate(semantic_levels):
    for col, vis in enumerate(visual_levels):
        ax = axes[row, col]
        img_path = get_img_path(base_dir, vis, sem)

        if not os.path.exists(img_path):
            print(f"WARNING: {img_path} not found")
            ax.axis("off")
            continue

        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

# Add margins (space) between images
plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.92,
    bottom=0.08,
    wspace=0.15,  # horizontal margin between images
    hspace=0.15,  # vertical margin between images
)

# === GLOBAL AXIS FOR TICKS + LABELS ===
# This axis spans the whole figure but doesn't draw its own frame
big_ax = fig.add_subplot(111, frameon=False)

# Make sure the big axis doesn't cover the images visually
big_ax.tick_params(labelcolor='black', top=False, bottom=True, left=True, right=False)
big_ax.set_xlim(0, n_cols)
big_ax.set_ylim(0, n_rows)

# Ticks centered over each column/row
x_centers = [i + 0.5 for i in range(n_cols)]
y_centers = [i + 0.5 for i in range(n_rows)]

big_ax.set_xticks(x_centers)
big_ax.set_xticklabels(visual_levels)
big_ax.set_yticks(y_centers)
big_ax.set_yticklabels(semantic_levels, rotation=90, va='center')

# By default, y=0 at bottom, but our row=0 is top.
# If you want "similar" at the TOP and "different" at the BOTTOM, invert:
big_ax.invert_yaxis()

big_ax.set_xlabel("Visual", labelpad=15)
big_ax.set_ylabel("Semantic", labelpad=15)

# Hide the spines (frame) of the big axis
for spine in big_ax.spines.values():
    spine.set_visible(False)

plt.suptitle("Visual vs Semantic (similar / neutral / different)", fontsize=12)

plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved grid to: {output_path}")