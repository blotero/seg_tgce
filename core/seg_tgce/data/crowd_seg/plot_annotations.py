import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

labels_meta = {
    "NP1": 757,
    "NP10": 1157,
    "NP11": 957,
    "NP12": 867,
    "NP14": 278,
    "NP15": 102,
    "NP16": 698,
    "NP17": 574,
    "NP18": 729,
    "NP19": 308,
    "NP2": 812,
    "NP20": 864,
    "NP21": 868,
    "NP3": 1004,
    "NP4": 897,
    "NP5": 894,
    "NP6": 1259,
    "NP7": 554,
    "NP8": 536,
    "NP9": 820,
}


# Set the style for a publication-quality plot
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)

# Convert the dictionary to a DataFrame and sort by count
df = pd.DataFrame(list(labels_meta.items()), columns=["Labeler", "Count"])
df = df.sort_values("Count", ascending=False)

# Calculate normalized values (percentages)
total_annotations = 10173
df["Percentage"] = (df["Count"] / total_annotations * 100).round(1)

# Create the figure with a specific size
plt.figure(figsize=(12, 6))

# Create the bar plot
ax = sns.barplot(data=df, x="Labeler", y="Count", palette="Blues")

# Customize the plot
plt.title(
    "Annotation Count Distribution Across Labelers",
    pad=20,
    fontsize=16,
    fontweight="bold",
)
plt.xlabel("Labeler ID", labelpad=10, fontsize=14)
plt.ylabel("Number of Annotations", labelpad=10, fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Add value labels on top of each bar with both absolute and percentage values
for i, (count, percentage) in enumerate(zip(df["Count"], df["Percentage"])):
    label = f"{count}\n({percentage}%)"
    ax.text(i, count + 20, label, ha="center", fontsize=10, va="bottom")

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure with high DPI for publication quality
plt.savefig(
    "/home/brandon/unal/maestria/master_thesis/Cap5/Figures/crowd_seg_annotation_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
