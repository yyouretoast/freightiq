import os
import matplotlib.pyplot as plt
import numpy as np

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "retrieval_benchmark.png")

# Benchmark Data (60 Ground-Truth Queries over 500 Profiles)
strategies = [
    "SQLite Exact\nRelational",
    "ChromaDB Base\nDense Vector",
    "SQLite FTS5\nLexical (BM25)",
    "Reranked Cosine\n(Dense Fallback)",
    "Reranked Hybrid\n(Cross-Encoder)"
]

r1_scores = [0.967, 0.300, 0.450, 0.300, 0.700]
r5_scores = [0.967, 0.667, 0.667, 0.683, 0.867]
mrr_scores = [0.967, 0.429, 0.527, 0.433, 0.764]

x = np.arange(len(strategies))
width = 0.24

# Academic / Research White Style (matching Version A)
plt.style.use("default")
fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")

# Academic Publication Palette (high-contrast, non-neon, publication grade)
c_r1 = "#2563eb"   # Royal Blue
c_r5 = "#7c3aed"   # Purple
c_mrr = "#0d9488"  # Teal

rects1 = ax.bar(x - width, r1_scores, width, label="Recall@1", color=c_r1, alpha=0.90, edgecolor="#000000", linewidth=0.8)
rects2 = ax.bar(x, r5_scores, width, label="Recall@5", color=c_r5, alpha=0.90, edgecolor="#000000", linewidth=0.8)
rects3 = ax.bar(x + width, mrr_scores, width, label="MRR (Mean Reciprocal Rank)", color=c_mrr, alpha=0.90, edgecolor="#000000", linewidth=0.8)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="600",
                    color="#0f172a")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.set_ylabel("Metric Score (0.00 - 1.00)", fontsize=10, fontweight="500", color="#1e293b", labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=9.5, fontweight="500", color="#0f172a")
ax.set_ylim(0, 1.18)
ax.set_xlim(-0.6, len(strategies) - 0.4)

ax.grid(axis="y", linestyle=":", alpha=0.6, color="#cbd5e1")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#94a3b8")
ax.spines["bottom"].set_color("#94a3b8")

ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=8.8, loc="upper right", ncol=3)
ax.set_title("Figure 1: Multi-Strategy Retrieval Benchmark Across 60 Ground-Truth Queries (500 Carrier Profiles)",
             fontsize=11.5, fontweight="700", color="#0f172a", pad=14)

plt.tight_layout()
plt.savefig(out_path, dpi=300, facecolor="#ffffff")
plt.close()
print(f"Benchmark chart saved successfully to: {out_path}")
