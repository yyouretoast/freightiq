import os
import matplotlib.pyplot as plt
import numpy as np

# Output path
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "retrieval_benchmark.png")

# Benchmark Data (60 Ground-Truth Queries over 500 Profiles)
strategies = [
    "SQLite Exact\nRelational",
    "ChromaDB Base\nDense Vector",
    "SQLite FTS5\nLexical (BM25)",
    "Reranked Cosine\n(Fallback)",
    "Reranked Hybrid\n(Cross-Encoder)"
]

r1_scores = [0.967, 0.500, 0.633, 0.500, 0.850]
r5_scores = [0.967, 0.733, 0.817, 0.733, 0.900]
mrr_scores = [0.967, 0.596, 0.701, 0.595, 0.872]

x = np.arange(len(strategies))
width = 0.24

# Styling
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
fig.patch.set_facecolor("#070b14")
ax.set_facecolor("#0a0f1e")

# Colors
c_r1 = "#00d4ff"   # Cyan
c_r5 = "#b347ff"   # Purple
c_mrr = "#00e676"  # Emerald

rects1 = ax.bar(x - width, r1_scores, width, label="Recall@1", color=c_r1, alpha=0.92, edgecolor="#070b14", linewidth=1.5)
rects2 = ax.bar(x, r5_scores, width, label="Recall@5", color=c_r5, alpha=0.92, edgecolor="#070b14", linewidth=1.5)
rects3 = ax.bar(x + width, mrr_scores, width, label="MRR (Mean Reciprocal Rank)", color=c_mrr, alpha=0.92, edgecolor="#070b14", linewidth=1.5)

# Value Labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="600",
                    color="#e2e8f0")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Titles & Labels
ax.set_title("FreightIQ · Empirical Retrieval Benchmark Across 60 Ground-Truth Queries",
             fontsize=14, fontweight="700", color="#ffffff", pad=18)
ax.text(0.5, 1.02, "Evaluated across 500 carrier profiles · Structured, Qualitative, and Multi-Constraint Queries",
        transform=ax.transAxes, ha="center", fontsize=9.5, color="#94a3b8")

ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=10, fontweight="600", color="#cbd5e1", labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=9.5, fontweight="600", color="#e2e8f0")
ax.set_ylim(0, 1.12)

# Grid & Spines
ax.grid(axis="y", linestyle="--", alpha=0.15, color="#ffffff")
for spine in ax.spines.values():
    spine.set_color((0.0, 0.83, 1.0, 0.3))
    spine.set_linewidth(1.0)

# Legend
legend = ax.legend(loc="upper left", framealpha=0.25, facecolor="#070b14", edgecolor="#00d4ff", fontsize=9)
for text in legend.get_texts():
    text.set_color("#f1f5f9")

# Callout annotation box for the Cross-Encoder gain
callout_text = (
    "Neural Cross-Encoder Impact:\n"
    "• Recall@1: 0.500 -> 0.850 (+70.0%)\n"
    "• Overall MRR: 0.596 -> 0.872 (+46.3%)\n"
    "• Qualitative Jargon: 1.000 Recall@1"
)
ax.text(0.98, 0.05, callout_text,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, color="#00e676",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#070b14", edgecolor="#00e676", alpha=0.85, linewidth=1.2))

plt.tight_layout()
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Benchmark chart saved successfully to: {out_path}")
