import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Output path
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

# Styling
plt.style.use("dark_background")
fig = plt.figure(figsize=(13.5, 6.5), dpi=300)
fig.patch.set_facecolor("#070b14")

# GridSpec: 2 columns (Main Chart: 74% width, KPI Takeaways Panel: 26% width)
gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.25], wspace=0.16, left=0.07, right=0.96, top=0.79, bottom=0.14)

ax = fig.add_subplot(gs[0])
ax.set_facecolor("#0a0f1e")

# Colors
c_r1 = "#00d4ff"   # Cyan
c_r5 = "#b347ff"   # Purple
c_mrr = "#00e676"  # Emerald

width = 0.22
rects1 = ax.bar(x - width, r1_scores, width, label="Recall@1", color=c_r1, alpha=0.92, edgecolor="#070b14", linewidth=1.2)
rects2 = ax.bar(x, r5_scores, width, label="Recall@5", color=c_r5, alpha=0.92, edgecolor="#070b14", linewidth=1.2)
rects3 = ax.bar(x + width, mrr_scores, width, label="MRR (Mean Reciprocal Rank)", color=c_mrr, alpha=0.92, edgecolor="#070b14", linewidth=1.2)

# Value Labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7.2, fontweight="700",
                    color="#e2e8f0")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=10, fontweight="600", color="#cbd5e1", labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=9.2, fontweight="600", color="#e2e8f0")
ax.set_ylim(0, 1.15)
ax.set_xlim(-0.6, 4.6)

# Grid & Spines
ax.grid(axis="y", linestyle="--", alpha=0.15, color="#ffffff")
for spine in ax.spines.values():
    spine.set_color((0.0, 0.83, 1.0, 0.3))
    spine.set_linewidth(1.0)

# Legend placed cleanly above the ax
legend = fig.legend(handles=[rects1, rects2, rects3],
                    labels=["Recall@1", "Recall@5", "MRR (Mean Reciprocal Rank)"],
                    loc="center left", bbox_to_anchor=(0.07, 0.84),
                    ncol=3, frameon=True,
                    facecolor="#070b14", edgecolor=(0.0, 0.83, 1.0, 0.4), fontsize=9)
for text in legend.get_texts():
    text.set_color("#f1f5f9")

# --- Right Panel: Key Engineering Takeaways ---
ax_kpi = fig.add_subplot(gs[1])
ax_kpi.set_facecolor("#0a0f1e")
ax_kpi.axis("off")

panel_box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.03,rounding_size=0.06",
                           facecolor="#0a0f1e", edgecolor=(0.0, 0.83, 1.0, 0.35),
                           linewidth=1.2, transform=ax_kpi.transAxes, clip_on=False)
ax_kpi.add_patch(panel_box)

ax_kpi.text(0.5, 0.92, "KEY TAKEAWAYS", transform=ax_kpi.transAxes,
            ha="center", va="center", fontsize=11, fontweight="700", color="#00d4ff")

kpis = [
    ("Recall@1 Surge", "0.300 → 0.700", "+133.3% gain via Cross-Encoder", "#00d4ff"),
    ("Overall MRR", "0.429 → 0.764", "+78.1% neural precision gain", "#00e676"),
    ("Hybrid Recall@5", "0.400 → 0.750", "+87.5% multi-constraint lift", "#b347ff"),
    ("Latency Impact", "~35ms overhead", "Production-viable neural rerank", "#ffaa33")
]

y_pos = 0.74
for title, val, sub, color in kpis:
    ax_kpi.text(0.08, y_pos, title, transform=ax_kpi.transAxes,
                ha="left", va="bottom", fontsize=8.5, fontweight="600", color="#94a3b8")
    ax_kpi.text(0.08, y_pos - 0.05, val, transform=ax_kpi.transAxes,
                ha="left", va="bottom", fontsize=11.5, fontweight="700", color=color)
    ax_kpi.text(0.08, y_pos - 0.095, sub, transform=ax_kpi.transAxes,
                ha="left", va="bottom", fontsize=7.5, color="#cbd5e1")
    y_pos -= 0.20

# Global Super Title & Subtitle with ample vertical space
fig.suptitle("FreightIQ · Multi-Strategy Retrieval Benchmark Across 60 Ground-Truth Queries",
             x=0.51, y=0.96, fontsize=13.5, fontweight="700", color="#ffffff")
fig.text(0.51, 0.905, "Empirical evaluation across 500 carrier profiles · Structured, Qualitative, and Multi-Constraint Queries",
         ha="center", fontsize=9.2, color="#94a3b8")

plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Benchmark chart saved successfully to: {out_path}")
