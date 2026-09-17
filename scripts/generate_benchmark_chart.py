import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Output directory: docs/assets
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")
os.makedirs(out_dir, exist_ok=True)


def generate_figure_1_retrieval_benchmark():
    """Figure 1: Multi-Strategy Retrieval Benchmark Across 60 Ground-Truth Queries."""
    out_path = os.path.join(out_dir, "retrieval_benchmark.png")
    strategies = [
        "SQLite Exact\nRelational",
        "ChromaDB Base\nDense Vector",
        "SQLite FTS5\nLexical (BM25)",
        "Reranked Cosine\n(Dense Fallback)",
        "Reranked Hybrid\n(Cross-Encoder)"
    ]

    r1_scores = [0.967, 0.300, 0.467, 0.300, 0.700]
    r5_scores = [0.967, 0.667, 0.667, 0.683, 0.867]
    mrr_scores = [0.967, 0.429, 0.535, 0.432, 0.764]

    x = np.arange(len(strategies))
    width = 0.24

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    c_r1 = "#2563eb"   # Royal Blue
    c_r5 = "#7c3aed"   # Purple
    c_mrr = "#0d9488"  # Teal

    rects1 = ax.bar(x - width, r1_scores, width, label="Recall@1", color=c_r1, alpha=0.92, edgecolor="#000000", linewidth=0.8)
    rects2 = ax.bar(x, r5_scores, width, label="Recall@5", color=c_r5, alpha=0.92, edgecolor="#000000", linewidth=0.8)
    rects3 = ax.bar(x + width, mrr_scores, width, label="MRR (Mean Reciprocal Rank)", color=c_mrr, alpha=0.92, edgecolor="#000000", linewidth=0.8)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.5, fontweight="600",
                        color="#0f172a")

    ax.set_ylabel("Metric Score (0.00 - 1.00)", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9.5, fontweight="600", color="#0f172a")
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
    print(f"[1/4] Figure 1 saved to: {out_path}")


def generate_figure_2_latency_tradeoff():
    """Figure 2: Retrieval Latency vs. Accuracy Pareto Trade-Off."""
    out_path = os.path.join(out_dir, "retrieval_latency_tradeoff.png")

    data = [
        {"name": "SQLite Exact Relational", "latency": 0.31, "mrr": 0.967, "r1": 0.967, "color": "#2563eb", "marker": "s", "offset": (18, -8), "ha": "left"},
        {"name": "SQLite FTS5 (BM25)", "latency": 0.22, "mrr": 0.535, "r1": 0.467, "color": "#0284c7", "marker": "o", "offset": (18, -14), "ha": "left"},
        {"name": "ChromaDB Base Vector", "latency": 270.60, "mrr": 0.429, "r1": 0.300, "color": "#64748b", "marker": "^", "offset": (-130, -28), "ha": "right"},
        {"name": "Reranked (Cosine Fallback)", "latency": 271.00, "mrr": 0.432, "r1": 0.300, "color": "#d97706", "marker": "d", "offset": (18, -6), "ha": "left"},
        {"name": "Reranked Hybrid (Cross-Encoder)", "latency": 499.37, "mrr": 0.764, "r1": 0.700, "color": "#7c3aed", "marker": "*", "offset": (-18, 18), "ha": "right"}
    ]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Shaded operational zones
    ax.axvspan(0.12, 1.8, color="#eff6ff", alpha=0.8, zorder=1)
    ax.axvspan(150, 900, color="#faf5ff", alpha=0.8, zorder=1)

    ax.text(0.15, 0.08, "SUB-MILLISECOND RELATIONAL & LEXICAL ZONE\n(< 1 ms latency | SQLite WAL & FTS5 BM25)",
            fontsize=8.2, fontweight="700", color="#1d4ed8", alpha=0.85, zorder=2)

    ax.text(170, 0.08, "NEURAL SEMANTIC & CROSS-ENCODER ZONE\n(250 - 500 ms latency | Dense Embedding & Cross-Encoder)",
            fontsize=8.2, fontweight="700", color="#6d28d9", alpha=0.85, zorder=2)

    # Plot Pareto optimal curve (FTS5 -> SQLite -> Hybrid Cross-Encoder)
    pareto_x = [0.22, 0.31, 499.37]
    pareto_y = [0.535, 0.967, 0.764]
    ax.plot(pareto_x[:2], pareto_y[:2], linestyle="--", color="#2563eb", linewidth=1.5, alpha=0.5, zorder=3)
    ax.plot([0.31, 499.37], [0.967, 0.764], linestyle=":", color="#7c3aed", linewidth=1.5, alpha=0.4, zorder=3)

    for item in data:
        size = 280 if item["marker"] == "*" else 180
        ax.scatter(item["latency"], item["mrr"], s=size, color=item["color"],
                   marker=item["marker"], edgecolor="#0f172a", linewidth=1.2, zorder=5)

        # Annotations with callouts
        label_text = f"{item['name']}\nLatency: {item['latency']:.2f} ms | MRR: {item['mrr']:.3f} | R@1: {item['r1']:.3f}"
        ax.annotate(
            label_text,
            xy=(item["latency"], item["mrr"]),
            xytext=item["offset"],
            textcoords="offset points",
            fontsize=8.2,
            fontweight="600",
            color="#0f172a",
            ha=item.get("ha", "left"),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor=item["color"], alpha=0.92, linewidth=1.1),
            arrowprops=dict(arrowstyle="->", color=item["color"], lw=1.0, alpha=0.8)
        )

    # Architectural takeaway callout box
    callout_text = (
        "Core Architectural Rationale (ADR-001 & ADR-002):\n"
        "• SQLite resolves discrete constraint queries in 0.31 ms with 0.967 MRR (1,600x faster than Cross-Encoder).\n"
        "• The 499 ms Cross-Encoder re-ranker is reserved exclusively for qualitative queries, where dense search alone fails."
    )
    ax.text(0.04, 0.95, callout_text, transform=ax.transAxes, fontsize=8.4, fontweight="500",
            color="#1e293b", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0))

    ax.set_xscale("log")
    ax.set_xlim(0.08, 1600)
    ax.set_ylim(0.0, 1.15)

    ax.set_xlabel("Retrieval Latency in Milliseconds (log scale)", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)

    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g} ms"))
    ax.grid(True, which="major", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.grid(True, which="minor", linestyle=":", alpha=0.3, color="#e2e8f0")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    ax.set_title("Figure 2: Retrieval Latency vs. Accuracy Pareto Trade-Off (60 Ground-Truth Queries)",
                 fontsize=11.5, fontweight="700", color="#0f172a", pad=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="#ffffff")
    plt.close()
    print(f"[2/4] Figure 2 saved to: {out_path}")


def generate_figure_3_stratified_categories():
    """Figure 3: Category-Stratified Retrieval Performance."""
    out_path = os.path.join(out_dir, "retrieval_stratified_categories.png")

    categories = [
        "Structured Queries\n(20 cases: State, Equip, Safety)",
        "Qualitative Queries\n(20 cases: Freight Jargon & Notes)",
        "Multi-Constraint Hybrid\n(20 cases: Filter + Jargon)"
    ]

    # Metrics per category [ChromaDB Base, FTS5 BM25, Cross-Encoder Hybrid]
    r1_dense = [0.350, 0.450, 0.100]
    r1_fts5 = [0.650, 0.600, 0.150]
    r1_cross = [0.900, 0.650, 0.550]

    mrr_dense = [0.508, 0.568, 0.210]
    mrr_fts5 = [0.756, 0.610, 0.239]
    mrr_cross = [0.942, 0.727, 0.625]

    x = np.arange(len(categories))
    width = 0.25

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    c_dense = "#64748b"  # Slate Blue
    c_fts5 = "#0284c7"   # Cyan / Blue
    c_cross = "#7c3aed"  # Royal Violet

    rects1 = ax.bar(x - width, r1_dense, width, label="ChromaDB Base Vector (Dense)", color=c_dense, alpha=0.92, edgecolor="#000000", linewidth=0.8)
    rects2 = ax.bar(x, r1_fts5, width, label="SQLite FTS5 Lexical (BM25)", color=c_fts5, alpha=0.92, edgecolor="#000000", linewidth=0.8)
    rects3 = ax.bar(x + width, r1_cross, width, label="Reranked Hybrid (Cross-Encoder)", color=c_cross, alpha=0.92, edgecolor="#000000", linewidth=0.8)

    def autolabel_multiline(rects, mrr_list):
        for rect, mrr_val in zip(rects, mrr_list):
            height = rect.get_height()
            ax.annotate(f"R@1: {height:.2f}\n(MRR: {mrr_val:.2f})",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.8, fontweight="600",
                        color="#0f172a")

    autolabel_multiline(rects1, mrr_dense)
    autolabel_multiline(rects2, mrr_fts5)
    autolabel_multiline(rects3, mrr_cross)

    # Highlight Callout for Multi-Constraint Collapse
    ax.annotate(
        "Dense & Lexical Collapse (0.10 & 0.15 R@1)\nCross-Encoder Rescues to 0.550 (+450% gain)",
        xy=(x[2] + width, 0.550),
        xytext=(-85, 45),
        textcoords="offset points",
        fontsize=8.2,
        fontweight="700",
        color="#7c3aed",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#faf5ff", edgecolor="#7c3aed", linewidth=1.2),
        arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.2)
    )

    ax.set_ylabel("Metric Score (0.00 - 1.00)", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight="600", color="#0f172a")
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.6, len(categories) - 0.4)

    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=8.8, loc="upper right", ncol=3)
    ax.set_title("Figure 3: Stratified Retrieval Performance Across Distinct Query Categories",
                 fontsize=11.5, fontweight="700", color="#0f172a", pad=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="#ffffff")
    plt.close()
    print(f"[3/4] Figure 3 saved to: {out_path}")


def generate_figure_4_trajectory_matrix():
    """Figure 4: Agent Routing Trajectory Confusion Matrix."""
    out_path = os.path.join(out_dir, "agent_trajectory_matrix.png")

    labels_in = [
        "Structured Relational (5)",
        "Qualitative Semantic (5)",
        "NMFC Freight Class (2)",
        "FMCSA Safety Audit (3)",
        "Market Spot Rates (2)",
        "Guardrail & Safety (3)"
    ]

    labels_out = [
        "carrier_sql_query",
        "carrier_semantic_search",
        "freight_class_calc",
        "check_fmcsa_authority",
        "web_search",
        "Guardrail Refusal"
    ]

    # 6x6 Matrix (100% adherence: 5, 5, 2, 3, 2, 3)
    matrix = np.array([
        [5, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 3]
    ])

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Heatmap color rendering
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=6, aspect="auto")

    # Annotate numbers in cells
    for i in range(len(labels_in)):
        for j in range(len(labels_out)):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, f"{val}\n(100%)", ha="center", va="center",
                        fontsize=10.5, fontweight="700", color="#ffffff" if val >= 3 else "#0f172a")
            else:
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=9.5, fontweight="400", color="#94a3b8")

    ax.set_xticks(np.arange(len(labels_out)))
    ax.set_yticks(np.arange(len(labels_in)))
    ax.set_xticklabels(labels_out, fontsize=8.8, fontweight="600", color="#0f172a", rotation=25, ha="right")
    ax.set_yticklabels(labels_in, fontsize=9.0, fontweight="600", color="#0f172a")

    ax.set_xlabel("Selected Execution Tool / Output Action", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)
    ax.set_ylabel("Incoming Query Intent / Evaluated Category", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)

    # Success Badge below Title
    badge_text = "Overall Trajectory Fidelity: 100.0% (20 / 20 Cases Passed) · Zero False Routing · Zero Leaks"
    ax.text(0.5, 1.03, badge_text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.0, fontweight="700", color="#047857",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#10b981", linewidth=1.2))

    ax.set_title("Figure 4: Agent Routing Trajectory & Guardrail Compliance Matrix",
                 fontsize=11.5, fontweight="700", color="#0f172a", pad=38)

    # Clean borders
    for edge, spine in ax.spines.items():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="#ffffff")
    plt.close()
    print(f"[4/4] Figure 4 saved to: {out_path}")


def main():
    print("=" * 60)
    print("Generating FreightIQ Publication Benchmark Figures...")
    print("=" * 60)
    generate_figure_1_retrieval_benchmark()
    generate_figure_2_latency_tradeoff()
    generate_figure_3_stratified_categories()
    generate_figure_4_trajectory_matrix()
    print("=" * 60)
    print(f"All 4 figures generated successfully in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

