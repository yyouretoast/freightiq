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
        "SQLite Exact\n(Relational Filter)",
        "ChromaDB Base\n(Dense Vector)",
        "SQLite FTS5\n(Lexical BM25)",
        "Reranked Cosine\n(Dense Fallback)",
        "Reranked Hybrid\n(Cross-Encoder)"
    ]

    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmark_results.json")
    if os.path.exists(results_path):
        import json
        with open(results_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f).get("overall", {})
        strat_keys = [
            "SQLite Exact Query",
            "ChromaDB Base Vector",
            "FTS5 Lexical Search (BM25)",
            "Reranked Search (Cosine)",
            "Reranked Hybrid (Cross-Encoder)"
        ]
        h1 = [bench_data.get(k, {}).get("hit@1", 0.0) for k in strat_keys]
        h3 = [bench_data.get(k, {}).get("hit@3", 0.0) for k in strat_keys]
        h5 = [bench_data.get(k, {}).get("hit@5", 0.0) for k in strat_keys]
        mrr = [bench_data.get(k, {}).get("mrr", 0.0) for k in strat_keys]
    else:
        h1 = [1.000, 0.317, 0.467, 0.317, 0.717]
        h3 = [1.000, 0.567, 0.583, 0.567, 0.833]
        h5 = [1.000, 0.683, 0.667, 0.700, 0.883]
        mrr = [1.000, 0.445, 0.535, 0.449, 0.781]

    x = np.arange(len(strategies))
    width = 0.18

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12.0, 6.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    c_h1 = "#2563eb"   # Royal Blue
    c_h3 = "#3b82f6"   # Sky Blue
    c_h5 = "#7c3aed"   # Purple
    c_mrr = "#0d9488"  # Teal

    rects1 = ax.bar(x - 1.5 * width, h1, width, label="Hit@1", color=c_h1, alpha=0.92, edgecolor="#0f172a", linewidth=0.6)
    rects2 = ax.bar(x - 0.5 * width, h3, width, label="Hit@3", color=c_h3, alpha=0.92, edgecolor="#0f172a", linewidth=0.6)
    rects3 = ax.bar(x + 0.5 * width, h5, width, label="Hit@5", color=c_h5, alpha=0.92, edgecolor="#0f172a", linewidth=0.6)
    rects4 = ax.bar(x + 1.5 * width, mrr, width, label="MRR", color=c_mrr, alpha=0.92, edgecolor="#0f172a", linewidth=0.6)

    # Clean non-overlapping data labels
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.0, fontweight="600",
                        color="#0f172a")

    ax.set_ylabel("Metric Score (0.00 - 1.00)", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9.2, fontweight="600", color="#0f172a")
    ax.set_ylim(0, 1.20)
    ax.set_xlim(-0.6, len(strategies) - 0.4)

    # Demarcation line separating deterministic vs statistical retrieval
    ax.axvline(0.5, color="#cbd5e1", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(0.0, 1.13, "Deterministic Filter", ha="center", fontsize=8.0, fontweight="700", color="#1e293b",
            bbox=dict(boxstyle="square,pad=0.25", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.8))
    ax.text(2.5, 1.13, "Statistical Retrieval & Neural Ranking Pipelines", ha="center", fontsize=8.0, fontweight="700", color="#1e293b",
            bbox=dict(boxstyle="square,pad=0.25", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.8))

    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=8.5, loc="upper right", ncol=4)
    ax.set_title("Figure 1: Multi-Strategy Retrieval Benchmark Across 60 Ground-Truth Queries",
                 fontsize=11.5, fontweight="700", color="#0f172a", pad=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="#ffffff")
    plt.close()
    print(f"[1/4] Figure 1 saved to: {out_path}")


def generate_figure_2_latency_tradeoff():
    """Figure 2: Retrieval Latency vs. Accuracy Pareto Trade-Off."""
    out_path = os.path.join(out_dir, "retrieval_latency_tradeoff.png")

    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmark_results.json")
    if os.path.exists(results_path):
        import json
        with open(results_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f).get("overall", {})
        sqlite_mrr = bench_data.get("SQLite Exact Query", {}).get("mrr", 1.0)
        sqlite_h1 = bench_data.get("SQLite Exact Query", {}).get("hit@1", 1.0)
        fts5_mrr = bench_data.get("FTS5 Lexical Search (BM25)", {}).get("mrr", 0.535)
        fts5_h1 = bench_data.get("FTS5 Lexical Search (BM25)", {}).get("hit@1", 0.467)
        dense_mrr = bench_data.get("ChromaDB Base Vector", {}).get("mrr", 0.445)
        dense_h1 = bench_data.get("ChromaDB Base Vector", {}).get("hit@1", 0.317)
        cosine_mrr = bench_data.get("Reranked Search (Cosine)", {}).get("mrr", 0.449)
        cosine_h1 = bench_data.get("Reranked Search (Cosine)", {}).get("hit@1", 0.317)
        cross_mrr = bench_data.get("Reranked Hybrid (Cross-Encoder)", {}).get("mrr", 0.781)
        cross_h1 = bench_data.get("Reranked Hybrid (Cross-Encoder)", {}).get("hit@1", 0.717)
    else:
        sqlite_mrr, sqlite_h1 = 1.000, 1.000
        fts5_mrr, fts5_h1 = 0.535, 0.467
        dense_mrr, dense_h1 = 0.445, 0.317
        cosine_mrr, cosine_h1 = 0.449, 0.317
        cross_mrr, cross_h1 = 0.781, 0.717

    data = [
        {"name": "SQLite Exact (Relational)", "latency": 0.31, "mrr": sqlite_mrr, "h1": sqlite_h1, "color": "#2563eb", "marker": "s", "offset": (16, 6), "ha": "left"},
        {"name": "SQLite FTS5 (BM25)", "latency": 0.22, "mrr": fts5_mrr, "h1": fts5_h1, "color": "#0284c7", "marker": "o", "offset": (16, -16), "ha": "left"},
        {"name": "ChromaDB Base Vector", "latency": 270.60, "mrr": dense_mrr, "h1": dense_h1, "color": "#64748b", "marker": "^", "offset": (-14, -22), "ha": "right"},
        {"name": "Reranked (Cosine Fallback)", "latency": 271.00, "mrr": cosine_mrr, "h1": cosine_h1, "color": "#d97706", "marker": "d", "offset": (16, -14), "ha": "left"},
        {"name": "Reranked Hybrid (Cross-Encoder)", "latency": 499.37, "mrr": cross_mrr, "h1": cross_h1, "color": "#7c3aed", "marker": "*", "offset": (16, 6), "ha": "left"}
    ]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # True Pareto Frontier: connects FTS5 -> SQLite (which strictly dominates at 0.31ms / 1.000 MRR)
    # The Cross-Encoder is not connected via a false downward slope; it is annotated as the qualitative frontier.
    ax.plot([0.22, 0.31], [fts5_mrr, sqlite_mrr], linestyle="-", color="#2563eb", linewidth=2.0, alpha=0.8, zorder=3, label="Deterministic Pareto Frontier")

    for item in data:
        size = 240 if item["marker"] == "*" else 160
        ax.scatter(item["latency"], item["mrr"], s=size, color=item["color"],
                   marker=item["marker"], edgecolor="#0f172a", linewidth=1.0, zorder=5)

        label_text = f"{item['name']}\n{item['latency']:.2f} ms | MRR: {item['mrr']:.3f} | Hit@1: {item['h1']:.3f}"
        ax.annotate(
            label_text,
            xy=(item["latency"], item["mrr"]),
            xytext=item["offset"],
            textcoords="offset points",
            fontsize=8.0,
            fontweight="600",
            color="#0f172a",
            ha=item.get("ha", "left"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=item["color"], alpha=0.95, linewidth=1.0),
            arrowprops=dict(arrowstyle="->", color=item["color"], lw=0.9, alpha=0.8)
        )

    ax.set_xscale("log")
    ax.set_xlim(0.10, 1500)
    ax.set_ylim(0.2, 1.12)

    ax.set_xlabel("Retrieval Latency in Milliseconds (log scale)", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)

    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g} ms"))
    ax.grid(True, which="major", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.grid(True, which="minor", linestyle=":", alpha=0.25, color="#e2e8f0")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=8.5, loc="lower right")
    ax.set_title("Figure 2: Retrieval Latency vs. Accuracy Trade-Off (60 Ground-Truth Queries)",
                 fontsize=11.5, fontweight="700", color="#0f172a", pad=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="#ffffff")
    plt.close()
    print(f"[2/4] Figure 2 saved to: {out_path}")


def generate_figure_3_stratified_categories():
    """Figure 3: Category-Stratified Retrieval Performance."""
    out_path = os.path.join(out_dir, "retrieval_stratified_categories.png")

    categories = [
        "Structured Queries\n(State, Equip, Safety)",
        "Qualitative Queries\n(Freight Jargon & Notes)",
        "Multi-Constraint Hybrid\n(Filter + Jargon)"
    ]

    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmark_results.json")
    if os.path.exists(results_path):
        import json
        with open(results_path, "r", encoding="utf-8") as f:
            cat_data = json.load(f).get("categories", {})
        cat_keys = ["Structured", "Qualitative", "Hybrid"]
        h1_dense = [cat_data.get(k, {}).get("ChromaDB Base Vector", {}).get("hit@1", 0.0) for k in cat_keys]
        h1_fts5 = [cat_data.get(k, {}).get("FTS5 Lexical Search (BM25)", {}).get("hit@1", 0.0) for k in cat_keys]
        h1_cross = [cat_data.get(k, {}).get("Reranked Hybrid (Cross-Encoder)", {}).get("hit@1", 0.0) for k in cat_keys]
    else:
        h1_dense = [0.350, 0.450, 0.150]
        h1_fts5 = [0.650, 0.600, 0.150]
        h1_cross = [0.900, 0.650, 0.600]

    x = np.arange(len(categories))
    width = 0.24

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11.0, 5.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    c_dense = "#64748b"  # Slate Blue
    c_fts5 = "#0284c7"   # Cyan / Blue
    c_cross = "#7c3aed"  # Royal Violet

    rects1 = ax.bar(x - width, h1_dense, width, label="ChromaDB Base Vector (Dense)", color=c_dense, alpha=0.92, edgecolor="#0f172a", linewidth=0.6)
    rects2 = ax.bar(x, h1_fts5, width, label="SQLite FTS5 Lexical (BM25)", color=c_fts5, alpha=0.92, edgecolor="#0f172a", linewidth=0.6)
    rects3 = ax.bar(x + width, h1_cross, width, label="Reranked Hybrid (Cross-Encoder)", color=c_cross, alpha=0.92, edgecolor="#0f172a", linewidth=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.8, fontweight="600",
                        color="#0f172a")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_ylabel("Hit@1 Score (0.00 - 1.00)", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.2, fontweight="600", color="#0f172a")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.6, len(categories) - 0.4)

    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=8.8, loc="upper right")
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
        "Structured Relational (n=5)",
        "Qualitative Semantic (n=5)",
        "NMFC Freight Class (n=2)",
        "FMCSA Safety Audit (n=3)",
        "Market Spot Rates (n=2)",
        "Guardrail & Safety (n=3)"
    ]

    labels_out = [
        "carrier_sql_query",
        "carrier_semantic_search",
        "freight_class_calc",
        "check_fmcsa_authority",
        "web_search",
        "Guardrail Refusal"
    ]

    matrix = np.array([
        [5, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 3]
    ])

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=6, aspect="auto")

    # Annotate numbers in cells, suppressing zeros for readability
    for i in range(len(labels_in)):
        for j in range(len(labels_out)):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, f"{val}\n(100%)", ha="center", va="center",
                        fontsize=9.5, fontweight="700", color="#ffffff" if val >= 3 else "#0f172a")

    ax.set_xticks(np.arange(len(labels_out)))
    ax.set_yticks(np.arange(len(labels_in)))
    ax.set_xticklabels(labels_out, fontsize=8.8, fontweight="600", color="#0f172a", rotation=25, ha="right")
    ax.set_yticklabels(labels_in, fontsize=9.0, fontweight="600", color="#0f172a")

    ax.set_xlabel("Selected Execution Tool / Output Action", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)
    ax.set_ylabel("Evaluated Query Intent / Category", fontsize=10, fontweight="600", color="#1e293b", labelpad=10)

    ax.set_title("Figure 4: Agent Routing Trajectory & Guardrail Compliance Matrix (N=20)",
                 fontsize=11.5, fontweight="700", color="#0f172a", pad=16)

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
