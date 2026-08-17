
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm



INPUT_CSV = Path("your/path/to/CCC_Statistics_Results_all.csv")
OUTPUT_DIR = Path("your/path/to/output")



TARGETS = {
    "Beta": ["PeriMac->Beta", "IntraMac->Beta"],
    "Alpha": ["PeriMac->Alpha", "IntraMac->Alpha"],
}


COMPARISONS = {
    "Ctrl_vs_Obesity": {
        "p_col": "P_Obesity_vs_Ctrl",
        "logfc_col": "Log2FC_Obesity_vs_Ctrl",
        "title": "Obesity vs Ctrl",
    },
    "Ctrl_vs_T2D": {
        "p_col": "P_T2D_vs_Ctrl",
        "logfc_col": "Log2FC_T2D_vs_Ctrl",
        "title": "T2D vs Ctrl",
    },
}


P_THRESHOLD = 0.05
TOP_N_LIST = [10, 15, 20]


def safe_name(value: str) -> str:

    return (
        str(value)
        .replace("->", "_to_")
        .replace(" ", "_")
        .replace("/", "-")
        .replace(".", "p")
    )


def load_data():

    df = pd.read_csv(INPUT_CSV)

    required = {"Direction", "LR_Pair"}
    for config in COMPARISONS.values():
        required.add(config["p_col"])
        required.add(config["logfc_col"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV : {missing}")

    for config in COMPARISONS.values():
        df[config["p_col"]] = pd.to_numeric(df[config["p_col"]], errors="coerce")
        df[config["logfc_col"]] = pd.to_numeric(df[config["logfc_col"]], errors="coerce")

 
    before = len(df)
    df = df[~df["LR_Pair"].str.contains("FGF.*FGFR|FGFR.*FGF", case=True, na=False)]
    print(f"Filtered FGF-FGFR pairs: {before} -> {len(df)} rows")
    return df


def build_plot_frame(df, directions, p_col, logfc_col, p_threshold, top_n):

    data = df[df["Direction"].isin(directions)].copy()
    data = data.dropna(subset=[p_col, logfc_col])
    data = data[data[p_col] < p_threshold].copy()
    data = data[data[logfc_col] != 0].copy()
    if data.empty:
        return data

    data["Regulation"] = np.where(data[logfc_col] > 0, "Up", "Down")
    selected_frames = []


    for direction in directions:
        dir_data = data[data["Direction"] == direction]
        if dir_data.empty:
            continue

        up_rank = (
            dir_data[dir_data["Regulation"] == "Up"]
            .groupby("LR_Pair")[logfc_col]
            .max()
            .sort_values(ascending=False)
        )
        down_rank = (
            dir_data[dir_data["Regulation"] == "Down"]
            .groupby("LR_Pair")[logfc_col]
            .min()
            .sort_values(ascending=True)
        )

        dir_up = up_rank.head(top_n).index.tolist()
        dir_down = down_rank.head(top_n).index.tolist()

        if dir_up:
            up_data = dir_data[
                (dir_data["Regulation"] == "Up") & (dir_data["LR_Pair"].isin(dir_up))
            ].copy()
            up_data["Row_ID"] = "Up::" + up_data["LR_Pair"]
            selected_frames.append(up_data)

        if dir_down:
            down_data = dir_data[
                (dir_data["Regulation"] == "Down") & (dir_data["LR_Pair"].isin(dir_down))
            ].copy()
            down_data["Row_ID"] = "Down::" + down_data["LR_Pair"]
            selected_frames.append(down_data)

    if not selected_frames:
        return data.iloc[0:0].copy()

    data = pd.concat(selected_frames, ignore_index=True)
    if data.empty:
        return data

    row_order = []
    for regulation, ascending in [("Up", False), ("Down", True)]:
        block = data[data["Regulation"] == regulation]
        if block.empty:
            continue
        rank = (
            block.groupby("LR_Pair")[logfc_col]
            .agg(lambda g: g.abs().max() if regulation == "Up" else -g.abs().max())
            .sort_values(ascending=ascending)
        )
        for pair in rank.index.tolist():
            row_order.append(f"{regulation}::{pair}")

    data["Block"] = pd.Categorical(data["Regulation"], categories=["Up", "Down"], ordered=True)
    data["Row_ID"] = pd.Categorical(data["Row_ID"], categories=row_order, ordered=True)
    data["Direction"] = pd.Categorical(data["Direction"], categories=directions, ordered=True)
    return data.sort_values(["Block", "Row_ID", "Direction"])


def plot_dotplot(data, directions, target, comparison_title,
                 p_col, logfc_col, p_threshold, top_n, output_path):

    pair_count = data["Row_ID"].nunique() if not data.empty else 1
    fig_height = max(5.2, 0.23 * pair_count + 2.4)
    fig, ax = plt.subplots(figsize=(4.1, fig_height))

    if data.empty:
        ax.text(0.5, 0.5, "No LR pairs selected", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        x_map = {d: i for i, d in enumerate(directions)}
        y_categories = data["Row_ID"].cat.categories.tolist()
        y_labels = [v.split("::", 1)[1] for v in y_categories]
        up_count = sum(v.startswith("Up::") for v in y_categories)


        gap = 1.2 if up_count > 0 and len(y_categories) > up_count else 0.0
        y_positions = {}
        for i, pair in enumerate(y_categories):
            y_positions[pair] = i + (gap if i >= up_count else 0.0)

        x = data["Direction"].astype(str).map(x_map)
        y = data["Row_ID"].astype(str).map(y_positions)

        vmax = max(abs(data[logfc_col].min()), abs(data[logfc_col].max()))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        scatter = ax.scatter(
            x, y,
            c=data[logfc_col],
            s=52,
            cmap="coolwarm",
            norm=norm,
            edgecolors="0.35",
            linewidths=0.4,
        )

        ax.set_xticks(range(len(directions)))
        ax.set_xticklabels(directions, fontsize=10)
        ax.set_yticks([y_positions[p] for p in y_categories])
        ax.set_yticklabels(y_labels, fontsize=8.5, fontstyle="italic")
        ax.set_xlim(-0.32, len(directions) - 0.68)
        ax.set_ylim(-0.6, max(y_positions.values()) + 0.6)
        ax.grid(False)


        for xb in np.arange(-0.5, len(directions) + 0.5, 1):
            ax.axvline(xb, color="0.25", linewidth=0.9, zorder=0)
        for i, pair in enumerate(y_categories):
            cy = y_positions[pair]
            if i == up_count - 1 and gap > 0:
                ax.axhline(cy + 0.6, color="0.2", linewidth=1.1, zorder=0)
            else:
                ax.axhline(cy + 0.5, color="0.25", linewidth=0.65, zorder=0)
        ax.axhline(-0.5, color="0.25", linewidth=0.65, zorder=0)

        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color("0.2")

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.075, pad=0.06)
        cbar.set_label(logfc_col, fontsize=10)
        fig.subplots_adjust(right=0.88)
        ax.invert_yaxis()

    ax.set_title(
        f"{target}: {comparison_title}\nP < {p_threshold:g}, Top {top_n} Up and Down",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    for target, directions in TARGETS.items():
        available = sorted(set(directions) & set(df["Direction"]))
        if len(available) != len(directions):
            missing = sorted(set(directions) - set(available))
            raise ValueError(f"Missing target directions for {target}: {missing}")

    summary = []
    for target, directions in TARGETS.items():
        for comparison_name, config in COMPARISONS.items():
            for top_n in TOP_N_LIST:
                data = build_plot_frame(
                    df, directions,
                    config["p_col"], config["logfc_col"],
                    P_THRESHOLD, top_n,
                )
                file_stem = f"{target}_{comparison_name}_P{safe_name(str(P_THRESHOLD))}_Top{top_n}"
                output_path = OUTPUT_DIR / file_stem
                plot_dotplot(
                    data, directions, target, config["title"],
                    config["p_col"], config["logfc_col"],
                    P_THRESHOLD, top_n, output_path,
                )
                summary.append({
                    "target": target,
                    "comparison": comparison_name,
                    "p_threshold": P_THRESHOLD,
                    "top_n": top_n,
                    "selected_up_lr_pairs": int(data.loc[data["Regulation"] == "Up", "LR_Pair"].nunique()) if not data.empty else 0,
                    "selected_down_lr_pairs": int(data.loc[data["Regulation"] == "Down", "LR_Pair"].nunique()) if not data.empty else 0,
                    "selected_rows": int(len(data)),
                    "png": str(output_path.with_suffix(".png")),
                    "pdf": str(output_path.with_suffix(".pdf")),
                })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "plot_summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
