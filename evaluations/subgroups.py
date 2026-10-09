"Compares model performance across demographic groups"
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from common import predictions, REPORTS, LABELS

def extract_demographics(df):
    df = df.copy()
    info = df["patient_info"].fillna("").astype(str)
    df["age"] = pd.to_numeric(
        info.str.extract(r"Age:\s*(\d+)", expand=False),
        errors="coerce"
    )
    
    df["gender"] = (
        info.str.extract(r"Gender:\s*([^,]+)", expand=False)
        .str.strip()
        .str.title()
    )

    df["race"] = (
        info.str.extract(r"Race:\s*([^,]+)", expand=False)
        .str.strip()
        .str.upper()
    )

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 65, float("inf")],
        labels=["0-17", "18-39", "40-64", "65+"],
        right=False
    )

    return df


def evaluate_subgroups(df, group_column):
    """Calculate model performance for each demographic group."""

    rows = []

    for group, subset in df.groupby(
        group_column, observed=True, dropna=False
    ):
        n = len(subset)
        if n < 20:
            print(
                f"Skipping {group_column}={group}: "
                f"only {n} observations"
            )
            continue
        y_true = subset["y_true"].to_numpy()
        y_pred = subset["y_pred"].to_numpy()
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        )
        undertriage = np.mean(y_pred > y_true)
        overtriage = np.mean(y_pred < y_true)
        esi1_mask = y_true == 1
        esi1_n = np.sum(esi1_mask)

        if esi1_n > 0:
            esi1_recall = np.mean(
                y_pred[esi1_mask] == 1
            )
            esi1_undertriage = np.mean(
                y_pred[esi1_mask] > 1
            )
        else:
            esi1_recall = np.nan
            esi1_undertriage = np.nan

        rows.append({
            "category": group_column,
            "group": str(group),
            "n": n,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "undertriage_rate": undertriage,
            "overtriage_rate": overtriage,
            "esi1_n": esi1_n,
            "esi1_recall": esi1_recall,
            "esi1_undertriage": esi1_undertriage
        })

    return pd.DataFrame(rows)


def main():
    df, _, y, pred, _, _ = predictions()
    df = extract_demographics(df)
    df["y_true"] = y
    df["y_pred"] = pred
    print("\nDemographic data preview:")
    print(
        df[["age", "age_group", "gender", "race"]].head()
    )
    results = []

    for column in ["age_group", "gender", "race"]:
        print(f"\n{'=' * 50}")
        print(f"Evaluation by {column}")
        print("=" * 50)

        result = evaluate_subgroups(df, column)

        if not result.empty:
            print(result.to_string(index=False))
            results.append(result)
        else:
            print("No eligible groups found.")

    if results:
        final_results = pd.concat(
            results,
            ignore_index=True
        )

        output_path = REPORTS / "subgroups.csv"

        final_results.to_csv(
            output_path,
            index=False
        )

        print(f"\nResults saved to: {output_path}")
    else:
        print("\nNo subgroup results to save.")


if __name__ == "__main__":
    main()
