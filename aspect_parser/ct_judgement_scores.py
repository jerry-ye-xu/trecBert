import numpy as np
import pandas as pd

def calc_gene_score(g1, g2, g3):
    print(g1, g2, g3)
    if g2 is np.NaN and g3 is np.NaN:
        print("1 gene")
        if g1 == "Exact":
            return 6
        elif g1 == "Missing Variant":
            return 2
        else:
            return 0
    elif g3 is np.NaN:
        print("2 genes")
        score_total = 0
        if g1 == "Exact":
            score_total += 3
        elif g1 == "Missing Variant":
            score_total += 1
        else:
            score_total += 0

        if g2 == "Exact":
            score_total += 3
        elif g2 == "Missing Variant":
            score_total += 1
        else:
            score_total += 0

        return score_total
    else:
        print("3 genes")
        score_total = 0
        if g1 == "Exact":
            score_total += 2
        elif g1 == "Missing Variant":
            score_total += 1
        else:
            score_total += 0

        if g2 == "Exact":
            score_total += 2
        elif g2 == "Missing Variant":
            score_total += 1
        else:
            score_total += 0

        if g3 == "Exact":
            score_total += 2
        elif g3 == "Different Variant":
            score_total += 1
        else:
            score_total += 0
        return score_total

if __name__ == "__main__":
    YEARS = [2017, 2018, 2019]
    for YEAR in YEARS:
        df = pd.read_csv(f"../data/pm_labels_{YEAR}/clinical_trials_judgments_{YEAR}.csv")
        print(df.head())
        df_human_pm = df[df["pm_rel_desc"] == "Human PM"]
        df_human_pm.head()

        df_human_pm["gene_score"] = df_human_pm.apply(lambda row: calc_gene_score(
        row["gene1_annotation_desc"], row["gene2_annotation_desc"], row["gene3_annotation_desc"],
    ), axis=1)

        disease_score_map = {
            "Exact": 3,
            "More General": 2,
            "More Specific": 2,
            "Not Disease": 0
        }

        df_human_pm["disease_score"] = df_human_pm["disease_desc"].map(disease_score_map)

        df_human_pm.to_pickle(f"./data/aspect_data/ct_judgement_scores_{YEAR}.pickle")

