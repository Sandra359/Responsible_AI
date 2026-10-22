"""Week 1, step 4: linear probes for edema and pleural effusion, BioClinicalBERT vs TF-IDF.

    python step4_probes.py

For each disease:
  - studies: usable Findings AND a clear label (positive = 1, negative = 0)
  - split:   StratifiedGroupKFold, 5 folds, groups = patient_id (no patient in train AND test)
  - models:  BERT    standardised 768-D embedding -> logistic regression
             TF-IDF  words + word pairs (fitted on the training fold only) -> logistic regression
  - score:   collect the test-fold probabilities of all 5 folds, compute ONE AUROC

Output: results/step4_probes.csv, figures/week1/step4_roc.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import binary_label, findings_texts, study_table, usable_findings
from src.embeddings import embedding_path, load_embeddings

DISEASES = {"edema": "Edema", "pleural_effusion": "Pleural effusion"}
SEED = 0
RESULTS = Path("results/step4_probes.csv")
FIG = Path("figures/week1/step4_roc.png")


def bert_probe():
    """Standardise the 768 values, then logistic regression (C = inverse regularisation strength)."""
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=0.1, max_iter=3000))])


def tfidf_probe():
    """Words + word pairs -> TF-IDF vector -> logistic regression.
    Inside a Pipeline, the vocabulary is learned from the training fold only."""
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("clf", LogisticRegression(C=1.0, max_iter=3000))])


def cross_val_probs(make_model, X, y, groups):
    """Patient-wise 5-fold CV. Returns the out-of-fold probability for every study."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    probs = np.zeros(len(y))
    for fold, (train, test) in enumerate(cv.split(X, y, groups)):
        # sanity check: no patient appears in both parts
        assert not set(groups[train]) & set(groups[test])
        model = make_model()
        model.fit(X[train], y[train])
        probs[test] = model.predict_proba(X[test])[:, 1]
    return probs


def main():
    studies = study_table()
    X_all = load_embeddings(embedding_path("findings"), studies)
    texts_all = np.array(findings_texts(studies), dtype=object)
    usable = usable_findings(studies).values

    rows = []
    fig, axes = plt.subplots(1, len(DISEASES), figsize=(11, 5))
    for ax, (name, nice) in zip(axes, DISEASES.items()):
        label = binary_label(studies, name)
        keep = usable & label.notna().values                 # clear cases only
        y = label[keep].values.astype(int)
        groups = studies.loc[keep, "patient_id"].values
        X, texts = X_all[keep], texts_all[keep]

        p_bert = cross_val_probs(bert_probe, X, y, groups)
        p_tfidf = cross_val_probs(tfidf_probe, texts, y, groups)
        auc_bert, auc_tfidf = roc_auc_score(y, p_bert), roc_auc_score(y, p_tfidf)

        rows.append(dict(disease=name, n=len(y), positives=int(y.sum()), negatives=int((1 - y).sum()),
                         patients=len(set(groups)), auroc_bert=round(auc_bert, 3),
                         auroc_tfidf=round(auc_tfidf, 3)))
        print(f"{nice:17s} n={len(y)} ({y.sum()} pos / {(1 - y).sum()} neg, {len(set(groups))} patients)  "
              f"AUROC  BERT={auc_bert:.3f}  TF-IDF={auc_tfidf:.3f}")

        # ROC curves: how true-positive rate trades off against false-positive rate
        for p, auc, col, lab in [(p_bert, auc_bert, "#1f77b4", "BioClinicalBERT"),
                                 (p_tfidf, auc_tfidf, "#ff7f0e", "TF-IDF")]:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{lab} (AUROC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance (0.5)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{nice} (n={len(y)})")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_aspect("equal")

    fig.suptitle("Linear probes on the Findings text, patient-wise 5-fold CV")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS, index=False)
    print(f"\nsaved {RESULTS} and {FIG}")


if __name__ == "__main__":
    main()