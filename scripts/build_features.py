"""Build feature matrices for classical + DL models.

Produces:
    data/processed/features_tfidf.npz   TF-IDF (word + char n-grams) + tool one-hot
    data/processed/tokens_bert.pt       DistilBERT tokenizer output with tool prepend

TODO(Sat): implement once make_dataset.py lands.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("implemented on Sat Apr 18; see SCOPE.md timeline")


if __name__ == "__main__":
    main()
