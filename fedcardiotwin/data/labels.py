"""Label spaces for the two experiment tracks.

Track A (federated, 6 hospital clients): the 26 scored classes of the
PhysioNet/CinC 2021 challenge, parsed from the official evaluation repo's
weights.csv so that merged equivalent classes (e.g. CRBBB|RBBB) follow the
official challenge definition exactly. Nothing is hardcoded.

Track B (continual twin, PTB-XL): the 5 diagnostic superclasses
(NORM, MI, STTC, CD, HYP) derived from scp_statements.csv.
"""
import os

import pandas as pd


class ScoredLabelSpace:
    """CinC-2021 scored classes with official equivalence merging."""

    def __init__(self, evaluation_repo_dir: str):
        weights_path = os.path.join(evaluation_repo_dir, "weights.csv")
        mapping_path = os.path.join(evaluation_repo_dir, "dx_mapping_scored.csv")
        if not (os.path.exists(weights_path) and os.path.exists(mapping_path)):
            raise FileNotFoundError(
                f"Official CinC-2021 evaluation files not found in {evaluation_repo_dir}. "
                "Clone https://github.com/physionetchallenges/evaluation-2021 there."
            )
        header = pd.read_csv(weights_path, nrows=0).columns.tolist()[1:]
        mapping = pd.read_csv(mapping_path, dtype={"SNOMEDCTCode": str})
        abbrev = dict(zip(mapping["SNOMEDCTCode"], mapping["Abbreviation"]))

        self.classes = []          # display name per class index (e.g. "CRBBB|RBBB")
        self.code_to_index = {}    # every SNOMED code (incl. merged) -> class index
        for i, col in enumerate(header):
            codes = col.split("|")
            self.classes.append("|".join(abbrev.get(c, c) for c in codes))
            for c in codes:
                self.code_to_index[c] = i

    @property
    def num_classes(self):
        return len(self.classes)

    def encode(self, dx_codes):
        """Multi-hot vector from a list of SNOMED code strings.
        Returns None if no code falls in the scored set (record is skipped)."""
        y = [0] * self.num_classes
        hit = False
        for c in dx_codes:
            idx = self.code_to_index.get(c.strip())
            if idx is not None:
                y[idx] = 1
                hit = True
        return y if hit else None


PTBXL_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class PTBXLSuperclassSpace:
    """PTB-XL 5-superclass multilabel space from scp_statements.csv."""

    def __init__(self, ptbxl_dir: str):
        scp = pd.read_csv(os.path.join(ptbxl_dir, "scp_statements.csv"), index_col=0)
        scp = scp[scp.diagnostic == 1]
        self.scp_to_super = scp.diagnostic_class.to_dict()
        self.classes = list(PTBXL_SUPERCLASSES)
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

    @property
    def num_classes(self):
        return len(self.classes)

    def encode(self, scp_codes: dict, min_likelihood: float = 0.0):
        y = [0] * self.num_classes
        hit = False
        for code, likelihood in scp_codes.items():
            if likelihood < min_likelihood:
                continue
            sup = self.scp_to_super.get(code)
            if sup in self.class_to_index:
                y[self.class_to_index[sup]] = 1
                hit = True
        return y if hit else None
