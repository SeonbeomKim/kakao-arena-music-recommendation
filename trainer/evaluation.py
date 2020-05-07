import numpy as np


def _ndcg(self, gt, rec):
    dcg = 0.0
    for i, r in enumerate(rec):
        if r in gt:
            dcg += 1.0 / np.log(i + 2)
    return dcg / self._idcgs[len(gt)]