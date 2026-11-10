import math

from rmscaling.metrics import binary_auc, binary_logloss, ndcg_at_k


def test_auc_handles_ties_and_one_class():
    assert binary_auc([0, 1], [0.1, 0.9]) == 1.0
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5
    assert binary_auc([1, 1], [0.2, 0.9]) is None


def test_logloss_and_group_ndcg():
    assert math.isclose(binary_logloss([0, 1], [0.1, 0.9]), -math.log(0.9))
    value, groups = ndcg_at_k([1, 0, 0, 1], [0.9, 0.1, 0.8, 0.2], [7, 7, 8, 8])
    assert groups == 2
    assert value is not None and 0 < value <= 1
