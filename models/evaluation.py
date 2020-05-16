import numpy as np
import tensorflow as tf

FLOAT_MAX = np.finfo(np.float32).max


def calculate_metric(items, labels):
    dcg = 0.0
    for i, item_id in enumerate(items):
        if item_id in labels:
            dcg += 1.0 / np.log(i + 2)

    ndcg = dcg / (sum((1.0 / np.log(i + 2) for i in range(len(labels)))))
    return ndcg


def evaluate_ndcg(labels, pred_y):
    ndcg_list = []
    for idx, items in enumerate(pred_y):
        ndcg = calculate_metric(items, labels[idx])
        ndcg_list.append(ndcg)
    return np.mean(ndcg_list)


def ndcg_at_k(rank_list, scored_items, k=50):
    _, ranked_scores = tf.math.top_k(scored_items, k=k)
    ndcg = evaluate_ndcg(rank_list, ranked_scores.numpy())
    return ndcg

# class ArenaEvaluator:
#     def _idcg(self, l):
#         return sum((1.0 / np.log(i + 2) for i in range(l)))
#
#     def __init__(self):
#         self._idcgs = [self._idcg(i) for i in range(101)]
#
#     def _ndcg(self, gt, rec):
#         dcg = 0.0
#         for i, r in enumerate(rec):
#             if r in gt:
#                 dcg += 1.0 / np.log(i + 2)
#
#         return dcg / self._idcgs[len(gt)]
#
#     def _eval(self, gt_fname, rec_fname):
#         gt_playlists = load_json(gt_fname)
#         gt_dict = {g["id"]: g for g in gt_playlists}
#         rec_playlists = load_json(rec_fname)
#
#         gt_ids = set([g["id"] for g in gt_playlists])
#         rec_ids = set([r["id"] for r in rec_playlists])
#
#         if gt_ids != rec_ids:
#             raise Exception("결과의 플레이리스트 수가 올바르지 않습니다.")
#
#         rec_song_counts = [len(p["songs"]) for p in rec_playlists]
#         rec_tag_counts = [len(p["tags"]) for p in rec_playlists]
#
#         if set(rec_song_counts) != set([100]):
#             raise Exception("추천 곡 결과의 개수가 맞지 않습니다.")
#
#         if set(rec_tag_counts) != set([10]):
#             raise Exception("추천 태그 결과의 개수가 맞지 않습니다.")
#
#         rec_unique_song_counts = [len(set(p["songs"])) for p in rec_playlists]
#         rec_unique_tag_counts = [len(set(p["tags"])) for p in rec_playlists]
#
#         if set(rec_unique_song_counts) != set([100]):
#             raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")
#
#         if set(rec_unique_tag_counts) != set([10]):
#             raise Exception("한 플레이리스트에 중복된 태그 추천은 허용되지 않습니다.")
#
#         music_ndcg = 0.0
#         tag_ndcg = 0.0
#
#         for rec in rec_playlists:
#             gt = gt_dict[rec["id"]]
#             music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
#             tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])
#
#         music_ndcg = music_ndcg / len(rec_playlists)
#         tag_ndcg = tag_ndcg / len(rec_playlists)
#         score = music_ndcg * 0.85 + tag_ndcg * 0.15
#
#         return music_ndcg, tag_ndcg, score
#
#     def evaluate(self, gt_fname, rec_fname):
#         try:
#             music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
#             print(f"Music nDCG: {music_ndcg:.6}")
#             print(f"Tag nDCG: {tag_ndcg:.6}")
#             print(f"Score: {score:.6}")
#         except Exception as e:
#             print(e)
