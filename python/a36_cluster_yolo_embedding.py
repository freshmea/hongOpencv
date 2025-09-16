#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
여러 개의 .npz 임베딩 파일(예: feats_run.npz, id5_feats.npz ...)을 불러와서
1) 코사인 유사도 행렬을 계산하고
2) 유사도 임계값(th) 이상인 노드끼리 엣지를 이어 그래프를 만든 뒤
3) 연결 성분(connected components)으로 같은 물체 그룹을 형성합니다.

- 기본은 같은 class인 것끼리만 매칭(옵션으로 해제 가능)
- 결과: 콘솔 출력 + CSV/JSON 저장 가능
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="코사인 유사도 기반 ID 클러스터링")
    p.add_argument("--feats", type=str, nargs="+", required=True,
                   help="입력 npz 파일 경로(여러 개 가능). 예: feats_run.npz id5_feats.npz ...")
    p.add_argument("--th", type=float, default=0.60,
                   help="코사인 유사도 임계값 (기본 0.60). 이 이상이면 같은 물체로 연결")
    p.add_argument("--same_class_only", action="store_true", default=True,
                   help="같은 class 끼리만 매칭 (기본 ON). 끄려면 --no-same_class_only 사용")
    p.add_argument("--no-same_class_only", dest="same_class_only", action="store_false")
    p.add_argument("--min_count", type=int, default=1,
                   help="counts >= min_count 인 ID만 사용 (기본 1)")
    p.add_argument("--save_json", type=str, default=None, help="클러스터 결과 JSON 저장 경로")
    p.add_argument("--save_csv", type=str, default=None, help="클러스터 결과 CSV 저장 경로")
    p.add_argument("--print_pairs_topk", type=int, default=0,
                   help="상위 유사도 pair K개를 출력 (0이면 건너뜀)")
    return p.parse_args()


def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    # 필수 키 체크
    required = ["ids", "feats", "classes", "counts", "last_seen"]
    for k in required:
        if k not in data:
            raise ValueError(f"{path} 에 키 '{k}'가 없습니다. 생성 스크립트 버전을 확인하세요.")
    return {k: data[k] for k in required}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    # 2D (N, D)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def build_graph_cosine(names: List[str], feats: np.ndarray, classes: np.ndarray,
                       th: float, same_class_only: bool) -> Dict[int, List[int]]:
    """
    names[i]: 각 노드의 식별 문자열(파일#ID 형태)
    feats: (N, D) unit-normalized
    classes: (N,)
    th: cosine similarity threshold
    same_class_only: True면 class가 같은 경우에만 엣지 인정
    return: 인접 리스트 그래프(dict: idx -> [neighbors ...])
    """
    N = len(names)
    sim = feats @ feats.T  # (N, N), 코사인 유사도 (정규화 가정)
    graph: Dict[int, List[int]] = {i: [] for i in range(N)}

    for i in range(N):
        # 대칭이므로 j > i만 체크
        for j in range(i + 1, N):
            if same_class_only and classes[i] != classes[j]:
                continue
            if sim[i, j] >= th:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """그래프 연결 성분 리스트 반환"""
    N = len(graph)
    visited = [False] * N
    comps = []

    for s in range(N):
        if visited[s]:
            continue
        # BFS/DFS
        stack = [s]
        visited[s] = True
        comp = [s]
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        comps.append(sorted(comp))
    return comps


def save_clusters_json(path: str, clusters: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)


def save_clusters_csv(path: str, clusters: List[Dict]):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "size", "member_idx", "name", "id", "class", "count", "last_seen"])
        for c in clusters:
            for m in c["members"]:
                writer.writerow([
                    c["cluster_id"], c["size"], m["idx"], m["name"],
                    m["id"], m["class"], m["count"], f"{m['last_seen']:.3f}"
                ])


def main():
    args = parse_args()

    # 여러 파일 로드 후 concat
    all_names, all_ids, all_feats, all_classes, all_counts, all_last = [], [], [], [], [], []
    for path in args.feats:
        pack = load_npz(path)
        ids = pack["ids"].astype(np.int32)
        feats = pack["feats"].astype(np.float32)
        classes = pack["classes"].astype(np.int32)
        counts = pack["counts"].astype(np.int32)
        last = pack["last_seen"].astype(np.float64)

        # filter by min_count
        mask = counts >= args.min_count
        ids, feats, classes, counts, last = ids[mask], feats[mask], classes[mask], counts[mask], last[mask]

        # 각 샘플의 “식별 이름”: 파일명#ID
        tag = os.path.basename(path)
        names = [f"{tag}#{tid}" for tid in ids]

        all_names.extend(names)
        all_ids.append(ids)
        all_feats.append(feats)
        all_classes.append(classes)
        all_counts.append(counts)
        all_last.append(last)

    if len(all_names) == 0:
        print("선택된 데이터가 없습니다. --min_count 조건을 낮춰보세요.")
        return

    all_ids = np.concatenate(all_ids)
    all_feats = np.vstack(all_feats)
    all_classes = np.concatenate(all_classes)
    all_counts = np.concatenate(all_counts)
    all_last = np.concatenate(all_last)

    # 정규화 (코사인)
    feats_norm = normalize_rows(all_feats)

    # 그래프 만들기
    graph = build_graph_cosine(all_names, feats_norm, all_classes, args.th, args.same_class_only)
    comps = connected_components(graph)

    # 결과 정리
    clusters = []
    for cid, comp in enumerate(sorted(comps, key=len, reverse=True), start=1):
        members = []
        for idx in comp:
            members.append({
                "idx": int(idx),
                "name": all_names[idx],
                "id": int(all_ids[idx]),
                "class": int(all_classes[idx]),
                "count": int(all_counts[idx]),
                "last_seen": float(all_last[idx]),
            })
        clusters.append({
            "cluster_id": cid,
            "size": len(comp),
            "members": members
        })

    # 요약 출력
    print(f"\n총 노드: {len(all_names)}개, 형성된 클러스터: {len(clusters)}개 (유사도 임계값={args.th}, same_class_only={args.same_class_only})")
    for c in clusters[:10]:  # 상위 10개만 간단히 출력
        mnames = [m["name"] for m in c["members"]]
        print(f"- Cluster#{c['cluster_id']} (size={c['size']}): {', '.join(mnames)}")

    # 상위 유사 pair 보고 싶을 때
    if args.print_pairs_topk > 0:
        sim = feats_norm @ feats_norm.T
        triu_i, triu_j = np.triu_indices(sim.shape[0], k=1)
        pairs = [(sim[i, j], i, j) for i, j in zip(triu_i, triu_j)]
        pairs.sort(reverse=True)
        print(f"\n== Top {args.print_pairs_topk} cosine-sim pairs ==")
        k = min(args.print_pairs_topk, len(pairs))
        for s, i, j in pairs[:k]:
            if args.same_class_only and all_classes[i] != all_classes[j]:
                continue
            print(f"{s:.4f} : {all_names[i]} (class={all_classes[i]}) <-> {all_names[j]} (class={all_classes[j]})")

    # 저장 옵션
    if args.save_json:
        save_clusters_json(args.save_json, clusters)
        print(f"\nJSON 저장: {args.save_json}")
    if args.save_csv:
        save_clusters_csv(args.save_csv, clusters)
        print(f"CSV 저장: {args.save_csv}")


if __name__ == "__main__":
    main()