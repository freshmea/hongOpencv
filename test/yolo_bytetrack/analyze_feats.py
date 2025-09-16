#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
저장된 ID-임베딩 DB(.npz)를 불러와서:
1. 전체 ID 갯수 / 임베딩 shape 확인
2. 특정 ID만 따로 추출해서 별도 npz로 저장
3. (옵션) 임베딩들 간의 거리 계산 테스트
"""

import argparse
import logging

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="저장된 임베딩 DB 분석/추출")
    p.add_argument("--feats", type=str, required=True, help="입력 feats_run.npz 경로")
    p.add_argument("--list", action="store_true", help="저장된 ID와 카운트만 출력")
    p.add_argument("--extract_id", type=int, default=None, help="특정 ID만 추출해서 별도 저장")
    p.add_argument("--out", type=str, default=None, help="추출 저장 경로 (예: id5_feats.npz)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = np.load(args.feats)
    ids = data["ids"]
    feats = data["feats"]
    classes = data["classes"]
    counts = data["counts"]
    last_seen = data["last_seen"]

    logging.info(f"파일 로드: {args.feats}")
    logging.info(f"저장된 ID 개수: {len(ids)}")
    logging.info(f"임베딩 shape: {feats.shape} (NumIDs x 512)")

    if args.list:
        logging.info("=== 저장된 ID 목록 ===")
        for i, tid in enumerate(ids):
            logging.info(f"ID={tid}, class={classes[i]}, count={counts[i]}, last_seen={last_seen[i]:.1f}")

    if args.extract_id is not None:
        if args.extract_id not in ids:
            logging.error(f"ID {args.extract_id} 는 데이터에 없습니다.")
            return
        idx = np.where(ids == args.extract_id)[0][0]
        sub = {
            "ids": np.array([ids[idx]], dtype=np.int32),
            "feats": feats[idx:idx+1],
            "classes": np.array([classes[idx]], dtype=np.int32),
            "counts": np.array([counts[idx]], dtype=np.int32),
            "last_seen": np.array([last_seen[idx]], dtype=np.float64),
        }
        out_path = args.out or f"id{args.extract_id}_feats.npz"
        np.savez_compressed(out_path, **sub)
        logging.info(f"ID={args.extract_id} 의 임베딩을 {out_path} 로 저장 완료")


if __name__ == "__main__":
    main()
