#!/usr/bin/env python3
"""
data_preprocessing.py

Usage:
    python data_preprocessing.py <input_dir> <output_dir>

This script reads the raw Zenodo dataset (unzipped under <input_dir>/dataset or directly under <input_dir>),
performs cleaning, segment merging, sanity checks, stats filtering, 70/20/10 split, and writes
train.jsonl, test.jsonl, and production.jsonl under <output_dir>.
"""
import os
import json
import random
import re
import unicodedata
import shutil
import argparse
from collections import Counter

# reproducible shuffling
random.seed(42)

SEGMENT_ORDER = ["analysis", "argument", "facts", "judgement", "statute"]

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    text = re.sub(r"page \\d+ of \\d+", "", text)
    return text

def get_stats(txt: str):
    words = len(txt.split())
    sents = len([s for s in re.split(r'[\\.\\!?]\\s+', txt) if s.strip()])
    return words, sents

def gather_cases(data_root: str):
    cases = []
    TMP_DIR = os.path.join(data_root, "tmp_merge")
    os.makedirs(TMP_DIR, exist_ok=True)

    # IN-Abs & UK-Abs
    for folder in ("IN-Abs", "UK-Abs"):
        base = os.path.join(data_root, folder)
        for split in ("train-data", "test-data"):
            src = os.path.join(base, split)
            jdir = os.path.join(src, "judgement")
            sdir = os.path.join(src, "summary")
            if not os.path.isdir(jdir):
                continue
            for fn in os.listdir(jdir):
                if not fn.endswith(".txt"):
                    continue
                docp = os.path.join(jdir, fn)
                if folder == "UK-Abs":
                    fullp = os.path.join(sdir, "full", fn)
                    sump = fullp if os.path.isfile(fullp) else os.path.join(sdir, fn)
                else:
                    sump = os.path.join(sdir, fn)
                if os.path.isfile(sump):
                    cases.append((docp, sump, fn))

    # IN-Ext
    base = os.path.join(data_root, "IN-Ext")
    jdir = os.path.join(base, "judgement")
    if os.path.isdir(jdir):
        for fn in os.listdir(jdir):
            if not fn.endswith(".txt"):
                continue
            docp = os.path.join(jdir, fn)
            fullA1 = os.path.join(base, "summary", "full", "A1", fn)
            if os.path.isfile(fullA1):
                cases.append((docp, fullA1, fn))
            else:
                parts = []
                for seg in SEGMENT_ORDER:
                    segp = os.path.join(base, "summary", "segment-wise", "A1", seg, fn)
                    if os.path.isfile(segp):
                        with open(segp, "r", encoding="utf-8", errors="ignore") as f:
                            parts.append(f.read())
                if parts:
                    tmpf = os.path.join(TMP_DIR, fn)
                    with open(tmpf, "w", encoding="utf-8") as f:
                        f.write("\\n\\n".join(parts))
                    cases.append((docp, tmpf, fn))

    return cases

def sanity_checks(cases):
    # drop empty
    kept = []
    for doc, summ, fn in cases:
        if os.path.getsize(doc)==0 or os.path.getsize(summ)==0:
            print(f"Skipping empty file: {fn}")
        else:
            kept.append((doc, summ, fn))
    cases = kept
    print(f"{len(cases)} cases after dropping empty")

    # missing
    missing = [fn for doc, summ, fn in cases if not (os.path.isfile(doc) and os.path.isfile(summ))]
    assert not missing, f"Missing files: {missing}"

    # duplicates
    dupes = [fn for fn,c in Counter(fn for _,_,fn in cases).items() if c>1]
    assert not dupes, f"Duplicate files: {dupes}"

    print("Sanity checks passed")
    return cases

def filter_by_stats(cases):
    filtered = []
    for doc, summ, fn in cases:
        raw_doc = open(doc, "r", encoding="utf-8", errors="ignore").read()
        raw_sum = open(summ, "r", encoding="utf-8", errors="ignore").read()
        doc_clean = clean_text(raw_doc)
        sum_clean = clean_text(raw_sum)

        dw, ds = get_stats(doc_clean)
        sw, ss = get_stats(sum_clean)
        ratio = sw/dw if dw else 0

        if 50 <= sw <= 1500 and 0.01 <= ratio <= 0.5:
            filtered.append((doc, summ, fn))
    print(f"{len(filtered)} cases after stats filtering")
    return filtered

def split_cases(cases):
    random.shuffle(cases)
    N = len(cases)
    n_train = int(0.7 * N)
    n_test  = int(0.2 * N)
    return {
        "train":      cases[:n_train],
        "test":       cases[n_train:n_train+n_test],
        "production": cases[n_train+n_test:]
    }

def dump_jsonl(splits, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for split, subset in splits.items():
        path = os.path.join(out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as out_f:
            for doc, summ, fn in subset:
                raw_doc = open(doc, "r", encoding="utf-8", errors="ignore").read()
                raw_sum = open(summ, "r", encoding="utf-8", errors="ignore").read()
                doc_clean = clean_text(raw_doc)
                sum_clean = clean_text(raw_sum)
                dw, ds = get_stats(doc_clean)
                sw, ss = get_stats(sum_clean)
                ratio = sw/dw if dw else 0
                record = {
                    "filename": fn,
                    "judgement": doc_clean,
                    "summary": sum_clean,
                    "meta": {
                        "doc_words": dw, "doc_sents": ds,
                        "sum_words": sw, "sum_sents": ss,
                        "ratio": ratio
                    }
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {split} ({len(subset)} records) to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",  help="Path where 'dataset' subfolder resides")
    parser.add_argument("output_dir", help="Directory to write JSONL splits")
    args = parser.parse_args()

    # detect root
    root = os.path.join(args.input_dir, "dataset")
    if not os.path.isdir(root):
        root = args.input_dir

    cases = gather_cases(root)
    cases = sanity_checks(cases)
    cases = filter_by_stats(cases)
    splits = split_cases(cases)
    dump_jsonl(splits, args.output_dir)

if __name__ == "__main__":
    main()
