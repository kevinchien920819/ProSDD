"""Evaluate ProSDD score files with the imported CM metrics."""

import json
from pathlib import Path

import numpy as np

from .calculate_metrics import calculate_minDCF_EER_CLLR_actDCF


def _rows(path):
    with open(path, encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            parts = line.split()
            if parts:
                yield line_number, parts


def load_protocol_labels(protocol_path):
    """Read CM labels from ASVspoof 2019, ASVspoof5, or two-column keys.

    Headerless formats have 5, 10, or 2 columns respectively. Named TSV
    headers may use filename/trial_anon and cm-label/cm_label.
    """
    labels = {}
    columns = None
    width = None
    for line_number, parts in _rows(protocol_path):
        if columns is None:
            header = [part.lower() for part in parts]
            id_name = next((name for name in ("filename", "trial_anon") if name in header), None)
            label_name = next((name for name in ("cm-label", "cm_label") if name in header), None)
            width = len(parts)
            if id_name is not None and label_name is not None:
                columns = (header.index(id_name), header.index(label_name))
                continue
            columns = {2: (0, 1), 5: (1, 4), 10: (1, 8)}.get(width)
            if columns is None:
                raise ValueError(
                    f"{protocol_path}:{line_number}: expected 2-column keys, "
                    "5-column ASVspoof 2019, 10-column ASVspoof5, or a named TSV header"
                )
        if len(parts) != width:
            raise ValueError(f"{protocol_path}:{line_number}: inconsistent protocol column count")
        utt_id, label = parts[columns[0]], parts[columns[1]].lower()
        if label not in ("bonafide", "spoof"):
            raise ValueError(
                f"{protocol_path}:{line_number}: {utt_id} has invalid CM label {label!r}; "
                "expected bonafide or spoof"
            )
        if utt_id in labels:
            raise ValueError(f"{protocol_path}:{line_number}: duplicate utterance ID {utt_id}")
        labels[utt_id] = label
    if not labels:
        raise ValueError(f"{protocol_path}: no CM labels found")
    if set(labels.values()) != {"bonafide", "spoof"}:
        raise ValueError(f"{protocol_path}: evaluation requires both bonafide and spoof samples")
    return labels


def load_scores(score_path):
    """Read the two-column utterance-ID/score output of main_eval.py."""
    scores = {}
    first_row = True
    for line_number, parts in _rows(score_path):
        if first_row:
            first_row = False
            if parts in (["filename", "cm-score"], ["trial_anon", "cm-score"]):
                continue
        if len(parts) != 2:
            raise ValueError(f"{score_path}:{line_number}: expected two columns: utterance_id score")
        utt_id, value = parts
        if utt_id in scores:
            raise ValueError(f"{score_path}:{line_number}: duplicate utterance ID {utt_id}")
        try:
            score = float(value)
        except ValueError as exc:
            raise ValueError(f"{score_path}:{line_number}: {utt_id} has non-numeric score {value!r}") from exc
        if not np.isfinite(score):
            raise ValueError(f"{score_path}:{line_number}: {utt_id} score must be finite")
        scores[utt_id] = score
    if not scores:
        raise ValueError(f"{score_path}: no scores found")
    return scores


def evaluate_score_file(score_path, protocol_path, output_path=None):
    """Align every score with a protocol label and return/save CM metrics.

    Higher scores indicate bonafide. Scores are passed through without
    calibration; Cllr and actDCF interpret the supplied scores as LLRs.
    EER is returned as a fraction, with eer_percent provided for display.
    """
    if output_path is not None and Path(output_path).resolve() in {
        Path(score_path).resolve(), Path(protocol_path).resolve()
    }:
        raise ValueError("Metrics output must differ from score and protocol input paths")

    scores_by_id = load_scores(score_path)
    labels_by_id = load_protocol_labels(protocol_path)
    missing = labels_by_id.keys() - scores_by_id.keys()
    extra = scores_by_id.keys() - labels_by_id.keys()
    if missing or extra:
        raise ValueError(
            "Score/protocol utterance IDs do not match: "
            f"{len(missing)} missing scores {sorted(missing)[:3]}, "
            f"{len(extra)} scores without labels {sorted(extra)[:3]}"
        )
    scores = np.asarray(list(scores_by_id.values()), dtype=np.float64)
    keys = np.asarray([labels_by_id[utt_id] for utt_id in scores_by_id])
    min_dcf, eer, cllr, act_dcf = calculate_minDCF_EER_CLLR_actDCF(
        scores, keys, output_file=None, printout=False,
    )
    metrics = {
        "min_dcf": float(min_dcf),
        "eer": float(eer),
        "eer_percent": float(eer * 100),
        "cllr": float(cllr),
        "act_dcf": float(act_dcf),
        "sample_count": int(keys.size),
        "bonafide_count": int(np.sum(keys == "bonafide")),
        "spoof_count": int(np.sum(keys == "spoof")),
        "cost_model": {"Pspoof": 0.05, "Cmiss": 1, "Cfa": 10},
        "score_path": str(score_path),
        "protocol_path": str(protocol_path),
        "score_handling": "as supplied; higher is bonafide; no calibration",
    }
    if not all(np.isfinite(metrics[name]) for name in ("min_dcf", "eer", "cllr", "act_dcf")):
        raise ValueError("CM evaluator returned a non-finite metric")
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metrics


def print_metrics(metrics):
    print(
        f"EER={metrics['eer_percent']:.6f}% | "
        f"Cllr={metrics['cllr']:.6f} bits | "
        f"minDCF={metrics['min_dcf']:.6f} | actDCF={metrics['act_dcf']:.6f}",
        flush=True,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="從 ProSDD 分數檔計算 EER、Cllr、minDCF、actDCF。")
    parser.add_argument("--score_path", required=True, help="兩欄 utterance_id score 分數檔")
    parser.add_argument("--protocol_path", required=True, help="含 bonafide／spoof 標籤的 protocol")
    parser.add_argument("--save_metrics_to", help="JSON 輸出路徑；省略時只印出指標")
    args = parser.parse_args()
    try:
        metrics = evaluate_score_file(args.score_path, args.protocol_path, args.save_metrics_to)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print_metrics(metrics)
    if args.save_metrics_to:
        print(f"Metrics saved to {args.save_metrics_to}", flush=True)
