from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from config import ATTRIBUTION_DIR, NLA_DIR, REPORT_DIR
from io_utils import ensure_dirs, read_json, write_json


STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "in", "to", "and", "or", "with", "whose",
    "that", "this", "it", "as", "for", "from", "activation", "token", "concept",
    "information", "about", "related", "likely", "represents",
}


def keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text.lower())
    return [word.strip("-'") for word in words if word not in STOPWORDS]


def auto_overlap(explanation: str, source_tokens: list[str], expected_chain: list[str]) -> dict:
    exp_words = set(keywords(explanation))
    token_words = set()
    for token in source_tokens:
        token_words.update(keywords(token))
    expected_words = set()
    for term in expected_chain:
        expected_words.update(keywords(term))
    return {
        "explanation_source_overlap": sorted(exp_words & token_words),
        "explanation_expected_overlap": sorted(exp_words & expected_words),
        "source_expected_overlap": sorted(token_words & expected_words),
    }


def render_case(nla: dict, attr: dict) -> str:
    source_tokens = [
        str(feature.get("source_token_text") or "")
        for feature in attr.get("top_features", [])
        if feature.get("source_token_text")
    ]
    top_feature_lines = []
    for feature in attr.get("top_features", [])[:10]:
        top_feature_lines.append(
            "- rank {rank}: layer={layer}, feature={feature_id}, token={token_pos} {token_text!r}, contribution={contrib:.6g}".format(
                rank=feature.get("rank"),
                layer=feature.get("feature_layer"),
                feature_id=feature.get("feature_id"),
                token_pos=feature.get("source_token_pos"),
                token_text=feature.get("source_token_text"),
                contrib=float(feature.get("contribution") or 0.0),
            )
        )

    overlap = auto_overlap(nla["explanation"], source_tokens, nla.get("expected_chain", []))
    return "\n".join(
        [
            f"## {nla['id']}",
            "",
            f"Group: {nla['group']}",
            f"Prompt: {nla['prompt']}",
            f"Expected chain: {', '.join(nla.get('expected_chain', [])) or '(none)'}",
            f"Target: layer {nla['target_layer']}, token {nla['target_token_pos']} {nla['target_token_text']!r}",
            "",
            f"NLA explanation: {nla['explanation']}",
            f"Round-trip MSE: {nla.get('round_trip_mse')}",
            f"Cosine MSE 2(1-cos): {nla.get('cosine_mse_2_1_minus_cos')}",
            f"Confabulation candidate: {nla.get('confabulation_candidate')}",
            "",
            "Top attribution contributors:",
            *(top_feature_lines or ["- (no resolved contributors)"]),
            "",
            "Automatic overlap:",
            f"- explanation ↔ source tokens: {', '.join(overlap['explanation_source_overlap']) or '(none)'}",
            f"- explanation ↔ expected chain: {', '.join(overlap['explanation_expected_overlap']) or '(none)'}",
            f"- source tokens ↔ expected chain: {', '.join(overlap['source_expected_overlap']) or '(none)'}",
            "",
            "Manual judgment: [match / mismatch / ambiguous]",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NLA claims with layer-32 transcoder attribution.")
    parser.add_argument("--nla-results", default=str(NLA_DIR / "nla_results.json"))
    parser.add_argument("--attribution-results", default=str(ATTRIBUTION_DIR / "attribution_results.json"))
    args = parser.parse_args()

    ensure_dirs(REPORT_DIR)
    nla_rows = {row["id"]: row for row in read_json(Path(args.nla_results))}
    attr_rows = {row["id"]: row for row in read_json(Path(args.attribution_results))}
    common_ids = [case_id for case_id in nla_rows if case_id in attr_rows]

    sections = ["# NLA x Transcoder Grounding Pilot Report", ""]
    summary = Counter()
    machine_rows = []
    for case_id in common_ids:
        nla = nla_rows[case_id]
        attr = attr_rows[case_id]
        source_tokens = [
            str(feature.get("source_token_text") or "")
            for feature in attr.get("top_features", [])
            if feature.get("source_token_text")
        ]
        overlap = auto_overlap(nla["explanation"], source_tokens, nla.get("expected_chain", []))
        if overlap["explanation_source_overlap"] or overlap["source_expected_overlap"]:
            summary["auto_overlap_present"] += 1
        else:
            summary["auto_overlap_absent"] += 1
        machine_rows.append({"id": case_id, **overlap})
        sections.append(render_case(nla, attr))

    sections.insert(
        2,
        "\n".join(
            [
                "## Automatic Summary",
                "",
                f"- Cases compared: {len(common_ids)}",
                f"- Cases with any simple overlap: {summary['auto_overlap_present']}",
                f"- Cases without simple overlap: {summary['auto_overlap_absent']}",
                "",
                "Interpretation still requires manual judgment; this script only prepares the evidence.",
                "",
            ]
        ),
    )

    report_path = REPORT_DIR / "pilot_report.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    write_json(REPORT_DIR / "auto_overlap.json", machine_rows)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()

