"""B1 unified schema + builder tests (synthetic fixtures, no downloads)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema import UNIFIED_COLUMNS, frame_fingerprint, json_dumps, validate_unified_df  # noqa: E402
from src.data.prepare_unified import (  # noqa: E402
    build_faithbench_canonical,
    build_halueval_canonical,
    build_ragtruth_canonical,
)


def make_valid_df(n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "sample_id": f"ds:{i}",
            "source_dataset": "halueval",
            "source_group_id": f"ds:g{i}",
            "task": "qa",
            "domain": "halueval",
            "question": f"q{i}",
            "context": f"c{i}",
            "answer": f"a{i}",
            "label": i % 2,
            "span_annotations": "[]",
            "generator_model": "",
            "official_split": "",
            "experiment_split": "",
            "quality": "",
            "native_record_id": str(i),
            "native_metadata": "{}",
            "label_mapping_version": "b1-labels-v1",
        })
    return pd.DataFrame(rows)


def test_valid_frame_passes():
    assert validate_unified_df(make_valid_df()) is True


def test_missing_column_raises():
    df = make_valid_df().drop(columns=["answer"])
    with pytest.raises(ValueError, match="missing canonical columns"):
        validate_unified_df(df)


def test_duplicate_sample_id_raises():
    df = make_valid_df()
    df.loc[1, "sample_id"] = df.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="globally unique"):
        validate_unified_df(df)


def test_empty_group_id_raises():
    df = make_valid_df()
    df.loc[0, "source_group_id"] = "   "
    with pytest.raises(ValueError, match="source_group_id"):
        validate_unified_df(df)


def test_invalid_label_raises():
    df = make_valid_df()
    df.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        validate_unified_df(df)


def test_unknown_task_raises():
    df = make_valid_df()
    df.loc[0, "task"] = "chat"
    with pytest.raises(ValueError, match="unknown task"):
        validate_unified_df(df)


def test_empty_answer_raises():
    df = make_valid_df()
    df.loc[0, "answer"] = "   "
    with pytest.raises(ValueError, match="answer must be non-empty"):
        validate_unified_df(df)


def test_bad_json_metadata_raises():
    df = make_valid_df()
    df.loc[0, "span_annotations"] = "{not json"
    with pytest.raises(ValueError, match="span_annotations"):
        validate_unified_df(df)


def _halueval_frame():
    return pd.DataFrame([
        {"sample_id": "q_0_correct", "item_idx": 0, "question": "q", "context": "c", "answer": "right", "label": 0, "split": "train"},
        {"sample_id": "q_0_hallucinated", "item_idx": 0, "question": "q", "context": "c", "answer": "wrong", "label": 1, "split": "train"},
        {"sample_id": "q_1_correct", "item_idx": 1, "question": "q2", "context": "c2", "answer": "right2", "label": 0, "split": "val"},
    ])


def test_halueval_paired_rows_share_group():
    df = build_halueval_canonical(_halueval_frame())
    assert len(df) == 3
    assert df.loc[df["sample_id"] == "halueval:q_0_correct", "source_group_id"].iloc[0] == "halueval:q_0"
    assert df.loc[df["sample_id"] == "halueval:q_0_hallucinated", "source_group_id"].iloc[0] == "halueval:q_0"
    assert df.loc[df["sample_id"] == "halueval:q_1_correct", "source_group_id"].iloc[0] == "halueval:q_1"
    assert set(df["label"]) == {0, 1}
    assert set(df["experiment_split"].unique()) == {"train", "val"}
    assert (df["task"] == "qa").all()
    validate_unified_df(df)


def _ragtruth_sources():
    return {
        "100": {"source_id": "100", "task_type": "QA", "source": "MARCO",
                "source_info": {"question": "how to cook?", "passages": "passage 1: heat oil"},
                "prompt": "p"},
        "200": {"source_id": "200", "task_type": "Summary", "source": "CNN/DM",
                "source_info": "The source article text.", "prompt": "p"},
        "300": {"source_id": "300", "task_type": "Data2txt", "source": "Yelp",
                "source_info": {"name": "Subway", "stars": 3.0}, "prompt": "p"},
    }


def _ragtruth_responses():
    return [
        {"id": "1", "source_id": "100", "model": "gpt-3.5-turbo-0613", "temperature": 0.7,
         "labels": [], "split": "train", "quality": "good", "response": "heat the oil"},
        {"id": "2", "source_id": "100", "model": "gpt-4-0613", "temperature": 0.7,
         "labels": [{"start": 4, "end": 9, "text": "the oil", "label_type": "Evident Baseless Info"}],
         "split": "train", "quality": "good", "response": "heat the oil"},
        {"id": "3", "source_id": "200", "model": "llama-2-7b-chat", "temperature": 0.7,
         "labels": [], "split": "test", "quality": "good", "response": "summary text"},
        {"id": "4", "source_id": "300", "model": "mistral-7B-instruct", "temperature": 0.7,
         "labels": [], "split": "test", "quality": "truncated", "response": "Subway is great"},
    ]


def test_ragtruth_multiple_responses_share_group():
    df = build_ragtruth_canonical(_ragtruth_responses(), _ragtruth_sources())
    assert len(df) == 4
    assert (df["source_group_id"] == "ragtruth:100").sum() == 2
    assert (df["source_group_id"] == "ragtruth:200").sum() == 1
    assert (df["source_group_id"] == "ragtruth:300").sum() == 1
    validate_unified_df(df)


def test_ragtruth_task_mapping():
    df = build_ragtruth_canonical(_ragtruth_responses(), _ragtruth_sources())
    qa = df[df["native_record_id"] == "1"].iloc[0]
    assert qa["task"] == "qa"
    assert qa["question"] == "how to cook?"
    assert qa["context"] == "passage 1: heat oil"
    summ = df[df["native_record_id"] == "3"].iloc[0]
    assert summ["task"] == "summarization"
    assert summ["context"] == "The source article text."
    d2t = df[df["native_record_id"] == "4"].iloc[0]
    assert d2t["task"] == "data_to_text"
    assert json_dumps({"name": "Subway", "stars": 3.0}) in d2t["context"]
    assert d2t["official_split"] == "test"
    assert d2t["quality"] == "truncated"
    assert d2t["generator_model"] == "mistral-7B-instruct"


def test_ragtruth_label_and_spans():
    df = build_ragtruth_canonical(_ragtruth_responses(), _ragtruth_sources())
    assert int(df.loc[df["native_record_id"] == "1", "label"].iloc[0]) == 0
    assert int(df.loc[df["native_record_id"] == "2", "label"].iloc[0]) == 1
    spans = df.loc[df["native_record_id"] == "2", "span_annotations"].iloc[0]
    assert "Evident Baseless Info" in spans


def test_ragtruth_missing_source_excluded():
    responses = [{"id": "9", "source_id": "999", "model": "m", "labels": [], "split": "train",
                  "quality": "good", "response": "orphan"}]
    df = build_ragtruth_canonical(responses, _ragtruth_sources())
    assert len(df) == 0
    assert df.attrs["exclusions"] == [{"native_id": "9", "reason": "missing_source"}]


def test_ragtruth_unknown_task_excluded():
    sources = {"500": {"source_id": "500", "task_type": "Chat", "source": "X", "source_info": "s"}}
    responses = [{"id": "10", "source_id": "500", "model": "m", "labels": [], "split": "train",
                  "quality": "good", "response": "hi"}]
    df = build_ragtruth_canonical(responses, sources)
    assert len(df) == 0
    assert df.attrs["exclusions"][0]["reason"] == "unknown_task_type:Chat"


def test_ragtruth_invalid_span_counted():
    responses = [{"id": "11", "source_id": "200", "model": "m", "labels": [{"start": 999, "end": 1005}],
                  "split": "train", "quality": "good", "response": "short"}]
    df = build_ragtruth_canonical(responses, _ragtruth_sources())
    assert len(df) == 1  # row kept, span flagged
    assert df.attrs["n_invalid_spans"] == 1


def _faithbench_sample(batch_id, sample_id, labels, raw_id=None, summary="summary text"):
    return (batch_id, {
        "sample_id": sample_id,
        "source": "Poseidon grossed $181M.",
        "summary": summary,
        "annotations": [{"summary_start": 0, "summary_end": 7, "summary_span": summary[:7], "label": labels}] if labels else [],
        "metadata": {"summarizer": "mistralai/Mistral-7B-Instruct-v0.3", "raw_sample_id": raw_id} if raw_id is not None
                    else {"summarizer": "mistralai/Mistral-7B-Instruct-v0.3"},
    })


def test_faithbench_ids_unique_across_batches():
    samples = [_faithbench_sample(1, 0, ["Questionable"], raw_id=5),
               _faithbench_sample(2, 0, ["Benign"], raw_id=6),
               _faithbench_sample(2, 1, [], raw_id=7)]
    df = build_faithbench_canonical(samples)
    assert df["sample_id"].is_unique
    validate_unified_df(df)


def test_faithbench_group_from_raw_id():
    samples = [_faithbench_sample(1, 0, ["Questionable"], raw_id=5),
               _faithbench_sample(1, 1, [], raw_id=5)]
    df = build_faithbench_canonical(samples)
    assert (df["source_group_id"] == "faithbench:raw_5").all()


def test_faithbench_group_fallback_hash_deterministic():
    samples = [_faithbench_sample(1, 0, [], raw_id=None),
               _faithbench_sample(1, 1, [], raw_id=None)]
    df1 = build_faithbench_canonical(samples)
    df2 = build_faithbench_canonical(samples)
    assert df1["source_group_id"].tolist() == df2["source_group_id"].tolist()
    assert (df1["source_group_id"] == df2["source_group_id"]).all()


def test_faithbench_worst_label_aggregation():
    samples = [_faithbench_sample(1, 0, ["Benign", "Unwanted"], raw_id=1)]
    df = build_faithbench_canonical(samples)
    assert int(df.loc[0, "label"]) == 1  # worst severity wins


def test_faithbench_empty_summary_excluded():
    samples = [_faithbench_sample(1, 0, [], raw_id=1, summary="   ")]
    df = build_faithbench_canonical(samples)
    assert len(df) == 0
    assert df.attrs["exclusions"][0]["reason"] == "empty_summary"


def test_deterministic_fingerprint():
    hal = build_halueval_canonical(_halueval_frame())
    df1 = pd.concat([hal], ignore_index=True).sort_values(["source_dataset", "sample_id"]).reset_index(drop=True)
    df2 = pd.concat([hal], ignore_index=True).sort_values(["source_dataset", "sample_id"]).reset_index(drop=True)
    assert frame_fingerprint(df1) == frame_fingerprint(df2)
