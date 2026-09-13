from pathlib import Path

import torch

from carl_mt.data import ContextCollator, ParallelContextDataset, TranslationExample
from carl_mt.data_audit import audit_data
from carl_mt.clean_splits import filter_cross_split_overlap


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = -1

    def __init__(self) -> None:
        self.src_lang = None
        self.tgt_lang = None

    def __call__(
        self,
        text=None,
        *,
        text_target=None,
        add_special_tokens=True,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        if text_target is not None:
            rows = [[201] + self._encode(value) + [101] for value in text_target]
            width = max(map(len, rows))
            padded = [row + [0] * (width - len(row)) for row in rows]
            return {"input_ids": torch.tensor(padded, dtype=torch.long)}
        return {"input_ids": self._encode(text)}

    @staticmethod
    def _encode(text):
        return [10 + (ord(char) % 70) for char in text]

    @staticmethod
    def build_inputs_with_special_tokens(payload):
        return [100] + list(payload) + [101]

    @staticmethod
    def get_special_tokens_mask(ids, already_has_special_tokens=True):
        return [int(value in {99, 100, 101}) for value in ids]

    @staticmethod
    def pad(features, padding=True, return_tensors="pt"):
        width = max(len(feature["input_ids"]) for feature in features)
        ids = [
            feature["input_ids"] + [0] * (width - len(feature["input_ids"]))
            for feature in features
        ]
        mask = [[int(value != 0) for value in row] for row in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def test_collator_masks_target_padding_and_marks_only_context() -> None:
    collator = ContextCollator(
        FakeTokenizer(),
        context_token_id=101,
        src_lang="zh_CN",
        tgt_lang="en_XX",
        max_source_length=9,
        max_target_length=20,
    )
    batch = collator(
        [
            TranslationExample("题目", "上下文很长", "short", 1),
            TranslationExample("题", "文", "a much longer target", 2),
        ]
    )
    assert (batch["labels"] == -100).any()
    assert torch.equal(batch["decoder_attention_mask"], batch["labels"].ne(-100))
    for row in range(2):
        marker = batch["delimiter_mask"][row].nonzero().item()
        assert (batch["input_ids"][row] == 101).sum() == 2
        assert batch["delimiter_mask"][row].sum() == 1
        assert batch["source_mask"][row, :marker].any()
        assert not batch["source_mask"][row, marker:].any()
        assert not batch["context_mask"][row, : marker + 1].any()
        assert batch["context_mask"][row].sum() >= 1


def test_dataset_rejects_misaligned_files(tmp_path: Path) -> None:
    (tmp_path / "train.src").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "train.ctx").write_text("c\n", encoding="utf-8")
    (tmp_path / "train.tgt").write_text("d\ne\n", encoding="utf-8")
    try:
        ParallelContextDataset(tmp_path, "train")
    except ValueError as error:
        assert "unaligned" in str(error)
    else:
        raise AssertionError("misaligned files should be rejected")


def test_context_interventions_keep_targets_and_derange_contexts(tmp_path: Path) -> None:
    for extension, values in {
        "src": ["s1", "s2", "s3"],
        "ctx": ["c1", "c2", "c3"],
        "tgt": ["t1", "t2", "t3"],
    }.items():
        (tmp_path / f"test.{extension}").write_text(
            "\n".join(values) + "\n", encoding="utf-8"
        )
    dataset = ParallelContextDataset(tmp_path, "test")
    original = [(item.context, item.target) for item in dataset.examples]
    dataset.apply_context_condition("shuffled", seed=3)
    assert all(
        item.context != original[index][0]
        for index, item in enumerate(dataset.examples)
    )
    assert [item.target for item in dataset.examples] == ["t1", "t2", "t3"]


def test_data_audit_detects_normalized_cross_split_input(tmp_path: Path) -> None:
    rows = {
        "train": (["Ａ 标题"], ["语境"], ["target one"]),
        "valid": (["A标题"], ["语境"], ["different target"]),
        "test": (["另一个"], ["上下文"], ["target two"]),
    }
    for split, streams in rows.items():
        for extension, values in zip(("src", "ctx", "tgt"), streams, strict=True):
            (tmp_path / f"{split}.{extension}").write_text(
                "\n".join(values) + "\n", encoding="utf-8"
            )
    report = audit_data(tmp_path)
    assert report["overlap_with_train"]["valid"]["normalized_inputs"] == 1
    assert report["overlap_with_train"]["valid"]["normalized_triples"] == 0


def test_overlap_filter_writes_manifest_and_preserves_train(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "filtered"
    source.mkdir()
    rows = {
        "train": (["Ａ 标题"], ["语境"], ["train target"]),
        "valid": (["A标题", "保留"], ["语境", "验证"], ["drop", "keep"]),
        "test": (["保留", "测试"], ["验证", "语境二"], ["drop", "keep two"]),
    }
    for split, streams in rows.items():
        for extension, values in zip(("src", "ctx", "tgt"), streams, strict=True):
            (source / f"{split}.{extension}").write_text(
                "\n".join(values) + "\n", encoding="utf-8"
            )
    manifest = filter_cross_split_overlap(source, output)
    assert manifest["filtered_rows"] == {"train": 1, "valid": 1, "test": 1}
    assert len(manifest["removed"]) == 2
    assert ParallelContextDataset(output, "train")[0].target == "train target"
