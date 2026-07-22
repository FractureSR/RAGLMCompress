from __future__ import annotations

import csv
import contextlib
import io
import json
import os
import pickle
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from PIL import Image

from evaluation.eval_baselines import main as standalone_main
from evaluation.eval_delta_baselines import (
    _evaluate_codec,
    _print_summary,
    _query_for_retriever,
    main as delta_main,
)
from compression.rac_index import FixedIndexCoder
from compression.baselines import MissingBinaryError
from utils.baseline_data import BaselineSample, load_base_samples, load_eval_split


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(1)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class PreparedTextDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="baseline-tests-")
        root = Path(self.temp.name)
        self.database = root / "db"
        self.database.mkdir()
        self.source = root / "source.jsonl"
        self.documents = [
            ("base alpha " * 60).strip(),
            ("eval beta " * 60).strip(),
            ("base gamma " * 60).strip(),
            ("eval delta " * 60).strip(),
            ("eval epsilon " * 60).strip(),
        ]
        with self.source.open("w", encoding="utf-8") as handle:
            for document in self.documents:
                handle.write(json.dumps({"text": document}) + "\n")

        base_indices = [0, 2]
        eval_indices = [1, 3, 4]
        _write_json(
            self.database / "meta.json",
            {
                "dataset": os.fspath(self.source),
                "base_doc_indices": base_indices,
                "signals": "bm25",
            },
        )
        with (self.database / "eval_docs.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for index in eval_indices:
                handle.write(json.dumps({"text": self.documents[index]}) + "\n")
        _write_json(
            self.database / "base_chunks.json",
            [
                {"id": 0, "doc_idx": 0, "text": self.documents[0]},
                {"id": 1, "doc_idx": 2, "text": self.documents[2]},
            ],
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_persisted_eval_prefix_and_calibration_tail(self) -> None:
        split = load_eval_split(
            self.database, modality="text", n_samples=1, calib_samples=1
        )
        self.assertEqual([sample.source_index for sample in split.test], [1])
        self.assertEqual(split.test[0].eval_position, 0)
        self.assertEqual(split.test[0].sample_id, "doc000000")
        self.assertEqual(
            [sample.source_index for sample in split.calibration], [3]
        )
        self.assertEqual(split.test[0].data, self.documents[1].encode("utf-8"))

    def test_reconstructs_only_prepared_base_samples(self) -> None:
        samples = load_base_samples(self.database, modality="text")
        self.assertEqual([sample.source_index for sample in samples], [0, 2])
        self.assertEqual(
            [sample.data.decode("utf-8") for sample in samples],
            [self.documents[0], self.documents[2]],
        )

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is not installed")
    def test_standalone_evaluator_end_to_end(self) -> None:
        output = Path(self.temp.name) / "standalone.csv"
        exit_code = standalone_main(
            [
                "--database",
                os.fspath(self.database),
                "--modality",
                "text",
                "--codecs",
                "zstd",
                "--n-samples",
                "1",
                "--calib-samples",
                "1",
                "--output",
                os.fspath(output),
            ]
        )
        self.assertEqual(exit_code, 0)
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_index"], "1")
        self.assertEqual(rows[0]["roundtrip_ok"], "1")

    @unittest.skipIf(shutil.which("zli"), "requires OpenZL to be absent")
    def test_allow_missing_writes_header_only_csv(self) -> None:
        output = Path(self.temp.name) / "missing.csv"
        exit_code = standalone_main(
            [
                "--database",
                os.fspath(self.database),
                "--modality",
                "text",
                "--codecs",
                "openzl",
                "--n-samples",
                "1",
                "--allow-missing",
                "--output",
                os.fspath(output),
            ]
        )
        self.assertEqual(exit_code, 0)
        with output.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])

    def test_delta_keep_going_survives_missing_adaptive_zstd(self) -> None:
        class IdentityDelta:
            def encode(self, target, _reference, metadata=None):
                return bytes(target)

            def decode(self, artifact, _reference, metadata=None):
                return bytes(artifact)

            def version(self):
                return "test"

        with mock.patch(
            "compression.baselines.create_codec",
            side_effect=MissingBinaryError("synthetic missing zstd"),
        ), mock.patch(
            "compression.baselines.create_delta_codec",
            return_value=IdentityDelta(),
        ):
            rows = delta_main(
                [
                    "--database",
                    os.fspath(self.database),
                    "--modality",
                    "text",
                    "--codecs",
                    "zstd-patch",
                    "--candidate-policy",
                    "all",
                    "--n-samples",
                    "1",
                    "--keep-going",
                ]
            )
        patch = next(row for row in rows if row.policy == "patch-only")
        adaptive = next(row for row in rows if row.policy == "adaptive")
        self.assertEqual(patch.error, "")
        self.assertEqual(patch.roundtrip_ok, 1)
        self.assertIn("fallback is unavailable", adaptive.error)

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is not installed")
    def test_delta_evaluator_charges_whole_reference_id_and_stop(self) -> None:
        output = Path(self.temp.name) / "delta.csv"
        rows = delta_main(
            [
                "--database",
                os.fspath(self.database),
                "--modality",
                "text",
                "--codecs",
                "zstd-patch",
                "--candidate-policy",
                "all",
                "--n-samples",
                "1",
                "--calib-samples",
                "1",
                "--output",
                os.fspath(output),
            ]
        )
        self.assertEqual(len(rows), 2)
        patch = next(row for row in rows if row.policy == "patch-only")
        adaptive = next(row for row in rows if row.policy == "adaptive")
        # Two full base documents -> one fixed ID bit, plus one stop bit.
        self.assertEqual(patch.side_info_bits, 2.0)
        self.assertEqual(patch.used_reference, 1)
        self.assertEqual(patch.roundtrip_ok, 1)
        self.assertIn(adaptive.side_info_bits, (1.0, 2.0))
        self.assertEqual(adaptive.roundtrip_ok, 1)
        self.assertTrue(output.is_file())


class DeltaFailureIsolationTests(unittest.TestCase):
    def test_adaptive_failure_preserves_valid_patch_row_and_invalidates_summary(self) -> None:
        class IdentityDelta:
            def encode(self, target, _reference, metadata=None):
                return bytes(target)

            def decode(self, artifact, _reference, metadata=None):
                return bytes(artifact)

        class BrokenZstd:
            def encode(self, _target, metadata=None):
                raise RuntimeError("synthetic zstd failure")

        sample = BaselineSample(
            sample_id="doc000000",
            modality="text",
            data=b"target payload",
            metadata={"modality": "text"},
            source_index=1,
            split="test",
            eval_position=0,
        )
        reference = BaselineSample(
            sample_id="doc000002",
            modality="text",
            data=b"reference payload",
            metadata={"modality": "text"},
            source_index=2,
            split="base",
        )
        rows = _evaluate_codec(
            sample=sample,
            modality="text",
            codec_name="fake-delta",
            delta_codec=IdentityDelta(),
            delta_codec_version="test",
            zstd_codec=BrokenZstd(),
            zstd_codec_version="test",
            references=[reference],
            reference_ids=[0],
            index_coder=FixedIndexCoder(1),
            retrieval_s=0.0,
            keep_going=True,
        )
        patch, adaptive = rows
        self.assertEqual(patch.policy, "patch-only")
        self.assertEqual(patch.error, "")
        self.assertEqual(patch.roundtrip_ok, 1)
        self.assertEqual(adaptive.policy, "adaptive")
        self.assertIn("synthetic zstd failure", adaptive.error)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_summary(rows)
        summary = output.getvalue()
        self.assertIn("patch-only", summary)
        self.assertIn("adaptive", summary)
        self.assertIn("INVALID", summary)
        self.assertIn("no aggregate rate reported", summary)


class PreparedMediaDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="baseline-media-tests-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_image_eval_is_whole_rgb_and_retrieval_query_matches_prepared_bytes(self) -> None:
        database = self.root / "image-db"
        database.mkdir()
        source = self.root / "source.png"
        image = Image.new("RGB", (3, 2))
        image.putdata(
            [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (4, 5, 6),
                (7, 8, 9),
                (10, 11, 12),
            ]
        )
        image.save(source)
        meta = {
            "dataset": os.fspath(self.root / "unused-image-dataset"),
            "modality": "image",
            "base_sample_indices": [0],
            "unit": {"patch_width": 2, "patch_height": 1},
        }
        _write_json(database / "meta.json", meta)
        with (database / "eval_samples.pkl").open("wb") as handle:
            pickle.dump([os.fspath(source)], handle)
        split = load_eval_split(database, modality="image")
        sample = split.test[0]
        self.assertEqual(sample.data, image.tobytes("raw", "RGB"))
        self.assertEqual(sample.metadata["width"], 3)
        self.assertEqual(sample.metadata["height"], 2)

        from utils.img_utils import patchify_images_for_compression

        expected = b"".join(
            record.data
            for record in patchify_images_for_compression(
                [os.fspath(source)], [0], patch_width=2, patch_height=1
            )
        )
        self.assertEqual(_query_for_retriever(sample, "image", meta), expected)

    def test_audio_eval_is_header_free_pcm_u8(self) -> None:
        database = self.root / "audio-db"
        database.mkdir()
        pcm = bytes((index * 13) % 256 for index in range(321))
        wav_bytes = _wav_bytes(pcm, 22050)
        _write_json(
            database / "meta.json",
            {
                "dataset": os.fspath(self.root / "unused-audio-dataset"),
                "modality": "audio",
                "base_sample_indices": [0],
            },
        )
        with (database / "eval_samples.pkl").open("wb") as handle:
            pickle.dump([wav_bytes], handle)
        sample = load_eval_split(database, modality="audio").test[0]
        self.assertEqual(sample.data, pcm)
        self.assertEqual(sample.metadata["sample_rate"], 22050)
        self.assertEqual(sample.metadata["sample_format"], "pcm_u8")

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is not installed")
    def test_image_delta_end_to_end_uses_semantic_bpp(self) -> None:
        dataset = self.root / "clic2024" / "bmp"
        dataset.mkdir(parents=True)
        base_path = dataset / "000-base.bmp"
        eval_path = dataset / "001-eval.bmp"
        base_image = Image.new("RGB", (5, 3), (20, 40, 60))
        eval_image = base_image.copy()
        eval_image.putpixel((4, 2), (200, 10, 30))
        base_image.save(base_path)
        eval_image.save(eval_path)

        database = self.root / "image-delta-db"
        database.mkdir()
        _write_json(
            database / "meta.json",
            {
                "dataset": os.fspath(dataset),
                "modality": "image",
                "base_sample_indices": [0],
                "signals": "bm25",
                "kgram": 4,
                "unit": {"patch_width": 2, "patch_height": 1},
            },
        )
        with (database / "eval_samples.pkl").open("wb") as handle:
            pickle.dump([os.fspath(eval_path)], handle)
        with (database / "base_chunks.pkl").open("wb") as handle:
            pickle.dump(
                [{"id": 0, "sample_idx": 0, "ext": "bmp", "data": b"x"}],
                handle,
            )

        references = load_base_samples(database, modality="image")
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].data, base_image.tobytes("raw", "RGB"))
        rows = delta_main(
            [
                "--database",
                os.fspath(database),
                "--modality",
                "image",
                "--codecs",
                "zstd-patch",
                "--candidate-policy",
                "all",
                "--n-samples",
                "1",
            ]
        )
        patch = next(row for row in rows if row.policy == "patch-only")
        self.assertEqual(patch.source_pixels, 15)
        self.assertAlmostEqual(patch.bpp, patch.total_bits / 15)
        self.assertEqual(patch.side_info_bits, 2.0)
        self.assertEqual(patch.roundtrip_ok, 1)

    def test_audio_base_reconstruction_matches_pcm_payload(self) -> None:
        dataset = self.root / "ljspeech_wav"
        dataset.mkdir()
        base_pcm = bytes((index * 7) % 256 for index in range(400))
        eval_pcm = bytes((index * 11) % 256 for index in range(500))
        (dataset / "000-base.wav").write_bytes(_wav_bytes(base_pcm, 16000))
        eval_wav = _wav_bytes(eval_pcm, 22050)
        (dataset / "001-eval.wav").write_bytes(eval_wav)

        database = self.root / "audio-base-db"
        database.mkdir()
        _write_json(
            database / "meta.json",
            {
                "dataset": os.fspath(dataset),
                "modality": "audio",
                "base_sample_indices": [0],
            },
        )
        with (database / "eval_samples.pkl").open("wb") as handle:
            pickle.dump([eval_wav], handle)
        samples = load_base_samples(database, modality="audio")
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].data, base_pcm)
        self.assertEqual(samples[0].metadata["sample_rate"], 16000)


if __name__ == "__main__":
    unittest.main()
