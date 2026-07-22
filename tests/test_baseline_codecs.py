from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

from compression.baselines import (
    CmixCodec,
    FLACCodec,
    OpenVCDiffCodec,
    OpenZLCodec,
    PNGCodec,
    WebPCodec,
    ZstdCodec,
    ZstdDictionaryCodec,
    ZstdPatchCodec,
    train_zstd_dictionary,
)
from compression.baselines.base import CommandResult


class GenericCodecTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is not installed")
    def test_zstd_22_roundtrip(self) -> None:
        payload = (b"whole document baseline\n" * 1000) + bytes(range(256))
        codec = ZstdCodec(timeout_s=30)
        encoded = codec.encode(payload)
        self.assertTrue(encoded.artifact)
        self.assertEqual(codec.decode(encoded.artifact).payload, payload)

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is not installed")
    def test_zstd_patch_roundtrip(self) -> None:
        reference = b"header\n" + (b"shared line\n" * 1000) + b"old tail\n"
        target = b"header\n" + (b"shared line\n" * 1000) + b"new tail\n"
        codec = ZstdPatchCodec(timeout_s=30)
        encoded = codec.encode(target, reference)
        self.assertTrue(encoded.artifact)
        self.assertEqual(codec.decode(encoded.artifact, reference).payload, target)

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is not installed")
    def test_zstd_dictionary_is_base_trained_shared_state(self) -> None:
        samples = [
            (f"record={index:04d};".encode() + b"shared schema payload;" * 40)
            for index in range(128)
        ]
        trained = train_zstd_dictionary(
            samples, dictionary_size=4096, timeout_s=30
        )
        self.assertTrue(trained.dictionary)
        codec = ZstdDictionaryCodec(trained.dictionary, timeout_s=30)
        target = b"record=9999;" + b"shared schema payload;" * 40
        artifact = codec.encode(target).artifact
        self.assertEqual(codec.decode(artifact).payload, target)

    def test_openzl_uses_release_cli_contract(self) -> None:
        codec = OpenZLCodec(binary=sys.executable)
        calls = []

        def fake_run(args, **_kwargs):
            calls.append(tuple(os.fspath(value) for value in args))
            if args[0] == "compress":
                self.assertEqual(args[1:3], ("--profile", "serial"))
                source = Path(args[3])
                self.assertEqual(args[4], "--output")
                Path(args[5]).write_bytes(b"ZL" + source.read_bytes())
            else:
                self.assertEqual(args[0], "decompress")
                self.assertEqual(args[2], "--output")
                Path(args[3]).write_bytes(Path(args[1]).read_bytes()[2:])
            return CommandResult(b"", b"", 0.001)

        codec._run = fake_run  # type: ignore[method-assign]
        payload = b"OpenZL serial whole sample"
        artifact = codec.encode(payload).artifact
        self.assertEqual(codec.decode(artifact).payload, payload)
        self.assertEqual([call[0] for call in calls], ["compress", "decompress"])

    def test_cmix_version_does_not_invoke_unsupported_flag(self) -> None:
        codec = CmixCodec(binary=sys.executable)
        self.assertIn("version flag unavailable", codec.version())

    def test_open_vcdiff_enables_target_matches(self) -> None:
        codec = OpenVCDiffCodec(binary=sys.executable)
        calls = []

        def fake_run(args, **_kwargs):
            calls.append(tuple(os.fspath(value) for value in args))
            values = list(args)
            if args[0] == "encode":
                target = Path(values[values.index("-target") + 1]).read_bytes()
                output = Path(values[values.index("-delta") + 1])
                output.write_bytes(b"VCD" + target)
            else:
                artifact = Path(values[values.index("-delta") + 1]).read_bytes()
                output = Path(values[values.index("-target") + 1])
                output.write_bytes(artifact[3:])
            return CommandResult(b"", b"", 0.001)

        codec._run = fake_run  # type: ignore[method-assign]
        reference = b"dictionary bytes"
        target = b"dictionary bytes with a changed tail"
        artifact = codec.encode(target, reference).artifact
        self.assertIn("-target_matches", calls[0])
        self.assertEqual(
            codec.decode(
                artifact,
                reference,
                metadata={"target_size": 70 * 1024 * 1024},
            ).payload,
            target,
        )
        self.assertIn(
            f"-max_target_file_size={70 * 1024 * 1024}", calls[1]
        )
        self.assertIn(
            f"-max_target_window_size={70 * 1024 * 1024}", calls[1]
        )


class ImageCodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "modality": "image",
            "width": 7,
            "height": 5,
            "channels": 3,
            "mode": "RGB",
        }
        self.payload = bytes(
            (x * 31 + y * 17 + channel * 73) % 256
            for y in range(self.metadata["height"])
            for x in range(self.metadata["width"])
            for channel in range(3)
        )

    def test_png_roundtrip(self) -> None:
        codec = PNGCodec()
        encoded = codec.encode(self.payload, self.metadata)
        self.assertEqual(codec.decode(encoded.artifact, self.metadata).payload, self.payload)

    def test_webp_roundtrip(self) -> None:
        try:
            codec = WebPCodec()
        except Exception as exc:  # pragma: no cover - build-dependent
            self.skipTest(str(exc))
        encoded = codec.encode(self.payload, self.metadata)
        self.assertEqual(codec.decode(encoded.artifact, self.metadata).payload, self.payload)


class AudioCodecTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("flac"), "flac CLI is not installed")
    def test_flac_pcm_u8_roundtrip(self) -> None:
        payload = bytes((index * 29) % 256 for index in range(1600))
        metadata = {
            "modality": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 1,
            "bits_per_sample": 8,
        }
        codec = FLACCodec(timeout_s=30)
        encoded = codec.encode(payload, metadata)
        self.assertTrue(encoded.artifact.startswith(b"fLaC"))
        self.assertEqual(codec.decode(encoded.artifact, metadata).payload, payload)


if __name__ == "__main__":
    unittest.main()
