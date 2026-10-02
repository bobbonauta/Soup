"""``soup infer --task vision``: image-text-to-text batch inference.

The real model load needs a vision-language checkpoint, so these tests use the
``_VISION_GENERATOR_OVERRIDE`` seam (the same idea as ``_ASR_TRANSCRIBER_OVERRIDE``
for ``--task asr``) and cover row parsing, path containment, output and exit codes.
"""

from __future__ import annotations

import json

import pytest

import soup_cli.commands.infer as infer_mod


def _write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _read_out(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


class TestVisionParts:
    def test_prompt_and_images(self):
        parts = infer_mod._vision_parts({"prompt": "what is it?", "images": ["a.png", "b.png"]})
        assert parts == [("text", "what is it?"), ("image", "a.png"), ("image", "b.png")]

    def test_prompt_without_images_is_allowed(self):
        assert infer_mod._vision_parts({"prompt": "hi"}) == [("text", "hi")]

    def test_content_keeps_interleaving(self):
        row = {"content": [
            {"type": "text", "text": "intro"},
            {"type": "text", "text": "PANEL 1h"},
            {"type": "image", "image": "p1.png"},
            {"type": "text", "text": "PANEL 4h"},
            {"type": "image", "image": "p4.png"},
        ]}
        assert infer_mod._vision_parts(row) == [
            ("text", "intro"), ("text", "PANEL 1h"), ("image", "p1.png"),
            ("text", "PANEL 4h"), ("image", "p4.png"),
        ]

    @pytest.mark.parametrize("row", [
        "not a dict",
        {},
        {"prompt": ""},
        {"prompt": "x", "images": "a.png"},
        {"prompt": "x", "images": [""]},
        {"prompt": "x", "images": [3]},
        {"content": []},
        {"content": ["text"]},
        {"content": [{"type": "audio", "audio": "a.wav"}]},
        {"content": [{"type": "image"}]},
        {"content": [{"type": "text", "text": 5}]},
    ])
    def test_malformed_rows_are_rejected(self, row):
        assert infer_mod._vision_parts(row) is None

    def test_too_many_images_rejected(self):
        row = {"prompt": "x", "images": ["a.png"] * (infer_mod._MAX_IMAGES_PER_ROW + 1)}
        assert infer_mod._vision_parts(row) is None

    def test_reader_drops_bad_lines(self, tmp_path):
        p = tmp_path / "in.jsonl"
        p.write_text(
            json.dumps({"prompt": "ok", "images": ["a.png"]}) + "\n"
            "not json\n"
            "\n"
            + json.dumps({"prompt": ""}) + "\n",
            encoding="utf-8",
        )
        rows = infer_mod._read_vision_rows(p)
        assert rows == [{"prompt": "ok", "images": ["a.png"]}]


class TestResolveMediaPath:
    def test_image_label_in_traversal_error(self, tmp_path):
        with pytest.raises(ValueError, match="image path .* must stay under the image dir"):
            infer_mod._resolve_media_path("../../etc/passwd", tmp_path, "image")

    def test_unc_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="UNC"):
            infer_mod._resolve_media_path("\\\\host\\share\\x.png", tmp_path, "image")

    def test_null_byte_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="null byte"):
            infer_mod._resolve_media_path("a\x00.png", tmp_path, "image")

    def test_contained_relative_path(self, tmp_path):
        assert infer_mod._resolve_media_path("a.png", tmp_path, "image") == str(tmp_path / "a.png")

    def test_asr_wrapper_keeps_audio_wording(self, tmp_path):
        with pytest.raises(ValueError, match="audio path"):
            infer_mod._resolve_asr_audio("../x.wav", tmp_path)


class TestInferVision:
    def test_writes_responses_and_passes_resolved_parts(self, tmp_path, monkeypatch):
        seen = []

        def fake(parts):
            seen.append(parts)
            return f"saw {sum(1 for k, _ in parts if k == 'image')} image(s)"

        monkeypatch.setattr(infer_mod, "_VISION_GENERATOR_OVERRIDE", fake)
        _write_rows(tmp_path / "in.jsonl", [
            {"prompt": "describe", "images": ["a.png", "b.png"]},
            {"content": [{"type": "text", "text": "t"}, {"type": "image", "image": "c.png"}]},
        ])
        monkeypatch.chdir(tmp_path)
        infer_mod._infer_vision(
            model="x", base=None, input_file="in.jsonl", device="cpu",
            output_file="out.jsonl", max_tokens=8, temperature=0.0,
            trust_remote_code=False,
        )
        out = _read_out(tmp_path / "out.jsonl")
        assert [o["response"] for o in out] == ["saw 2 image(s)", "saw 1 image(s)"]
        assert out[0]["prompt"] == "describe" and out[0]["images"] == ["a.png", "b.png"]
        assert "content" in out[1] and "prompt" not in out[1]
        assert all(isinstance(o["seconds"], float) for o in out)
        # image paths are resolved against the input file's directory, text is untouched
        assert seen[0] == [
            ("text", "describe"),
            ("image", str(tmp_path.resolve() / "a.png")),
            ("image", str(tmp_path.resolve() / "b.png")),
        ]

    def test_image_dir_is_the_containment_base(self, tmp_path, monkeypatch):
        (tmp_path / "imgs").mkdir()
        seen = []
        monkeypatch.setattr(
            infer_mod, "_VISION_GENERATOR_OVERRIDE", lambda parts: seen.append(parts) or "ok"
        )
        _write_rows(tmp_path / "in.jsonl", [{"prompt": "p", "images": ["a.png"]}])
        monkeypatch.chdir(tmp_path)
        infer_mod._infer_vision(
            model="x", base=None, input_file="in.jsonl", device="cpu",
            output_file="out.jsonl", max_tokens=8, temperature=0.0,
            trust_remote_code=False, image_dir="imgs",
        )
        assert seen[0][1] == ("image", str(tmp_path.resolve() / "imgs" / "a.png"))

    def test_traversal_row_skipped_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(infer_mod, "_VISION_GENERATOR_OVERRIDE", lambda parts: "ok")
        _write_rows(tmp_path / "in.jsonl", [
            {"prompt": "p", "images": ["../evil.png"]},
            {"prompt": "p", "images": ["fine.png"]},
        ])
        monkeypatch.chdir(tmp_path)
        infer_mod._infer_vision(
            model="x", base=None, input_file="in.jsonl", device="cpu",
            output_file="out.jsonl", max_tokens=8, temperature=0.0,
            trust_remote_code=False,
        )
        out = _read_out(tmp_path / "out.jsonl")
        assert [o["images"] for o in out] == [["fine.png"]]

    def test_generator_oserror_skips_the_row(self, tmp_path, monkeypatch):
        calls = []

        def fake(parts):
            calls.append(parts)
            if len(calls) == 1:
                raise OSError("cannot identify image file")
            return "second"

        monkeypatch.setattr(infer_mod, "_VISION_GENERATOR_OVERRIDE", fake)
        _write_rows(tmp_path / "in.jsonl", [
            {"prompt": "p", "images": ["broken.png"]},
            {"prompt": "p", "images": ["ok.png"]},
        ])
        monkeypatch.chdir(tmp_path)
        infer_mod._infer_vision(
            model="x", base=None, input_file="in.jsonl", device="cpu",
            output_file="out.jsonl", max_tokens=8, temperature=0.0,
            trust_remote_code=False,
        )
        assert [o["response"] for o in _read_out(tmp_path / "out.jsonl")] == ["second"]

    def test_all_rows_failing_exits_2_without_output(self, tmp_path, monkeypatch):
        import click

        monkeypatch.setattr(infer_mod, "_VISION_GENERATOR_OVERRIDE", lambda parts: "ok")
        _write_rows(tmp_path / "in.jsonl", [{"prompt": "p", "images": ["../evil.png"]}])
        monkeypatch.chdir(tmp_path)
        with pytest.raises(click.exceptions.Exit) as exc:
            infer_mod._infer_vision(
                model="x", base=None, input_file="in.jsonl", device="cpu",
                output_file="out.jsonl", max_tokens=8, temperature=0.0,
                trust_remote_code=False,
            )
        assert exc.value.exit_code == 2
        assert not (tmp_path / "out.jsonl").exists()

    def test_no_valid_rows_exits_1(self, tmp_path, monkeypatch):
        import click

        monkeypatch.setattr(infer_mod, "_VISION_GENERATOR_OVERRIDE", lambda parts: "ok")
        (tmp_path / "in.jsonl").write_text("not json\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(click.exceptions.Exit) as exc:
            infer_mod._infer_vision(
                model="x", base=None, input_file="in.jsonl", device="cpu",
                output_file="out.jsonl", max_tokens=8, temperature=0.0,
                trust_remote_code=False,
            )
        assert exc.value.exit_code == 1

    def test_image_dir_outside_cwd_rejected(self, tmp_path, monkeypatch):
        import click

        monkeypatch.setattr(infer_mod, "_VISION_GENERATOR_OVERRIDE", lambda parts: "ok")
        _write_rows(tmp_path / "in.jsonl", [{"prompt": "p", "images": ["a.png"]}])
        monkeypatch.chdir(tmp_path)
        with pytest.raises(click.exceptions.Exit) as exc:
            infer_mod._infer_vision(
                model="x", base=None, input_file="in.jsonl", device="cpu",
                output_file="out.jsonl", max_tokens=8, temperature=0.0,
                trust_remote_code=False, image_dir="..",
            )
        assert exc.value.exit_code == 1


class TestVisionCli:
    def test_help_lists_vision_task_and_image_dir(self):
        import re

        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["infer", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0, result.output
        cleaned = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--image-dir" in cleaned
        assert "vision" in cleaned

    def test_cli_routes_task_vision(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.setattr(infer_mod, "_VISION_GENERATOR_OVERRIDE", lambda parts: "routed")
        _write_rows(tmp_path / "in.jsonl", [{"prompt": "p", "images": ["a.png"]}])
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, [
            "infer", "--model", "x", "--input", "in.jsonl", "--output", "out.jsonl",
            "--task", "vision", "--device", "cpu",
        ])
        assert result.exit_code == 0, result.output
        assert _read_out(tmp_path / "out.jsonl")[0]["response"] == "routed"

    def test_cli_rejects_cuda_graphs_for_vision(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        _write_rows(tmp_path / "in.jsonl", [{"prompt": "p", "images": ["a.png"]}])
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, [
            "infer", "--model", "x", "--input", "in.jsonl", "--output", "out.jsonl",
            "--task", "vision", "--cuda-graphs",
        ])
        assert result.exit_code != 0

    def test_unknown_task_message_lists_vision(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        _write_rows(tmp_path / "in.jsonl", [{"prompt": "p"}])
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, [
            "infer", "--model", "x", "--input", "in.jsonl", "--output", "out.jsonl",
            "--task", "nonsense",
        ])
        assert result.exit_code == 2
        assert "vision" in result.output
