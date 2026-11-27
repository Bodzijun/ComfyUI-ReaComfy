import json
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_mappings_are_consistent(pack):
    classes = pack.NODE_CLASS_MAPPINGS
    names = pack.NODE_DISPLAY_NAME_MAPPINGS
    assert set(classes) == set(names)
    for key, cls in classes.items():
        spec = cls.INPUT_TYPES()
        assert "required" in spec or "optional" in spec, key
        assert hasattr(cls, cls.FUNCTION), key
        assert len(cls.RETURN_TYPES) >= 0, key


def test_version_matches_pyproject(pack):
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{pack.__version__}"' in text


def test_no_machine_specific_paths():
    for path in [ROOT / "nodes.py", ROOT / "README.md", ROOT / "__init__.py"]:
        text = path.read_text(encoding="utf-8")
        assert "pinokio" not in text.lower(), path
        assert "G:\\" not in text, path


def test_cache_dir_env_override(nodes, tmp_path, monkeypatch):
    target = tmp_path / "cache"
    monkeypatch.setenv("REAPER_AI_CACHE", str(target))
    assert nodes.cache_dir() == target
    assert target.is_dir()


def test_cache_dir_follows_project_pointer(nodes, tmp_path, monkeypatch):
    monkeypatch.delenv("REAPER_AI_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    project_cache = tmp_path / "My Project" / "REAPER_AI_CACHE"
    (tmp_path / "REAPER_AI_CACHE").mkdir()
    (tmp_path / "REAPER_AI_CACHE" / "current_project.txt").write_text(
        str(project_cache), encoding="utf-8"
    )
    assert nodes.cache_dir() == project_cache


class _FakeIsolator:
    calls = []

    def __init__(self, model, device=None, mode="offline", lookahead_frames=0):
        _FakeIsolator.calls.append((model, device, mode, lookahead_frames))

    def isolate(self, audio, sample_rate):
        return audio * 0.25  # "voice" = a quarter of the signal


@pytest.fixture
def fake_inworld(monkeypatch, nodes):
    package = types.ModuleType("inworld")
    module = types.ModuleType("inworld.voice_isolation")
    module.VoiceIsolator = _FakeIsolator
    package.voice_isolation = module
    monkeypatch.setitem(sys.modules, "inworld", package)
    monkeypatch.setitem(sys.modules, "inworld.voice_isolation", module)
    nodes._VOICE_ISOLATOR_CACHE.clear()
    _FakeIsolator.calls.clear()
    yield
    nodes._VOICE_ISOLATOR_CACHE.clear()


def test_voice_isolator_per_channel(nodes, fake_inworld):
    audio = {"waveform": torch.randn(1, 2, 4800), "sample_rate": 44100}
    voice, background = nodes.ReaperVoiceIsolator().isolate(audio)
    assert voice["waveform"].shape == (1, 2, 4800)
    assert voice["sample_rate"] == 44100
    assert torch.allclose(voice["waveform"], audio["waveform"] * 0.25)
    assert torch.allclose(
        voice["waveform"] + background["waveform"], audio["waveform"], atol=1e-6
    )
    assert _FakeIsolator.calls == [("inworld/voice-isolator", None, "offline", 0)]


def test_voice_isolator_strength_and_mono(nodes, fake_inworld):
    audio = {"waveform": torch.randn(1, 2, 960), "sample_rate": 48000}
    voice, background = nodes.ReaperVoiceIsolator().isolate(
        audio,
        mode="streaming",
        strength=0.5,
        channels="mono",
        lookahead_ms="40",
        device="cpu",
    )
    dry = audio["waveform"].mean(dim=1, keepdim=True)
    assert voice["waveform"].shape == (1, 1, 960)
    assert torch.allclose(voice["waveform"], dry * 0.625, atol=1e-6)
    assert torch.allclose(voice["waveform"] + background["waveform"], dry, atol=1e-6)
    assert _FakeIsolator.calls == [("inworld/voice-isolator", "cpu", "streaming", 4)]


def test_voice_isolator_reuses_model(nodes, fake_inworld):
    audio = {"waveform": torch.randn(1, 1, 480), "sample_rate": 48000}
    node = nodes.ReaperVoiceIsolator()
    node.isolate(audio)
    node.isolate(audio)
    assert len(_FakeIsolator.calls) == 1
    node.isolate(audio, keep_model_loaded=False)
    assert nodes._VOICE_ISOLATOR_CACHE == {}


def test_voice_isolator_missing_package(nodes, monkeypatch):
    nodes._VOICE_ISOLATOR_CACHE.clear()
    monkeypatch.setitem(sys.modules, "inworld", None)
    monkeypatch.setitem(sys.modules, "inworld.voice_isolation", None)
    with pytest.raises(RuntimeError, match="inworld-voice-isolation"):
        nodes.ReaperVoiceIsolator().isolate(
            {"waveform": torch.zeros(1, 1, 480), "sample_rate": 48000}
        )


# ---- 0.6.0 nodes: Nemotron RE-USE, ZONOS3 -----------------------------------


def _naive_selective_scan(x, dt, a, b, c):
    batch, length, dim = x.shape
    out = torch.zeros(batch, length, dim)
    for bi in range(batch):
        for d in range(dim):
            h = torch.zeros(a.shape[-1])
            for t in range(length):
                h = (
                    torch.exp(dt[bi, t, d] * a[d]) * h
                    + dt[bi, t, d] * b[bi, t] * x[bi, t, d]
                )
                out[bi, t, d] = (h * c[bi, t]).sum()
    return out


def test_mamba_fallback_scan_matches_naive(nodes):
    torch.manual_seed(0)
    x, dt = torch.randn(2, 9, 3), torch.rand(2, 9, 3) * 2
    a = -torch.rand(3, 4) * 20  # strong decays must stay finite
    b, c = torch.randn(2, 9, 4), torch.randn(2, 9, 4)
    got = nodes._MambaFallback._scan(x, dt, a, b, c)
    assert torch.allclose(got, _naive_selective_scan(x, dt, a, b, c), atol=1e-5)


def test_mamba_fallback_has_mamba_ssm_parameter_names(nodes):
    block = nodes._MambaFallback(d_model=64, d_state=16, d_conv=4, expand=4)
    names = set(dict(block.named_parameters()))
    assert names == {
        "in_proj.weight",
        "conv1d.weight",
        "conv1d.bias",
        "x_proj.weight",
        "dt_proj.weight",
        "dt_proj.bias",
        "A_log",
        "D",
        "out_proj.weight",
    }
    assert block.in_proj.weight.shape == (512, 64)
    assert block.x_proj.weight.shape == (4 + 32, 256)
    y = block(torch.randn(3, 7, 64))
    assert y.shape == (3, 7, 64) and torch.isfinite(y).all()


def test_process_in_chunks_is_seamless(nodes):
    signal = torch.randn(10_000)
    out = nodes._process_in_chunks(signal, 1000, 2.0, 0.5, lambda chunk: chunk * 2)
    assert torch.allclose(out, signal * 2, atol=1e-5)
    whole = nodes._process_in_chunks(signal, 1000, 0, 0, lambda chunk: chunk[:-10])
    assert whole.shape == signal.shape


def test_isolated_import_restores_modules(nodes, tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("VALUE = 'dblm'\n")
    sentinel = types.ModuleType("src")
    monkeypatch.setitem(sys.modules, "src", sentinel)
    cwd = Path.cwd()
    with nodes._IsolatedImport(tmp_path):
        import src

        assert src.VALUE == "dblm"
        assert Path.cwd() == tmp_path
    assert sys.modules["src"] is sentinel
    assert Path.cwd() == cwd


def test_zonos3_controls_mapping(nodes):
    controls = nodes.zonos3_controls(
        {"pace": -1.0, "projection": -1.5, "valence": 2.0},
        {"deep": 2.0},
        "sad",
        1.0,
        "polish",
        1.2,
    )
    assert controls == {
        "delivery.tempo": -1.0,
        "delivery.whisper": 1.5,
        "mood.valence": 2.0,
        "texture.deep": 2.0,
        "emotions.sad": 5.0,
        "accents_l1.pol": 1.2,
    }
    assert nodes.zonos3_controls() == {}
    preset = nodes.zonos3_controls(preset="BBC newsreader")
    assert preset == {
        "accents_en.british": 1.5,
        "texture.authoritative": 3.0,
        "room.clean": 2.0,
    }
    assert nodes.zonos3_controls(extra={"emotions.relieved": 3}) == {
        "emotions.relieved": 3.0
    }


def test_zonos3_languages_include_ukrainian(nodes):
    spec = nodes.ReaperZonos3TTS.INPUT_TYPES()["required"]["language"]
    assert spec[0][0] == "uk" and spec[1]["default"] == "uk"
    assert len(nodes.ZONOS3_LANGUAGES) == 44 == len(set(nodes.ZONOS3_LANGUAGES))


def test_zonos3_server_backend(nodes):
    import http.server
    import io
    import json
    import threading

    import soundfile as sf

    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            buffer = io.BytesIO()
            sf.write(buffer, [0.5] * 441, 44100, format="WAV", subtype="PCM_16")
            data = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("X-Seed", "7")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        reference = {"waveform": torch.randn(1, 2, 48000) * 0.1, "sample_rate": 48000}
        audio, info = nodes.ReaperZonos3TTS().speak(
            "Перший рядок.\nДругий рядок.",
            language="uk",
            emotion="happy",
            pace=0.5,
            seed=42,
            reference_audio=reference,
            gap_ms=100,
            backend="server",
            server_url=f"http://127.0.0.1:{server.server_port}",
        )
    finally:
        server.shutdown()
    assert len(requests) == 2
    path, body = requests[0]
    assert path == "/v2/generate"
    assert body["text"] == "Перший рядок." and body["language"] == "uk"
    assert body["stream"] is False and body["format"] == "wav" and body["seed"] == 42
    assert body["speaker_audio_base64"] and body["speaker_mode"] == "embedding"
    assert body["controls"] == {"delivery.tempo": 0.5, "emotions.happy": 2.5}
    assert audio["sample_rate"] == 44100
    assert audio["waveform"].shape == (1, 1, 441 + 4410 + 441)
    assert json.loads(info)["seeds"] == ["7", "7"]


def test_zonos3_unconfigured_backends_fall_back_to_automatic(nodes, monkeypatch):
    monkeypatch.delenv("ZONOS3_HOME", raising=False)
    monkeypatch.delenv("ZONOS3_PYTHON", raising=False)
    calls = []

    def fake_generate(request, device, timeout):
        calls.append(request["text"])
        return torch.zeros(441), 44100, 5

    monkeypatch.setattr(nodes._Zonos3Runtime, "generate", staticmethod(fake_generate))
    # Old workflows keep backend="server"; nothing listens on this port.
    nodes.ReaperZonos3TTS().speak(
        "Hi", backend="server", server_url="http://127.0.0.1:9"
    )
    nodes.ReaperZonos3TTS().speak("Yo", backend="reference (venv)")
    assert calls == ["Hi", "Yo"]
    with pytest.raises(RuntimeError, match="ZONOS3_HOME"):
        nodes._zonos3_reference_generate("", "", {}, "cpu", "f32", 5)


def test_github_zip_download(nodes, monkeypatch, tmp_path):
    import io
    import urllib.request
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("DBLM-SE-main/src/modeling_dblm.py", "# model\n")
        archive.writestr("DBLM-SE-main/library/__init__.py", "")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout=0: Response(buffer.getvalue())
    )
    repo = tmp_path / "code" / "DBLM-SE"
    nodes._download_github_zip("Yorch233/DBLM-SE", repo, "src/modeling_dblm.py")
    assert (repo / "src" / "modeling_dblm.py").is_file()
    assert [p.name for p in (tmp_path / "code").iterdir()] == ["DBLM-SE"]


class _FakeSEMamba(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, mag, pha):
        return mag * self.gain, pha, None


def test_reuse_node_with_fake_model(nodes, monkeypatch):
    import math

    def stft(y, n_fft, hop_size, win_size, compress_factor, center=True, addeps=False):
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=torch.hann_window(win_size),
            center=center,
            return_complex=True,
        )
        return torch.log1p(spec.abs()), spec.angle(), None

    def istft(mag, pha, n_fft, hop_size, win_size, compress_factor):
        mag = torch.expm1(torch.relu(mag))
        return torch.istft(
            torch.polar(mag, pha),
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=torch.hann_window(win_size),
        )

    cfg = {
        "stft_cfg": {
            "sampling_rate": 8000,
            "n_fft": 320,
            "hop_size": 40,
            "win_size": 320,
        },
        "model_cfg": {"compress_factor": "relu_log1p"},
    }
    calls = []

    def fake_load(device, backend):
        calls.append((device, backend))
        return _FakeSEMamba(cfg), cfg, stft, istft, False

    monkeypatch.setattr(nodes, "_load_reuse", fake_load)
    t = torch.arange(48000) / 48000
    tone = (0.3 * torch.sin(2 * math.pi * 440 * t)).reshape(1, 1, -1).repeat(1, 2, 1)
    speech, removed = nodes.ReaperNemotronReUSE().enhance(
        {"waveform": tone, "sample_rate": 48000}, chunk_seconds=0.5, device="cpu"
    )
    assert calls == [("cpu", "auto")]
    assert speech["sample_rate"] == 48000 and speech["waveform"].shape == (1, 2, 48000)
    core = slice(2400, -2400)
    assert torch.allclose(speech["waveform"][..., core], tone[..., core], atol=1e-3)
    assert torch.allclose(speech["waveform"] + removed["waveform"], tone, atol=1e-6)
    bwe, _ = nodes.ReaperNemotronReUSE().enhance(
        {"waveform": tone[:, :1, :16000], "sample_rate": 16000},
        bandwidth_extension="48000",
        channels="mono",
        chunk_seconds=0,
        device="cpu",
    )
    assert bwe["sample_rate"] == 48000 and bwe["waveform"].shape == (1, 1, 48000)


def test_isolated_import_beats_regular_package_elsewhere(nodes, tmp_path, monkeypatch):
    # Another custom node ships a regular "models" package on sys.path ...
    other = tmp_path / "other"
    (other / "models").mkdir(parents=True)
    (other / "models" / "__init__.py").write_text("OWNER = 'other'\n")
    monkeypatch.syspath_prepend(str(other))
    # ... while the checkout's models/ has no __init__.py (RE-USE layout).
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "net.py").write_text(
        "from .helper import VALUE\nfrom mamba_ssm import Mamba\n"
    )
    (repo / "models" / "helper.py").write_text("VALUE = 'repo'\n")
    shim = types.SimpleNamespace(Mamba="fallback")
    with nodes._IsolatedImport(
        repo, names=("models", "mamba_ssm"), modules={"mamba_ssm": shim}
    ):
        from models.net import VALUE, Mamba

    assert VALUE == "repo" and Mamba == "fallback"
    assert (
        "models" not in sys.modules
        or getattr(sys.modules["models"], "OWNER", None) != "repo"
    )
    assert sys.modules.get("mamba_ssm") is not shim


def test_ensure_modules_installs_only_missing(nodes, monkeypatch):
    import subprocess

    calls = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: calls.append(cmd) or Done()
    )
    # The fake package never becomes importable: one plain install, one
    # upgrade retry, then a clear error naming the module.
    with pytest.raises(RuntimeError, match="fake-pkg==1"):
        nodes._ensure_modules(
            {"json": "json-pkg", "reacomfy_not_a_module_xyz": "fake-pkg==1"}, "test"
        )
    assert len(calls) == 2
    assert calls[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert calls[0][-1] == "fake-pkg==1" and "json-pkg" not in calls[0]
    assert "--upgrade" in calls[1] and "--upgrade" not in calls[0]
    calls.clear()
    nodes._ensure_modules({"json": "json-pkg"}, "test")
    assert calls == []


def test_clearvoice_and_lavasr_install_without_pinned_deps(nodes, monkeypatch):
    calls = []
    monkeypatch.setattr(
        nodes, "_ensure_modules", lambda req, purpose, **kw: calls.append((req, kw))
    )
    nodes._ensure_clearvoice()
    nodes._ensure_lavasr()
    no_deps = [req for req, kw in calls if kw.get("no_deps")]
    assert no_deps[0] == {"clearvoice:ClearVoice": "clearvoice==0.1.2"}
    assert "LavaSR.model:LavaEnhance2" in no_deps[1]
    assert all("git+" not in spec for spec in no_deps[1].values())
    assert "numpy" not in str(calls)


FAKE_ZONOS3_WORKER = r"""
import json, sys, wave
print("loading weights...", flush=True)  # noise the protocol must ignore
sys.stdout.write("@@REACOMFY " + json.dumps({"ready": True}) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    job = json.loads(line)
    with wave.open(job["out"], "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes(b"\x00\x10" * len(job["request"]["text"]))
    sys.stdout.write("@@REACOMFY " + json.dumps({"ok": True, "seed": 99}) + "\n"); sys.stdout.flush()
"""


def test_zonos3_managed_worker_protocol(nodes, monkeypatch, tmp_path):
    base = tmp_path / "zonos3"
    base.mkdir()
    # A stale script on disk must be replaced by the current worker.
    (base / "reacomfy_zonos3_worker.py").write_text(
        "raise SystemExit(3)", encoding="utf-8"
    )
    monkeypatch.setattr(nodes, "ZONOS3_WORKER", FAKE_ZONOS3_WORKER)
    runtime = nodes._Zonos3Runtime
    monkeypatch.setattr(runtime, "base", staticmethod(lambda: base))
    monkeypatch.setattr(
        runtime, "python", classmethod(lambda cls: Path(sys.executable))
    )
    monkeypatch.setattr(runtime, "ensure_installed", classmethod(lambda cls: tmp_path))
    try:
        audio, info = nodes.ReaperZonos3TTS().speak(
            "Раз\nДва три", gap_ms=0, device="cpu", timeout_s=60
        )
        first = runtime.process
        nodes.ReaperZonos3TTS().speak("Ще", device="cpu", timeout_s=60)
        assert runtime.process is first  # the model stays loaded between runs
    finally:
        runtime.stop()
    assert audio["sample_rate"] == 44100
    assert audio["waveform"].shape == (1, 1, len("Раз") + len("Два три"))
    assert json.loads(info)["seeds"] == ["99", "99"]


def test_zonos3_worker_windows_shims(nodes, monkeypatch):
    namespace = {"__name__": "reacomfy_worker_test"}
    exec(compile(nodes.ZONOS3_WORKER, "worker", "exec"), namespace)
    monkeypatch.delitem(sys.modules, "fcntl", raising=False)
    monkeypatch.delenv("ZONOS3_AUDIO_CODEC_DEVICE", raising=False)
    real_import = __import__

    def no_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("No module named 'fcntl'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_fcntl)
    namespace["windows_shims"]("cpu")
    monkeypatch.setattr("builtins.__import__", real_import)
    import os

    assert sys.modules["fcntl"].flock(None, sys.modules["fcntl"].LOCK_EX) is None
    assert os.environ["ZONOS3_AUDIO_CODEC_DEVICE"] == "cpu"


def test_zonos3_edit_request_modes(nodes):
    spans = nodes.zonos3_edit_request(
        "QQ", " fixed  text ", "", "en_us", 1.2, 1.9, 0, 0.7
    )
    assert spans["spans"] == [[1.2, 1.9]] and spans["target_text"] == "fixed text"
    assert "seed" not in spans and "original_text" not in spans
    diff = nodes.zonos3_edit_request("QQ", "a b", "a c", "uk", -1, -1, 5, 0.7)
    assert diff["original_text"] == "a c" and "spans" not in diff and diff["seed"] == 5
    with pytest.raises(ValueError, match="original text"):
        nodes.zonos3_edit_request("QQ", "a b", "", "uk", -1, -1, 0, 0.7)
    with pytest.raises(ValueError, match="corrected text"):
        nodes.zonos3_edit_request("QQ", "  ", "a", "uk", -1, -1, 0, 0.7)


def test_zonos3_speech_edit_node(nodes, monkeypatch):
    seen = {}

    def fake_edit(request, device, timeout):
        seen.update(request)
        return torch.zeros(44100), 44100, 11, {"spans": [[0.5, 0.9]]}

    monkeypatch.setattr(nodes._Zonos3Runtime, "edit", staticmethod(fake_edit))
    clip = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
    audio, info = nodes.ReaperZonos3SpeechEdit().edit(
        clip, "the right word", "the wrong word", "en_us"
    )
    assert audio["sample_rate"] == 48000 and audio["waveform"].shape[-1] == 48000
    assert json.loads(info)["seed"] == 11
    assert seen["original_text"] == "the wrong word" and "audio_base64" in seen


FAKE_ZONOS3_EDIT_WORKER = r"""
import json, sys, wave
edit_folder = sys.argv[4] if len(sys.argv) > 4 else ""
sys.stdout.write("@@REACOMFY " + json.dumps({"ready": True, "edit": bool(edit_folder)}) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    job = json.loads(line)
    with wave.open(job["out"], "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes(b"\x00\x10" * 100)
    reply = {"ok": True, "seed": 3}
    if "edit" in job:
        reply["info"] = {"target": job["edit"]["target_text"], "folder": edit_folder}
    sys.stdout.write("@@REACOMFY " + json.dumps(reply) + "\n"); sys.stdout.flush()
"""


def test_zonos3_edit_worker_restarts_with_adapter(nodes, monkeypatch, tmp_path):
    base = tmp_path / "zonos3"
    base.mkdir()
    monkeypatch.setattr(nodes, "ZONOS3_WORKER", FAKE_ZONOS3_EDIT_WORKER)
    runtime = nodes._Zonos3Runtime
    extras = []
    monkeypatch.setattr(runtime, "base", staticmethod(lambda: base))
    monkeypatch.setattr(
        runtime, "python", classmethod(lambda cls: Path(sys.executable))
    )
    monkeypatch.setattr(runtime, "ensure_installed", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        runtime, "ensure_edit_extras", classmethod(lambda cls: extras.append(1))
    )
    try:
        runtime.generate({"text": "hi"}, "cpu", 60)
        plain = runtime.process
        samples, rate, seed, info = runtime.edit({"target_text": "fixed"}, "cpu", 60)
        assert runtime.process is not plain  # restarted with the edit adapter
        assert info["target"] == "fixed" and info["folder"].endswith("edit_adapter")
        edit_process = runtime.process
        runtime.generate({"text": "again"}, "cpu", 60)
        assert runtime.process is edit_process  # the edit worker also serves TTS
        assert extras == [1, 1] or extras == [1]
    finally:
        runtime.stop()
    assert rate == 44100 and seed == 3


def test_zonos3_worker_script_compiles(nodes):
    compile(nodes.ZONOS3_WORKER, "worker", "exec")
    compile(nodes.ZONOS3_WINSHIM, "winshim", "exec")


def test_speech_edit_keeps_original_outside_the_fix(nodes):
    import numpy as np

    rate = 16000
    rng = np.random.default_rng(0)
    original = rng.standard_normal((2, rate * 4)).astype(np.float32) * 0.1
    mono = original.mean(axis=0)
    # The fix replaces 1.0-1.5 s of the original with a 0.8 s fill; the decoded
    # clip is quieter (gain 0.5) and 5 ms late, like a codec would return it.
    s0, s1, fill_len = rate, int(1.5 * rate), int(0.8 * rate)
    fill = np.sin(np.arange(fill_len) / 7.0).astype(np.float32) * 0.2
    decoded = np.concatenate([mono[:s0], fill, mono[s1:]]) * 0.5
    lag = 80
    decoded = np.concatenate([np.zeros(lag, np.float32), decoded[:-lag]])
    out = nodes._splice_edit_into_original(
        original,
        decoded,
        rate,
        [[1.0, 1.5]],
        [[1.0, 1.0 + fill_len / rate]],
    )
    assert out.shape == (2, decoded.shape[0])
    # Untouched audio is the original itself (both channels, no codec, no lag).
    assert np.allclose(out[:, : s0 - 2000], original[:, : s0 - 2000], atol=1e-6)
    tail = s0 + fill_len + 2000
    assert np.allclose(
        out[:, tail:], original[:, s1 + 2000 :][:, : out.shape[1] - tail], atol=1e-6
    )
    # The fill is level-matched back to the original's loudness.
    inside = out[:, s0 + 100 : s0 + fill_len - 100]
    assert np.allclose(inside, fill[100 : fill_len - 100], atol=0.02)


def _inpaint_test_signal(sample_rate=48000, seconds=1.0):
    import numpy as np

    rng = np.random.default_rng(7)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    voiced = sum(np.sin(2 * np.pi * 140 * h * t + h) / h for h in range(1, 12)) * 0.2
    decay = 0.9 ** np.arange(64)
    breath = np.convolve(rng.standard_normal(len(t)), decay)[: len(t)] * 0.002
    clean = voiced + breath
    dirty = clean.copy()
    for pos in (6000, 15000, 27000, 39000):
        dirty[pos] += 0.5
        dirty[pos + 1] -= 0.3
    dirty[20000:20240] = 0.0  # 5 ms digital dropout
    return clean, dirty


def _inpaint(nodes, data, sample_rate=48000, **kwargs):
    audio = {
        "waveform": torch.from_numpy(data.astype("float32")).view(1, 1, -1),
        "sample_rate": sample_rate,
    }
    out, info = nodes.ReaperAudioInpaint().repair(audio, **kwargs)
    return out["waveform"][0, 0].numpy().astype("float64"), json.loads(info)


def test_audio_inpaint_removes_clicks_and_dropouts(nodes):
    import numpy as np

    clean, dirty = _inpaint_test_signal()
    repaired, info = _inpaint(nodes, dirty)
    before = np.sum((dirty - clean) ** 2)
    after = np.sum((repaired - clean) ** 2)
    assert info["repaired_clicks"] >= 4
    assert info["repaired_dropouts"] == 1
    assert after < before * 0.03  # at least ~15 dB less damage


def test_audio_inpaint_leaves_clean_audio_and_scales_with_strength(nodes):
    import numpy as np

    clean, dirty = _inpaint_test_signal()
    untouched, info = _inpaint(nodes, clean)
    assert info["repaired_regions"] == 0
    assert np.max(np.abs(untouched - clean)) < 1e-6
    full, _ = _inpaint(nodes, dirty, strength=1.0)
    half, _ = _inpaint(nodes, dirty, strength=0.5)
    off, _ = _inpaint(nodes, dirty, strength=0.0)
    assert np.max(np.abs(off - dirty)) < 1e-6
    assert np.allclose(half - dirty, (full - dirty) * 0.5, atol=1e-5)
    removed, _ = _inpaint(nodes, dirty, output="removed")
    assert np.allclose(removed, dirty - full, atol=1e-5)
