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
    (tmp_path / "REAPER_AI_CACHE" / "current_project.txt").write_text(str(project_cache), encoding="utf-8")
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
    assert torch.allclose(voice["waveform"] + background["waveform"], audio["waveform"], atol=1e-6)
    assert _FakeIsolator.calls == [("inworld/voice-isolator", None, "offline", 0)]


def test_voice_isolator_strength_and_mono(nodes, fake_inworld):
    audio = {"waveform": torch.randn(1, 2, 960), "sample_rate": 48000}
    voice, background = nodes.ReaperVoiceIsolator().isolate(
        audio, mode="streaming", strength=0.5, channels="mono", lookahead_ms="40", device="cpu")
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
        nodes.ReaperVoiceIsolator().isolate({"waveform": torch.zeros(1, 1, 480), "sample_rate": 48000})
