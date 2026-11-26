import json
import os
import shutil
import sys
import time
from pathlib import Path

import soundfile as sf
import torch


def ensure_clearvoice_librosa_compat():
    """Restore librosa.util.normalize removed in librosa 0.11 for ClearVoice."""
    try:
        import librosa.util as librosa_util

        if not hasattr(librosa_util, "normalize"):
            import numpy as np

            def normalize(
                array, norm=np.inf, axis=0, threshold=None, fill=None, **kwargs
            ):
                values = np.asarray(array)
                if threshold is None:
                    threshold = 1e-10
                length = np.linalg.norm(values, ord=norm, axis=axis, keepdims=True)
                return np.divide(values, np.maximum(length, threshold))

            librosa_util.normalize = normalize
    except Exception:
        pass


def phase_safe_audio_mix(dry, wet, strength, sample_rate, highpass_hz=None):
    """Blend magnitudes while retaining dry phase to avoid comb-filter flanging."""
    import numpy as np

    n = min(len(dry), len(wet))
    dry = np.asarray(dry[:n], dtype=np.float64)
    wet = np.asarray(wet[:n], dtype=np.float64)
    dry_spec = np.fft.rfft(dry)
    wet_spec = np.fft.rfft(wet)
    amount = np.full(dry_spec.shape, float(strength), dtype=np.float64)
    if highpass_hz is not None:
        freq = np.fft.rfftfreq(n, 1.0 / float(sample_rate))
        lo = max(0.0, float(highpass_hz) * 0.65)
        hi = max(lo + 1.0, float(highpass_hz) * 1.35)
        amount *= np.clip((freq - lo) / (hi - lo), 0.0, 1.0)
    magnitude = (1.0 - amount) * np.abs(dry_spec) + amount * np.abs(wet_spec)
    mixed = np.fft.irfft(magnitude * np.exp(1j * np.angle(dry_spec)), n=n)
    return mixed.astype(np.float32)


def band_limited_model_mix(dry, wet, strength, sample_rate, cutoff_hz):
    """Blend the model only above a smooth crossover, keeping dry phase below it."""
    import numpy as np

    n = min(len(dry), len(wet))
    dry = np.asarray(dry[:n], dtype=np.float64)
    wet = np.asarray(wet[:n], dtype=np.float64)
    if cutoff_hz <= 0:
        return (dry + float(strength) * (wet - dry)).astype(np.float32)
    dry_spec = np.fft.rfft(dry)
    wet_spec = np.fft.rfft(wet)
    freq = np.fft.rfftfreq(n, 1.0 / float(sample_rate))
    width = max(500.0, min(1500.0, float(cutoff_hz) * 0.25))
    mask = np.clip((freq - (float(cutoff_hz) - width)) / (2.0 * width), 0.0, 1.0)
    mixed = dry_spec + mask * float(strength) * (wet_spec - dry_spec)
    return np.fft.irfft(mixed, n=n).astype(np.float32)


def ensure_yue2_path() -> None:
    """Make sibling YuE2-ComfyUI packages importable from any ComfyUI launch cwd."""
    custom_nodes = Path(__file__).resolve().parent.parent
    candidates = [custom_nodes / "YuE2-ComfyUI", custom_nodes / "yue2-comfyui"]
    for candidate in candidates:
        if (candidate / "yue2_comfy" / "load_midi.py").is_file():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


def cache_dir() -> Path:
    configured = os.environ.get("REAPER_AI_CACHE", "")
    if configured:
        path = Path(configured)
    else:
        path = Path.home() / "REAPER_AI_CACHE"
        pointer = path / "current_project.txt"
        if pointer.is_file():
            pointed = pointer.read_text(encoding="utf-8").strip()
            if pointed:
                path = Path(pointed)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_input_path() -> Path:
    request = cache_dir() / "reaper_to_comfy.json"
    if not request.is_file():
        raise FileNotFoundError(f"REAPER input request not found: {request}")
    data = json.loads(request.read_text(encoding="utf-8"))
    path = Path(str(data.get("path", ""))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"REAPER input audio not found: {path}")
    return path


def request_data() -> dict:
    primary = cache_dir() / "reaper_to_comfy.json"
    candidates = [primary]
    fallback = Path.home() / "REAPER_AI_CACHE" / "reaper_to_comfy.json"
    if fallback != primary:
        candidates.append(fallback)
    for request in candidates:
        if request.is_file():
            return json.loads(request.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "REAPER input request not found in: "
        + ", ".join(str(path) for path in candidates)
    )


class ReaperInput:
    DESCRIPTION = "Loads the audio file prepared by REAPER. Inputs: optional refresh_token. Outputs: AUDIO and source_path."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"refresh_token": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "source_path")
    FUNCTION = "load"
    CATEGORY = "REAPER AI/Inputs"

    @classmethod
    def IS_CHANGED(cls, refresh_token=""):
        manifest = cache_dir() / "reaper_to_comfy.json"
        try:
            stat = manifest.stat()
            return f"{refresh_token}|{stat.st_mtime_ns}|{stat.st_size}"
        except OSError:
            return f"{refresh_token}|missing|{time.time_ns()}"

    def load(self, refresh_token=""):
        path = resolve_input_path()
        samples, rate = sf.read(str(path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(samples.T).unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": int(rate)}, str(path))


class ReaperReferenceInput:
    DESCRIPTION = "Loads the reference audio path prepared by REAPER. Use this as the reference_audio input for EQ Match or Dynamic Voice Match. Outputs AUDIO and reference_path."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"refresh_token": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "reference_path")
    FUNCTION = "load"
    CATEGORY = "REAPER AI/Inputs"

    @classmethod
    def IS_CHANGED(cls, refresh_token=""):
        manifest = cache_dir() / "reaper_to_comfy.json"
        try:
            stat = manifest.stat()
            return f"{refresh_token}|{stat.st_mtime_ns}|{stat.st_size}"
        except OSError:
            return f"{refresh_token}|missing|{time.time_ns()}"

    def load(self, refresh_token=""):
        data = request_data()
        raw_path = str(data.get("reference_path", ""))
        if not raw_path and data.get("source") == "reference":
            raw_path = str(data.get("path", ""))
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"REAPER reference audio not found: {path}")
        samples, rate = sf.read(str(path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(samples.T).unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": int(rate)}, str(path))


class ReaperEQMatch:
    DESCRIPTION = "Matches target audio to a reference using STFT/iSTFT DSP, then applies peak-safe output normalization. Useful for reducing exaggerated VoxCPM2 or voice-clone frequencies without changing timing."

    @staticmethod
    def _apply_peak_ceiling(waveform, ceiling_db):
        """Attenuate only when needed; never boost a quiet matched file."""
        peak = waveform.abs().amax()
        ceiling = waveform.new_tensor(10.0).pow(float(ceiling_db) / 20.0)
        gain = torch.minimum(waveform.new_tensor(1.0), ceiling / peak.clamp_min(1e-12))
        return waveform * gain

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_audio": ("AUDIO",),
                "reference_audio": ("AUDIO",),
                "match_strength": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_correction_db": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.5, "max": 18.0, "step": 0.5},
                ),
                "smoothing_bins": (
                    "INT",
                    {"default": 9, "min": 3, "max": 127, "step": 2},
                ),
                "low_frequency_hz": (
                    "FLOAT",
                    {"default": 80.0, "min": 0.0, "max": 20000.0, "step": 10.0},
                ),
                "high_frequency_hz": (
                    "FLOAT",
                    {"default": 16000.0, "min": 1000.0, "max": 24000.0, "step": 100.0},
                ),
                "output_ceiling_db": (
                    "FLOAT",
                    {"default": -1.0, "min": -6.0, "max": -0.1, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "match"
    CATEGORY = "REAPER AI/Enhancement"

    def match(
        self,
        target_audio,
        reference_audio,
        match_strength=0.65,
        max_correction_db=6.0,
        smoothing_bins=9,
        low_frequency_hz=80.0,
        high_frequency_hz=16000.0,
        output_ceiling_db=-1.0,
    ):
        if target_audio is None or reference_audio is None:
            raise ValueError(
                "REAPER EQ Match requires target_audio and reference_audio"
            )
        target = target_audio["waveform"].float()
        reference = reference_audio["waveform"].float()
        target_rate = int(target_audio["sample_rate"])
        reference_rate = int(reference_audio["sample_rate"])
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        batch, channels, target_length = target.shape
        reference = reference.reshape(-1, reference.shape[-1])
        if reference_rate != target_rate:
            try:
                import torchaudio

                reference = torchaudio.functional.resample(
                    reference, reference_rate, target_rate
                )
            except Exception:
                length = max(
                    1, int(round(reference.shape[-1] * target_rate / reference_rate))
                )
                reference = torch.nn.functional.interpolate(
                    reference.unsqueeze(1),
                    size=length,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
        target_flat = target.reshape(-1, target_length)
        n_fft, hop = 2048, 512
        window = torch.hann_window(n_fft, device=target.device, dtype=target.dtype)
        target_spec = torch.stft(
            target_flat,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        reference_spec = torch.stft(
            reference.to(target.device),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        eps = 1e-7
        target_db = 20.0 * torch.log10(target_spec.abs().clamp_min(eps)).mean(
            dim=2
        ).mean(dim=0)
        reference_db = 20.0 * torch.log10(reference_spec.abs().clamp_min(eps)).mean(
            dim=2
        ).mean(dim=0)
        correction = reference_db - target_db
        kernel = max(3, int(smoothing_bins) | 1)
        kernel = min(
            kernel,
            correction.numel() if correction.numel() % 2 else correction.numel() - 1,
        )
        if kernel >= 3:
            correction = torch.nn.functional.avg_pool1d(
                correction.view(1, 1, -1), kernel, 1, kernel // 2
            ).view(-1)[: correction.numel()]
        frequencies = torch.linspace(
            0.0, target_rate / 2.0, correction.numel(), device=target.device
        )
        mask = (frequencies >= float(low_frequency_hz)) & (
            frequencies <= min(float(high_frequency_hz), target_rate / 2.0)
        )
        # Match spectral shape, not the reference file's overall loudness.
        # Without removing this broadband offset, max_correction_db can act
        # like a gain control across nearly the entire signal.
        if bool(mask.any()):
            correction = correction - correction[mask].mean()
        correction = correction.mul(float(match_strength)).clamp(
            -float(max_correction_db), float(max_correction_db)
        )
        correction = torch.where(mask, correction, torch.zeros_like(correction))
        gain = torch.pow(torch.tensor(10.0, device=target.device), correction / 20.0)
        matched = torch.istft(
            target_spec * gain.view(1, -1, 1),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            length=target_length,
        )
        matched = self._apply_peak_ceiling(matched, output_ceiling_db)
        return (
            {
                "waveform": matched.reshape(batch, channels, target_length),
                "sample_rate": target_rate,
            },
        )


class ReaperDynamicEQMatch(ReaperEQMatch):
    DESCRIPTION = "Dynamically stabilizes a voice recording against a reference spectrum. Uses slow STFT correction, voice gating and limited EQ moves to reduce interview mic-distance and angle changes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_audio": ("AUDIO",),
                "reference_audio": ("AUDIO",),
                "match_strength": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_correction_db": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.5, "max": 12.0, "step": 0.5},
                ),
                "smoothing_bins": (
                    "INT",
                    {"default": 21, "min": 3, "max": 127, "step": 2},
                ),
                "temporal_smoothing": (
                    "FLOAT",
                    {"default": 0.88, "min": 0.0, "max": 0.99, "step": 0.01},
                ),
                "voice_gate_db": (
                    "FLOAT",
                    {"default": -35.0, "min": -80.0, "max": -6.0, "step": 1.0},
                ),
                "low_frequency_hz": (
                    "FLOAT",
                    {"default": 80.0, "min": 0.0, "max": 20000.0, "step": 10.0},
                ),
                "high_frequency_hz": (
                    "FLOAT",
                    {"default": 16000.0, "min": 1000.0, "max": 24000.0, "step": 100.0},
                ),
                "output_ceiling_db": (
                    "FLOAT",
                    {"default": -1.0, "min": -6.0, "max": -0.1, "step": 0.1},
                ),
            }
        }

    def match(
        self,
        target_audio,
        reference_audio,
        match_strength=0.45,
        max_correction_db=3.0,
        smoothing_bins=21,
        temporal_smoothing=0.88,
        voice_gate_db=-35.0,
        low_frequency_hz=80.0,
        high_frequency_hz=16000.0,
        output_ceiling_db=-1.0,
    ):
        if target_audio is None or reference_audio is None:
            raise ValueError(
                "REAPER Dynamic EQ Match requires target_audio and reference_audio"
            )
        target = target_audio["waveform"].float()
        reference = reference_audio["waveform"].float()
        target_rate = int(target_audio["sample_rate"])
        reference_rate = int(reference_audio["sample_rate"])
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        batch, channels, target_length = target.shape
        reference = reference.reshape(-1, reference.shape[-1])
        if reference_rate != target_rate:
            try:
                import torchaudio

                reference = torchaudio.functional.resample(
                    reference, reference_rate, target_rate
                )
            except Exception:
                length = max(
                    1, int(round(reference.shape[-1] * target_rate / reference_rate))
                )
                reference = torch.nn.functional.interpolate(
                    reference.unsqueeze(1),
                    size=length,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)

        target_flat = target.reshape(-1, target_length)
        target_analysis = target_flat.mean(dim=0, keepdim=True)
        reference_analysis = reference.mean(dim=0, keepdim=True).to(target.device)
        n_fft, hop = 2048, 512
        window = torch.hann_window(n_fft, device=target.device, dtype=target.dtype)
        target_spec = torch.stft(
            target_flat,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        analysis_spec = torch.stft(
            target_analysis,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        reference_spec = torch.stft(
            reference_analysis,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        eps = 1e-7
        target_db = 20.0 * torch.log10(analysis_spec.abs().clamp_min(eps)).squeeze(0)
        reference_db = 20.0 * torch.log10(reference_spec.abs().clamp_min(eps)).squeeze(
            0
        )
        reference_energy = 20.0 * torch.log10(
            reference_spec.abs().pow(2).mean(dim=1).sqrt().clamp_min(eps)
        ).squeeze(0)
        gate = reference_energy.max() + float(voice_gate_db)
        voiced = reference_energy >= gate
        if int(voiced.sum()) == 0:
            voiced = torch.ones_like(reference_energy, dtype=torch.bool)
        reference_profile = reference_db[:, voiced].mean(dim=1)

        bins = max(3, int(smoothing_bins) | 1)
        bins = min(
            bins,
            target_db.shape[0] if target_db.shape[0] % 2 else target_db.shape[0] - 1,
        )
        smoothed_target = target_db
        if bins >= 3:
            smoothed_target = (
                torch.nn.functional.avg_pool1d(
                    target_db.transpose(0, 1).unsqueeze(1), bins, 1, bins // 2
                )
                .squeeze(1)
                .transpose(0, 1)
            )
        correction = reference_profile.unsqueeze(1) - smoothed_target
        frequencies = torch.linspace(
            0.0, target_rate / 2.0, correction.shape[0], device=target.device
        )
        mask = (frequencies >= float(low_frequency_hz)) & (
            frequencies <= min(float(high_frequency_hz), target_rate / 2.0)
        )
        if bool(mask.any()):
            correction = correction - correction[mask].mean(dim=0, keepdim=True)
        alpha = float(temporal_smoothing)
        if alpha > 0.0 and correction.shape[1] > 1:
            filtered = correction.clone()
            for frame in range(1, correction.shape[1]):
                filtered[:, frame] = (
                    alpha * filtered[:, frame - 1]
                    + (1.0 - alpha) * correction[:, frame]
                )
            correction = filtered
        correction = correction.mul(float(match_strength)).clamp(
            -float(max_correction_db), float(max_correction_db)
        )
        correction = torch.where(
            mask.unsqueeze(1), correction, torch.zeros_like(correction)
        )
        gain = torch.pow(torch.tensor(10.0, device=target.device), correction / 20.0)
        matched = torch.istft(
            target_spec * gain.unsqueeze(0),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            length=target_length,
        )
        matched = self._apply_peak_ceiling(matched, output_ceiling_db)
        return (
            {
                "waveform": matched.reshape(batch, channels, target_length),
                "sample_rate": target_rate,
            },
        )


class ReaperDynamicVoiceLeveler:
    DESCRIPTION = "Offline dialogue leveler. Smoothly stabilizes quiet and loud speech while leaving gated pauses unchanged, then applies a transparent peak safety correction."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_level_db": (
                    "FLOAT",
                    {"default": -20.0, "min": -36.0, "max": -10.0, "step": 0.5},
                ),
                "voice_gate_db": (
                    "FLOAT",
                    {"default": -45.0, "min": -80.0, "max": -12.0, "step": 1.0},
                ),
                "max_gain_db": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 18.0, "step": 0.5},
                ),
                "max_cut_db": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 18.0, "step": 0.5},
                ),
                "window_ms": (
                    "FLOAT",
                    {"default": 300.0, "min": 80.0, "max": 1000.0, "step": 10.0},
                ),
                "attack_ms": (
                    "FLOAT",
                    {"default": 120.0, "min": 20.0, "max": 1000.0, "step": 10.0},
                ),
                "release_ms": (
                    "FLOAT",
                    {"default": 650.0, "min": 100.0, "max": 2000.0, "step": 25.0},
                ),
                "ceiling_db": (
                    "FLOAT",
                    {"default": -1.0, "min": -6.0, "max": -0.1, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "level"
    CATEGORY = "REAPER AI/Enhancement"

    def level(
        self,
        audio,
        target_level_db=-20.0,
        voice_gate_db=-45.0,
        max_gain_db=6.0,
        max_cut_db=6.0,
        window_ms=300.0,
        attack_ms=120.0,
        release_ms=650.0,
        ceiling_db=-1.0,
    ):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER Dynamic Voice Leveler requires an AUDIO input")
        waveform = audio["waveform"].float()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        batch, channels, length = waveform.shape
        if length < 2:
            return ({"waveform": waveform, "sample_rate": sample_rate},)
        analysis = waveform.mean(dim=1)
        frame_size = max(64, int(round(sample_rate * float(window_ms) / 1000.0)))
        hop = max(32, int(round(sample_rate * 0.05)))
        pad = max(0, frame_size - length)
        if pad:
            analysis_padded = torch.nn.functional.pad(analysis, (0, pad))
        else:
            analysis_padded = analysis
        frames = analysis_padded.unfold(-1, frame_size, hop)
        rms = frames.pow(2).mean(dim=-1).sqrt().clamp_min(1e-7)
        level_db = 20.0 * torch.log10(rms)
        active = level_db >= float(voice_gate_db)
        desired = torch.where(
            active, float(target_level_db) - level_db, torch.zeros_like(level_db)
        )
        desired = desired.clamp(-float(max_cut_db), float(max_gain_db))
        smoothed = torch.zeros_like(desired)
        attack_tau = max(0.001, float(attack_ms) / 1000.0)
        release_tau = max(0.001, float(release_ms) / 1000.0)
        hop_seconds = float(hop) / float(sample_rate)
        for index in range(desired.shape[-1]):
            if index == 0:
                smoothed[:, index] = desired[:, index]
            else:
                rising = desired[:, index] > smoothed[:, index - 1]
                alpha_attack = torch.exp(
                    torch.tensor(
                        -hop_seconds / attack_tau,
                        device=waveform.device,
                        dtype=waveform.dtype,
                    )
                )
                alpha_release = torch.exp(
                    torch.tensor(
                        -hop_seconds / release_tau,
                        device=waveform.device,
                        dtype=waveform.dtype,
                    )
                )
                alpha = torch.where(rising, alpha_attack, alpha_release)
                smoothed[:, index] = (
                    alpha * smoothed[:, index - 1] + (1.0 - alpha) * desired[:, index]
                )
        gain_db = torch.nn.functional.interpolate(
            smoothed.unsqueeze(1), size=length, mode="linear", align_corners=True
        ).squeeze(1)
        gain = torch.pow(
            torch.tensor(10.0, device=waveform.device, dtype=waveform.dtype),
            gain_db / 20.0,
        ).unsqueeze(1)
        leveled = waveform * gain
        ceiling = torch.pow(
            torch.tensor(10.0, device=waveform.device, dtype=waveform.dtype),
            torch.tensor(
                float(ceiling_db), device=waveform.device, dtype=waveform.dtype
            )
            / 20.0,
        )
        peak = leveled.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
        safety_gain = torch.minimum(torch.ones_like(peak), ceiling / peak)
        leveled = leveled * safety_gain
        return (
            {
                "waveform": leveled.reshape(batch, channels, length),
                "sample_rate": sample_rate,
            },
        )


class ReaperClearVoiceEnhance:
    DESCRIPTION = "Full-band speech enhancement using ClearVoice MossFormer2_SE_48K. Requires the optional clearvoice package; models are downloaded on first use."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (
                    ["MossFormer2_SE_48K", "FRCRN_SE_16K", "MossFormerGAN_SE_16K"],
                    {"default": "MossFormer2_SE_48K"},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "enhance"
    CATEGORY = "REAPER AI/Enhancement"

    def enhance(self, audio, model="MossFormer2_SE_48K", strength=1.0, device="auto"):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER ClearVoice Enhance requires an AUDIO input")
        try:
            ensure_clearvoice_librosa_compat()
            from clearvoice import ClearVoice
        except ImportError as error:
            raise RuntimeError(
                "ClearVoice is not installed. Install it in the ComfyUI environment with: pip install clearvoice"
            ) from error
        import tempfile
        import numpy as np

        waveform = audio["waveform"].detach().float().cpu()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError("Audio waveform must be [B,C,T]")
        if waveform.shape[0] != 1:
            raise ValueError("ClearVoice currently supports one audio batch at a time")
        # ClearVoice's public interface is file-based. Preserve the source rate
        # for 48 kHz models and let the model handle its native rate otherwise.
        workdir = Path(tempfile.mkdtemp(prefix="reacomfy_clearvoice_"))
        source = workdir / "input.wav"
        result = workdir / "output.wav"
        try:
            samples = waveform[0].mean(dim=0).numpy()
            sf.write(str(source), samples, sample_rate)
            kwargs = {"task": "speech_enhancement", "model_names": [str(model)]}
            if device == "cpu":
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            enhancer = ClearVoice(**kwargs)
            output = enhancer(input_path=str(source), online_write=False)
            if hasattr(enhancer, "write"):
                enhancer.write(output, output_path=str(result))
            elif isinstance(output, (str, os.PathLike)) and Path(output).is_file():
                result = Path(output)
            elif isinstance(output, dict) and "audio" in output:
                output = output["audio"]
            if not result.is_file():
                arr = (
                    output.detach().cpu().numpy()
                    if torch.is_tensor(output)
                    else np.asarray(output)
                )
                arr = np.squeeze(arr)
                sf.write(str(result), arr.astype(np.float32), sample_rate)
            enhanced, out_rate = sf.read(str(result), always_2d=True, dtype="float32")
            out = torch.from_numpy(enhanced.T).unsqueeze(0)
            if float(strength) < 1.0:
                dry = waveform
                if out.shape[-1] != dry.shape[-1]:
                    out = torch.nn.functional.interpolate(
                        out, size=dry.shape[-1], mode="linear", align_corners=False
                    )
                reference_np = dry[0].mean(dim=0).numpy()
                enhanced_np = out[0].mean(dim=0).numpy()
                step = 8
                ref_ds, cand_ds = reference_np[::step], enhanced_np[::step]
                max_lag = min(
                    int(0.08 * sample_rate / step), max(1, dry.shape[-1] // (4 * step))
                )
                best_lag, best_score = 0, -1e30
                for lag in range(-max_lag, max_lag + 1):
                    if lag > 0:
                        a, b = ref_ds[lag:], cand_ds[:-lag]
                    elif lag < 0:
                        a, b = ref_ds[:lag], cand_ds[-lag:]
                    else:
                        a, b = ref_ds, cand_ds
                    if a.size < 32 or b.size < 32:
                        continue
                    score = float(
                        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                    )
                    if score > best_score:
                        best_lag, best_score = lag, score
                if best_lag > 0:
                    shift = best_lag * step
                    aligned = np.empty_like(enhanced_np)
                    aligned[:-shift] = enhanced_np[shift:]
                    aligned[-shift:] = enhanced_np[-1]
                    enhanced_np = aligned
                elif best_lag < 0:
                    shift = -best_lag * step
                    aligned = np.empty_like(enhanced_np)
                    aligned[:shift] = enhanced_np[0]
                    aligned[shift:] = enhanced_np[:-shift]
                    enhanced_np = aligned
                # Keep the actual dry waveform in the partial blend. The
                # magnitude-only spectral mix can turn model residuals into
                # audible digital noise.
                mixed = band_limited_model_mix(
                    reference_np, enhanced_np, strength, int(out_rate), 0
                )
                out = torch.from_numpy(mixed.astype("float32")).view(1, 1, -1)
            return ({"waveform": out, "sample_rate": int(out_rate)},)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class ReaperClearVoiceSuperResolution:
    DESCRIPTION = "Restores speech bandwidth with ClearVoice MossFormer2_SR_48K. Best used after denoising; outputs 48 kHz audio."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "preserve_low_hz": (
                    "INT",
                    {"default": 6000, "min": 0, "max": 12000, "step": 250},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "restore"
    CATEGORY = "REAPER AI/Enhancement"

    def restore(self, audio, strength=1.0, preserve_low_hz=6000, artifact_guard=0.5):
        if audio is None or "waveform" not in audio:
            raise ValueError(
                "REAPER ClearVoice Super Resolution requires an AUDIO input"
            )
        try:
            ensure_clearvoice_librosa_compat()
            from clearvoice import ClearVoice
        except ImportError as error:
            raise RuntimeError(
                "ClearVoice is not installed. Install it in the ComfyUI environment with: pip install clearvoice"
            ) from error
        waveform = audio["waveform"].detach().float().cpu()
        source_rate = int(audio["sample_rate"])
        import numpy as np

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or waveform.shape[0] != 1:
            raise ValueError(
                "ClearVoice Super Resolution currently supports one audio batch at a time"
            )
        dry = waveform.mean(dim=1)
        if source_rate != 48000:
            import torchaudio

            dry_48 = torchaudio.functional.resample(dry, source_rate, 48000)
        else:
            dry_48 = dry
        enhancer = ClearVoice(
            task="speech_super_resolution", model_names=["MossFormer2_SR_48K"]
        )
        # ClearVoice 0.1.0 has a batch=1 squeeze bug in its SR decoder:
        # it returns a 1-D array and then indexes it as [batch, samples].
        # A duplicated two-item batch keeps its internal tensor 2-D.
        model_input = dry_48.numpy().astype("float32")
        if model_input.ndim == 1:
            model_input = model_input[None, :]
        if model_input.shape[0] == 1:
            model_input = np.repeat(model_input, 2, axis=0)
        output = enhancer(model_input, False)
        output = output[0] if getattr(output, "ndim", 0) > 1 else output
        enhanced = torch.from_numpy(np.asarray(output)).float().view(1, 1, -1)
        if float(strength) < 1.0:
            target_len = int(dry_48.shape[-1])
            enhanced_np = enhanced[0, 0].numpy()
            if enhanced_np.size != target_len:
                old_axis = np.linspace(0.0, 1.0, max(1, enhanced_np.size))
                new_axis = np.linspace(0.0, 1.0, target_len)
                enhanced_np = np.interp(new_axis, old_axis, enhanced_np).astype(
                    "float32"
                )
            reference_np = dry_48[0].numpy()
            step = 8
            ref_ds, cand_ds = reference_np[::step], enhanced_np[::step]
            max_lag = min(int(0.08 * 48000 / step), max(1, target_len // (4 * step)))
            best_lag, best_score = 0, -1e30
            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    a, b = ref_ds[lag:], cand_ds[:-lag]
                elif lag < 0:
                    a, b = ref_ds[:lag], cand_ds[-lag:]
                else:
                    a, b = ref_ds, cand_ds
                if a.size < 32 or b.size < 32:
                    continue
                score = float(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                )
                if score > best_score:
                    best_lag, best_score = lag, score
            if best_lag > 0:
                aligned = np.empty_like(enhanced_np)
                aligned[: -best_lag * step] = enhanced_np[best_lag * step :]
                aligned[-best_lag * step :] = enhanced_np[-1]
                enhanced_np = aligned
            elif best_lag < 0:
                advance = -best_lag * step
                aligned = np.empty_like(enhanced_np)
                aligned[:advance] = enhanced_np[0]
                aligned[advance:] = enhanced_np[:-advance]
                enhanced_np = aligned
            enhanced = torch.from_numpy(enhanced_np).view(1, 1, -1)
            # Keep the actual dry waveform in the partial blend. This avoids
            # the synthetic noise introduced by magnitude-only mixing.
            reference_np = dry_48[0].numpy()
            mixed = band_limited_model_mix(
                reference_np, enhanced_np, strength, 48000, int(preserve_low_hz)
            )
            enhanced = torch.from_numpy(mixed.astype("float32")).view(1, 1, -1)
        return ({"waveform": enhanced, "sample_rate": 48000},)


class ReaperAudioInpaint:
    DESCRIPTION = "CPU audio repair for hard clicks and short mouth/saliva noises. Uses separate impulse and short spectral-burst detectors, protects sustained speech fricatives, and reconstructs only isolated damaged regions from neighboring samples."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["auto", "de_click", "inpaint"], {"default": "auto"}),
                "sensitivity": (
                    "FLOAT",
                    {"default": 8.0, "min": 2.0, "max": 20.0, "step": 0.5},
                ),
                "mouth_sensitivity": (
                    "FLOAT",
                    {"default": 8.0, "min": 2.0, "max": 20.0, "step": 0.5},
                ),
                "speech_protect": (
                    "FLOAT",
                    {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_repair_ms": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.5, "max": 15.0, "step": 0.5},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "repair_info")
    FUNCTION = "repair"
    CATEGORY = "REAPER AI/Restoration"

    def repair(
        self,
        audio,
        mode="auto",
        sensitivity=8.0,
        mouth_sensitivity=8.0,
        speech_protect=0.80,
        max_repair_ms=8.0,
        strength=0.75,
    ):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER Audio Inpaint requires an AUDIO input")
        try:
            from scipy.ndimage import (
                binary_closing,
                gaussian_filter1d,
                maximum_filter1d,
                median_filter,
                uniform_filter1d,
            )
        except ImportError as error:
            raise RuntimeError(
                "REAPER Audio Inpaint requires scipy. Install it with: pip install scipy"
            ) from error
        import numpy as np

        waveform = audio["waveform"].detach().float().cpu()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or waveform.shape[0] != 1:
            raise ValueError(
                "REAPER Audio Inpaint currently supports one audio batch at a time"
            )
        data = waveform[0].numpy().copy()
        channels, length = data.shape
        if length < 16:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                json.dumps({"repaired_regions": 0, "mode": mode}),
            )
        mono = data.mean(axis=0)
        # Detector 1: isolated, sub-3 ms waveform discontinuities.  The small
        # median residual is deliberately kept independent of the mouth path.
        med = median_filter(mono, size=9, mode="nearest")
        residual = mono - med
        robust = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
        peak = max(float(np.max(np.abs(mono))), 1e-6)
        mode_factor = (
            1.50 if mode == "de_click" else (0.90 if mode == "inpaint" else 1.0)
        )
        threshold = max(
            robust * float(sensitivity) * mode_factor, peak * 0.15 * mode_factor
        )
        local_rms = np.sqrt(
            np.maximum(
                gaussian_filter1d(mono * mono, max(1.0, 0.002 * sample_rate)), 1e-12
            )
        )
        impulsive_ratio = np.abs(residual) / np.maximum(local_rms, peak * 0.002)
        ratio_limit = 3.0 if mode == "de_click" else (1.6 if mode == "inpaint" else 2.5)
        hard_candidate = (np.abs(residual) > threshold) & (
            impulsive_ratio > ratio_limit
        )

        # Detector 2: mouth/saliva clicks are often 2-15 ms noisy bursts rather
        # than single-sample spikes.  Compare short high-band energy and its
        # flux with a 20 ms local baseline.  This is a light-weight spectral
        # novelty proxy and avoids an STFT/ISTFT repair path.
        hp_sigma = max(1.0, 0.00012 * sample_rate)
        high_band = mono - gaussian_filter1d(mono, hp_sigma, mode="nearest")
        short_window = max(3, int(0.0015 * sample_rate))
        long_window = max(short_window + 2, int(0.020 * sample_rate))
        short_hf = np.sqrt(
            np.maximum(
                uniform_filter1d(high_band * high_band, short_window, mode="nearest"),
                1e-12,
            )
        )
        long_hf = np.sqrt(
            np.maximum(
                uniform_filter1d(high_band * high_band, long_window, mode="nearest"),
                1e-12,
            )
        )
        hf_flux = np.abs(np.diff(high_band, prepend=high_band[0]))
        short_flux = uniform_filter1d(hf_flux, short_window, mode="nearest")
        long_flux = uniform_filter1d(hf_flux, long_window, mode="nearest")
        energy_ratio = short_hf / np.maximum(long_hf, peak * 0.001)
        flux_ratio = short_flux / np.maximum(long_flux, peak * 0.0002)
        mouth_score = np.sqrt(
            np.maximum(energy_ratio, 0.0) * np.maximum(flux_ratio, 0.0)
        )
        score_center = float(np.median(mouth_score))
        score_mad = max(
            float(np.median(np.abs(mouth_score - score_center)) * 1.4826), 0.03
        )
        mouth_limit = score_center + score_mad * (1.5 + 0.32 * float(mouth_sensitivity))
        mouth_seed = (
            (mouth_score > mouth_limit)
            & (energy_ratio > 1.35)
            & (flux_ratio > 1.20)
            & (short_hf > peak * 0.0025)
        )
        mouth_expand = max(3, int(0.0015 * sample_rate))
        mouth_candidate = (
            maximum_filter1d(
                mouth_seed.astype(np.uint8), size=mouth_expand, mode="nearest"
            )
            > 0
        )
        mouth_candidate = binary_closing(
            mouth_candidate,
            structure=np.ones(max(2, int(0.00075 * sample_rate)), dtype=bool),
        )

        candidate = hard_candidate | mouth_candidate
        groups = []
        start = None
        for index, flag in enumerate(candidate):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                groups.append((start, index))
                start = None
        if start is not None:
            groups.append((start, length))
        max_samples = max(2, int(float(max_repair_ms) * sample_rate / 1000.0))
        hard_max_samples = max(2, int(0.003 * sample_rate))
        speech_context = max(4, int(0.020 * sample_rate))
        # A fricative keeps high-band energy alive on both sides of a candidate;
        # a mouth click is a local outlier with quiet/normal context.  Higher
        # speech_protect lowers the tolerated sustained-HF occupancy.
        hf_floor = max(float(np.median(short_hf)) * 1.8, peak * 0.0015)
        hf_active = short_hf > np.maximum(long_hf * 0.72, hf_floor)
        occupancy_limit = 0.72 - 0.34 * float(speech_protect)
        repaired = 0
        repaired_hard = 0
        repaired_mouth = 0
        protected_regions = 0
        for begin, end in groups:
            is_hard = bool(np.any(hard_candidate[begin:end]))
            is_mouth = bool(np.any(mouth_candidate[begin:end]))
            allowed_samples = (
                max_samples if is_mouth else min(max_samples, hard_max_samples)
            )
            if end - begin > allowed_samples or begin < 2 or end >= length - 2:
                continue
            context_start = max(0, begin - speech_context)
            context_end = min(length, end + speech_context)
            left_hf = (
                float(np.mean(hf_active[context_start:begin]))
                if begin > context_start
                else 0.0
            )
            right_hf = (
                float(np.mean(hf_active[end:context_end])) if context_end > end else 0.0
            )
            # At a consonant onset or release only one side may contain the
            # sustained noise, so either side is sufficient for protection.
            sustained_hf = max(left_hf, right_hf)
            # Never punch holes in sustained s/z/sh/zh energy.  Even a tall
            # sample inside a fricative is more safely left alone than treated
            # as an isolated click; speech_protect can be lowered for a more
            # aggressive pass when the source is known not to contain speech.
            if float(speech_protect) > 0.0 and sustained_hf > occupancy_limit:
                protected_regions += 1
                continue
            left = max(2, begin - max(2, int(0.001 * sample_rate)))
            right = min(length - 3, end + max(2, int(0.001 * sample_rate)))
            # Repair the short context around the impulse as well. Replacing
            # only the flagged samples can leave a discontinuity at the edge.
            span = np.arange(left, right)
            if span.size == 0:
                continue
            for channel in range(channels):
                # Cubic Hermite bridge preserves the local slope at both
                # edges; a plain linear bridge can create a low-frequency
                # thump even when the click itself is removed.
                y0, y1 = float(data[channel, left - 1]), float(data[channel, right])
                m0 = float(data[channel, left] - data[channel, left - 2])
                m1 = float(data[channel, right + 1] - data[channel, right - 1])
                u = (span.astype(np.float64) - float(left - 1)) / float(
                    right - left + 1
                )
                h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
                h10 = u**3 - 2.0 * u**2 + u
                h01 = -2.0 * u**3 + 3.0 * u**2
                h11 = u**3 - u**2
                repaired_values = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
                # Fade the correction to zero at the boundaries so the
                # repair cannot introduce a new discontinuity or LF thump.
                taper = 0.5 - 0.5 * np.cos(2.0 * np.pi * u)
                repaired_values = (
                    data[channel, span]
                    + (repaired_values - data[channel, span]) * taper
                )
                if float(strength) >= 1.0:
                    data[channel, span] = repaired_values
                elif float(strength) > 0.0:
                    data[channel, span] += (
                        repaired_values - data[channel, span]
                    ) * float(strength)
            repaired += 1
            repaired_hard += int(is_hard)
            repaired_mouth += int(is_mouth)
        input_peak = max(float(np.max(np.abs(waveform[0].numpy()))), 1e-8)
        output_peak = float(np.max(np.abs(data)))
        if output_peak > input_peak:
            data *= input_peak / output_peak
        result = torch.from_numpy(data).unsqueeze(0)
        info = json.dumps(
            {
                "mode": mode,
                "sample_rate": sample_rate,
                "repaired_regions": repaired,
                "repaired_hard_clicks": repaired_hard,
                "repaired_mouth_clicks": repaired_mouth,
                "speech_protected_regions": protected_regions,
                "sensitivity": float(sensitivity),
                "mouth_sensitivity": float(mouth_sensitivity),
                "speech_protect": float(speech_protect),
                "max_repair_ms": float(max_repair_ms),
            },
            indent=2,
        )
        return ({"waveform": result, "sample_rate": sample_rate}, info)


class ReaperRoomMatchHybrid:
    DESCRIPTION = "Hybrid room-profile estimator requiring two simultaneous recordings: a dry-ish lavalier and an original reverberant ambient microphone. Aligns both recordings, estimates a regularized transfer response, extracts early reflections, synthesizes a stable late tail, and returns an IR plus room metrics. Use denoise or light dereverb only on the lavalier; keep ambient reverberation intact. Optional stereo_ir preserves a practical L/R ambient balance; mono is the default. With one recording, use a dereverb chain or an existing IR instead. This is an acoustic approximation, not a measured swept-sine IR."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lapel_audio": ("AUDIO",),
                "ambient_audio": ("AUDIO",),
                "stereo_ir": ("BOOLEAN", {"default": False}),
                "ir_length_sec": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.25, "max": 5.0, "step": 0.05},
                ),
                "max_alignment_ms": (
                    "FLOAT",
                    {"default": 250.0, "min": 20.0, "max": 1000.0, "step": 10.0},
                ),
                "early_reflections_ms": (
                    "FLOAT",
                    {"default": 120.0, "min": 20.0, "max": 300.0, "step": 5.0},
                ),
                "tail_floor_db": (
                    "FLOAT",
                    {"default": -48.0, "min": -80.0, "max": -20.0, "step": 1.0},
                ),
                "tail_rt60_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05},
                ),
                "early_gain_db": (
                    "FLOAT",
                    {"default": 0.0, "min": -24.0, "max": 12.0, "step": 0.5},
                ),
                "late_gain_db": (
                    "FLOAT",
                    {"default": 0.0, "min": -24.0, "max": 12.0, "step": 0.5},
                ),
                "ir_high_cut_hz": (
                    "INT",
                    {"default": 12000, "min": 4000, "max": 20000, "step": 500},
                ),
                "apply_ir": ("BOOLEAN", {"default": False}),
                "mix": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "audio",
        "room_ir",
        "profile_json",
        "rt60_s",
        "pre_delay_ms",
        "drr_db",
    )
    FUNCTION = "match"
    CATEGORY = "REAPER AI/Room Acoustics"

    def match(
        self,
        lapel_audio,
        ambient_audio,
        stereo_ir=False,
        ir_length_sec=1.5,
        max_alignment_ms=250.0,
        early_reflections_ms=120.0,
        tail_floor_db=-48.0,
        tail_rt60_scale=1.0,
        early_gain_db=0.0,
        late_gain_db=0.0,
        ir_high_cut_hz=12000,
        apply_ir=False,
        mix=0.35,
    ):
        if (
            not lapel_audio
            or not ambient_audio
            or "waveform" not in lapel_audio
            or "waveform" not in ambient_audio
        ):
            raise ValueError(
                "REAPER Room Match Hybrid requires lapel_audio and ambient_audio"
            )
        import numpy as np

        try:
            from scipy.signal import fftconvolve, butter, sosfilt
            from scipy.ndimage import gaussian_filter1d
        except ImportError as error:
            raise RuntimeError("REAPER Room Match Hybrid requires scipy") from error

        sr = int(lapel_audio["sample_rate"])
        x = (
            lapel_audio["waveform"]
            .detach()
            .float()
            .cpu()
            .mean(dim=1)
            .reshape(-1)
            .numpy()
        )
        ambient_wave = ambient_audio["waveform"].detach().float().cpu()
        ambient_stereo = None
        if bool(stereo_ir) and ambient_wave.ndim >= 3 and ambient_wave.shape[1] >= 2:
            ambient_stereo = ambient_wave[0, :2, :].numpy()
        y = ambient_wave.mean(dim=1).reshape(-1).numpy()
        if int(ambient_audio["sample_rate"]) != sr:
            ay = torch.from_numpy(y).view(1, 1, -1)
            target = max(1, round(y.size * sr / int(ambient_audio["sample_rate"])))
            y = (
                torch.nn.functional.interpolate(
                    ay, size=target, mode="linear", align_corners=False
                )
                .reshape(-1)
                .numpy()
            )
            if ambient_stereo is not None:
                ast = torch.from_numpy(ambient_stereo).unsqueeze(0)
                ambient_stereo = torch.nn.functional.interpolate(
                    ast, size=target, mode="linear", align_corners=False
                )[0].numpy()
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        x -= x.mean()
        y -= y.mean()
        x /= max(np.sqrt(np.mean(x * x)), 1e-8)
        y /= max(np.sqrt(np.mean(y * y)), 1e-8)
        fft_n = 1 << max(12, (2 * n - 1).bit_length())
        corr = np.fft.irfft(
            np.fft.rfft(y, fft_n) * np.conj(np.fft.rfft(x, fft_n)), fft_n
        )
        corr = np.concatenate((corr[-(n - 1) :], corr[:n]))
        lags = np.arange(-(n - 1), n)
        limit = min(n - 1, int(sr * float(max_alignment_ms) / 1000.0))
        valid = np.abs(lags) <= limit
        lag = int(lags[valid][np.argmax(np.abs(corr[valid]))])
        if lag > 0:
            y, x = y[lag:], x[:-lag]
        elif lag < 0:
            x, y = x[-lag:], y[:lag]
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]

        # Regularized transfer estimate averaged over speech frames.
        frame = min(131072, n)
        if frame < 4096:
            frame = n
        hop = max(1, frame // 2)
        window = np.hanning(frame)
        cross = np.zeros(frame // 2 + 1, dtype=np.complex128)
        auto = np.zeros(frame // 2 + 1, dtype=np.float64)
        count = 0
        for start in range(0, max(1, n - frame + 1), hop):
            xs, ys = x[start : start + frame], y[start : start + frame]
            if xs.size < frame:
                break
            energy = np.mean(xs * xs)
            if energy < 1e-5:
                continue
            X, Y = np.fft.rfft(xs * window), np.fft.rfft(ys * window)
            cross += Y * np.conj(X)
            auto += np.abs(X) ** 2
            count += 1
        if count == 0:
            raise ValueError(
                "Room Match could not find usable speech frames in the two recordings"
            )
        H = (cross / count) / (auto / count + 1e-4)
        # Speech recordings make the raw transfer estimate noisy at the top
        # end. Smooth magnitude, clamp implausible boosts, and taper the band
        # above the selected cutoff before converting it back to an IR.
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-8))
        kernel = np.ones(129, dtype=np.float64) / 129.0
        mag_db = np.convolve(mag_db, kernel, mode="same")
        mag_db = np.clip(mag_db, -36.0, 6.0)
        freq = np.fft.rfftfreq(frame, 1.0 / sr)
        cutoff = min(float(ir_high_cut_hz), 0.45 * sr)
        # Natural room absorption is a slope, not a brick-wall cut. Keep a
        # small high-frequency remainder and let a gentle analogue-like
        # low-pass shape the roll-off.
        taper = 1.0 / (1.0 + (freq / max(cutoff, 1.0)) ** 2)
        taper = np.maximum(taper, 0.12)
        H = (10.0 ** (mag_db / 20.0)) * np.exp(1j * np.angle(H)) * taper
        response = np.fft.irfft(H, frame)
        response = np.nan_to_num(response)
        peak = int(np.argmax(np.abs(response[: frame // 2])))
        response = np.roll(response, -peak)
        ir_len = min(frame, max(64, int(float(ir_length_sec) * sr)))
        ir = response[:ir_len].copy()
        # A speech-derived transfer estimate contains circular FFT residue and
        # diffuse estimation noise. Keep measured early reflections, then force
        # a monotonic late decay and gate the residual noise floor.
        early_len = min(ir_len, max(1, int(float(early_reflections_ms) * sr / 1000.0)))
        peak_level = max(float(np.max(np.abs(ir[: max(1, early_len)]))), 1e-8)
        if ir_len > early_len:
            late = ir[early_len:]
            late_t = np.arange(late.size, dtype=np.float64) / sr
            # Fit the raw Schroeder decay before shaping, then use that
            # measured RT60 to form the synthetic late tail. The fallback is
            # only used when speech does not provide a stable decay slope.
            raw_energy = np.cumsum((late[::-1] ** 2))[::-1]
            raw_db = 10.0 * np.log10(
                np.maximum(raw_energy / max(raw_energy[0], 1e-12), 1e-12)
            )
            raw_fit = (raw_db < -5.0) & (raw_db > -35.0)
            raw_rt60 = 0.0
            if np.count_nonzero(raw_fit) > 8:
                raw_slope = np.polyfit(late_t[raw_fit], raw_db[raw_fit], 1)[0]
                if raw_slope < 0:
                    raw_rt60 = float(max(0.15, min(8.0, -60.0 / raw_slope)))
            tau = (
                (raw_rt60 * float(tail_rt60_scale)) / np.log(1000.0)
                if raw_rt60 > 0
                else 0.18 * float(tail_rt60_scale)
            )
            late *= np.exp(-late_t / max(tau, 0.03))
            ir[early_len:] = late
        sos = butter(
            2, min(cutoff * 1.5, 0.48 * sr), btype="lowpass", fs=sr, output="sos"
        )
        ir = sosfilt(sos, ir)
        # Use a smoothed RMS gate instead of zeroing individual samples. A
        # sample-wise gate creates the crunchy/dither-like fade heard in IRs.
        floor = peak_level * (10.0 ** (float(tail_floor_db) / 20.0))
        rms = np.sqrt(
            np.maximum(gaussian_filter1d(ir * ir, max(1.0, 0.010 * sr)), 1e-12)
        )
        gate_pos = np.clip((rms - floor * 0.35) / max(floor * 1.15, 1e-8), 0.0, 1.0)
        gate = gate_pos * gate_pos * (3.0 - 2.0 * gate_pos)
        ir *= gate
        # Room tails should not regain energy after they have decayed. Apply a
        # monotonic RMS envelope while retaining the fine reflection texture.
        if ir_len > early_len:
            # Do not anchor the tail to its first sample: a near-zero sample
            # at the early/late boundary would otherwise mute the whole tail.
            tail_rms = np.sqrt(
                np.maximum(
                    gaussian_filter1d(ir[early_len:] ** 2, max(1.0, 0.012 * sr)), 1e-12
                )
            )
            anchor_len = min(tail_rms.size, max(1, int(0.020 * sr)))
            anchor = max(float(np.max(tail_rms[:anchor_len])), 1e-8)
            tail_tau = (
                (raw_rt60 * float(tail_rt60_scale)) / np.log(1000.0)
                if raw_rt60 > 0
                else 0.18 * float(tail_rt60_scale)
            )
            envelope = anchor * np.exp(
                -np.arange(tail_rms.size, dtype=np.float64) / max(tail_tau * sr, 1.0)
            )
            ir[early_len:] *= np.clip(envelope / np.maximum(tail_rms, 1e-12), 0.0, 1.0)
        ir[:early_len] *= 10.0 ** (float(early_gain_db) / 20.0)
        if ir_len > early_len:
            ir[early_len:] *= 10.0 ** (float(late_gain_db) / 20.0)
        fade_len = min(ir_len, max(32, int(0.05 * sr)))
        ir[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
        direct_len = max(1, int(0.02 * sr))
        direct_energy = np.sum(ir[:direct_len] ** 2) + 1e-12
        tail_energy = np.sum(ir[direct_len:] ** 2) + 1e-12
        drr_db = float(10 * np.log10(direct_energy / tail_energy))
        edc = np.cumsum((ir[::-1] ** 2))[::-1]
        db = 10 * np.log10(np.maximum(edc / max(edc[0], 1e-12), 1e-12))
        fit = (db < -5) & (db > -35)
        if np.count_nonzero(fit) > 8:
            slope = np.polyfit(np.arange(ir_len)[fit] / sr, db[fit], 1)[0]
            rt60 = float(max(0.05, min(10.0, -60.0 / slope))) if slope < 0 else 0.0
        else:
            rt60 = 0.0
        ir /= max(np.max(np.abs(ir)), 1e-8)
        pre_delay_ms = float(1000.0 * peak / sr)

        ir_channels = (
            np.stack([ir, ir], axis=0) if ambient_stereo is not None else ir[None, :]
        )
        if ambient_stereo is not None:
            ch_rms = np.sqrt(np.mean(ambient_stereo[:, :n] ** 2, axis=1) + 1e-12)
            ch_rms /= max(float(np.mean(ch_rms)), 1e-8)
            ir_channels *= ch_rms[:, None]
            ir_channels /= max(float(np.max(np.abs(ir_channels))), 1e-8)
        source = torch.from_numpy(x.astype("float32")).view(1, 1, -1)
        ir_audio = torch.from_numpy(ir_channels.astype("float32")).view(
            1, ir_channels.shape[0], -1
        )
        output = source
        if apply_ir:
            wet_channels = np.stack(
                [
                    fftconvolve(x, ir_channels[ch], mode="full")[: x.size]
                    for ch in range(ir_channels.shape[0])
                ],
                axis=0,
            )
            wet_channels /= max(float(np.max(np.abs(wet_channels))), 1e-8)
            if ir_channels.shape[0] == 2:
                dry_channels = np.repeat(x[None, :], 2, axis=0)
                output = torch.from_numpy(
                    (
                        (1.0 - float(mix)) * dry_channels + float(mix) * wet_channels
                    ).astype("float32")
                ).view(1, 2, -1)
            else:
                output = torch.from_numpy(
                    ((1.0 - float(mix)) * x + float(mix) * wet_channels[0]).astype(
                        "float32"
                    )
                ).view(1, 1, -1)
        profile = json.dumps(
            {
                "sample_rate": sr,
                "alignment_lag_ms": 1000.0 * lag / sr,
                "pre_delay_ms": pre_delay_ms,
                "rt60_s": rt60,
                "drr_db": drr_db,
                "ir_length_sec": float(ir_length_sec),
                "method": "hybrid_transfer_plus_decay",
            },
            indent=2,
        )
        return (
            {"waveform": output, "sample_rate": sr},
            {"waveform": ir_audio, "sample_rate": sr},
            profile,
            rt60,
            pre_delay_ms,
            drr_db,
        )


class ReaperLavaSR:
    DESCRIPTION = "LavaSR v2 speech bandwidth extension. The model internally analyzes audio at 16 kHz, discards the original band above about 8 kHz, synthesizes a new high band, and returns 48 kHz audio. strength blends the result with dry audio; cutoff_hz controls the model's low/high reconstruction crossover. This is synthesis, not recovery of the original high-frequency samples."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "denoise": ("BOOLEAN", {"default": False}),
                "batch": ("BOOLEAN", {"default": False}),
                "cutoff_hz": (
                    "INT",
                    {"default": 0, "min": 0, "max": 20000, "step": 100},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "enhance"
    CATEGORY = "REAPER AI/Enhancement"

    _models = {}

    def enhance(self, audio, strength=1.0, denoise=False, batch=False, cutoff_hz=0):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER LavaSR requires an AUDIO input")
        waveform = audio["waveform"].detach().float().cpu()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or waveform.shape[0] != 1:
            raise ValueError(
                "REAPER LavaSR currently supports one audio batch at a time"
            )
        try:
            from LavaSR.model import LavaEnhance2
        except ImportError as error:
            raise RuntimeError(
                "LavaSR is not installed. In the active ComfyUI environment run: "
                "pip install git+https://github.com/ysharma3501/LavaSR.git "
                "and pip install --no-deps git+https://github.com/langtech-bsc/vocos.git@matcha"
            ) from error
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self._models.get(device)
        if model is None:
            model = LavaEnhance2("YatharthS/LavaSR", device=device)
            self._models[device] = model
        import tempfile

        workdir = Path(tempfile.mkdtemp(prefix="reacomfy_lavasr_"))
        source = workdir / "input.wav"
        try:
            sf.write(str(source), waveform[0].mean(dim=0).numpy(), sample_rate)
            cutoff = int(cutoff_hz) if int(cutoff_hz) > 0 else None
            input_audio, _ = model.load_audio(
                str(source), input_sr=16000, cutoff=cutoff
            )
            restored = model.enhance(
                input_audio, enhance=True, denoise=bool(denoise), batch=bool(batch)
            )
            restored = restored.detach().float().cpu().reshape(1, 1, -1)
            dry = waveform.mean(dim=1, keepdim=True)
            if sample_rate != 48000:
                dry = torch.nn.functional.interpolate(
                    dry,
                    size=max(1, round(dry.shape[-1] * 48000 / sample_rate)),
                    mode="linear",
                    align_corners=False,
                )
            length = min(dry.shape[-1], restored.shape[-1])
            dry, restored = dry[..., :length], restored[..., :length]
            output = dry + (restored - dry) * float(strength)
            return ({"waveform": output, "sample_rate": 48000},)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


_VOICE_ISOLATOR_CACHE = {}
_VOICE_ISOLATOR_FILES = {
    "offline": "denoiser.safetensors",
    "streaming": "denoiser_realtime.safetensors",
}


def _voice_isolator_local_checkpoint(mode):
    """Return a checkpoint placed manually in ComfyUI/models/voice_isolator, if any."""
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / "voice_isolator"
    except Exception:
        return None
    candidate = base / _VOICE_ISOLATOR_FILES[mode]
    return str(candidate) if candidate.is_file() else None


def _load_voice_isolator(mode, lookahead_frames, device):
    key = (mode, int(lookahead_frames), device)
    cached = _VOICE_ISOLATOR_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from inworld import voice_isolation
    except ImportError as error:
        raise RuntimeError(
            "Inworld Voice Isolator is not installed. Install it in the ComfyUI "
            "environment with: python -m pip install inworld-voice-isolation"
        ) from error
    model = _voice_isolator_local_checkpoint(mode) or "inworld/voice-isolator"
    isolator = voice_isolation.VoiceIsolator(
        model,
        device=None if device == "auto" else device,
        mode=mode,
        lookahead_frames=int(lookahead_frames) if mode == "streaming" else 0,
    )
    _VOICE_ISOLATOR_CACHE.clear()  # keep at most one model resident
    _VOICE_ISOLATOR_CACHE[key] = isolator
    return isolator


class ReaperVoiceIsolator:
    DESCRIPTION = (
        "Separates the foreground voice from noise, music and background using "
        "Inworld Voice Isolator (Apache-2.0, 48 kHz). Inputs: AUDIO, mode, strength. "
        "Outputs: voice and background (voice + background = input). Requires the "
        "optional inworld-voice-isolation package; weights download on first use."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (
                    ["offline", "streaming"],
                    {
                        "default": "offline",
                        "tooltip": "offline: best quality, sees 8 s chunks. streaming: causal realtime model.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "1.0 = fully isolated voice; lower values blend the dry signal back in.",
                    },
                ),
                "channels": (
                    ["per channel", "mono"],
                    {
                        "default": "per channel",
                        "tooltip": "per channel keeps stereo/multichannel layout; mono mixes down first (faster).",
                    },
                ),
            },
            "optional": {
                "lookahead_ms": (
                    ["20", "0", "40"],
                    {
                        "default": "20",
                        "tooltip": "Streaming mode only: trained look-ahead of the realtime model.",
                    },
                ),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("voice", "background")
    FUNCTION = "isolate"
    CATEGORY = "REAPER AI/Enhancement"

    def isolate(
        self,
        audio,
        mode="offline",
        strength=1.0,
        channels="per channel",
        lookahead_ms="20",
        device="auto",
        keep_model_loaded=True,
    ):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER Voice Isolator requires an AUDIO input")
        waveform = audio["waveform"].detach().float().cpu()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError("Audio waveform must be [B,C,T]")
        lookahead_frames = int(lookahead_ms) // 10
        isolator = _load_voice_isolator(mode, lookahead_frames, device)
        try:
            try:
                import comfy.model_management as model_management

                check_interrupt = (
                    model_management.throw_exception_if_processing_interrupted
                )
            except Exception:
                check_interrupt = lambda: None
            if channels == "mono":
                dry = waveform.mean(dim=1, keepdim=True)
            else:
                dry = waveform
            wet = torch.empty_like(dry)
            for b in range(dry.shape[0]):
                for c in range(dry.shape[1]):
                    check_interrupt()
                    out = isolator.isolate(dry[b, c].contiguous(), sample_rate)
                    out = torch.as_tensor(out).float().reshape(-1)
                    n = min(out.shape[-1], dry.shape[-1])
                    wet[b, c, :n] = out[:n]
                    if n < dry.shape[-1]:
                        wet[b, c, n:] = 0.0
            amount = float(strength)
            voice = dry + amount * (wet - dry) if amount < 1.0 else wet
            background = dry - voice
            return (
                {"waveform": voice.contiguous(), "sample_rate": sample_rate},
                {"waveform": background.contiguous(), "sample_rate": sample_rate},
            )
        finally:
            if not keep_model_loaded:
                _VOICE_ISOLATOR_CACHE.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


class ReaperVideoInput:
    DESCRIPTION = "REAPER video input. Outputs: video_path STRING (source path), video VIDEO (native file input), woosh_video WOOSH_VIDEO (Woosh 2026 V2A/Foley), images IMAGE frames, and fps FLOAT. limit_duration is optional and disabled by default; when enabled, max_duration_s limits only frame-based outputs."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "refresh_token": ("STRING", {"default": ""}),
                "max_duration_s": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0}),
                "limit_duration": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "VIDEO", "WOOSH_VIDEO", "IMAGE", "FLOAT")
    RETURN_NAMES = ("video_path", "video", "woosh_video", "images", "fps")
    FUNCTION = "load"
    CATEGORY = "REAPER AI/Video"

    def load(self, refresh_token="", max_duration_s=8.0, limit_duration=False):
        data = request_data()
        path = Path(str(data.get("video_path", data.get("path", "")))).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"REAPER video file not found: {path}")
        try:
            from comfy_api.input_impl import VideoFromFile

            video = VideoFromFile(str(path))
        except ImportError as error:
            raise RuntimeError(
                "This ComfyUI version does not provide the native VIDEO type"
            ) from error
        try:
            from woosh.utils.videoio import extract_video_frames

            end_time = float(max_duration_s) if limit_duration else None
            frames, rate, _pts = extract_video_frames(
                str(path), start_time=0, end_time=end_time
            )
            woosh_video = {"frames": frames, "rate": rate}
            images = frames.float().div(255.0)
        except Exception:
            woosh_video = None
            images, rate = None, 0.0
        return (str(path), video, woosh_video, images, float(rate))


class ReaperOutput:
    DESCRIPTION = "Writes ComfyUI AUDIO to the REAPER project cache. Resampling is optional and disabled by default; when enabled, output_sample_rate selects 44100, 48000, or 96000 Hz."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio": ("AUDIO",)},
            "optional": {
                "label": ("STRING", {"default": "ComfyUI result"}),
                "resample_output": ("BOOLEAN", {"default": False}),
                "output_sample_rate": (
                    ["44100", "48000", "96000"],
                    {"default": "48000"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "REAPER AI/Outputs"

    def save(
        self,
        audio,
        label="ComfyUI result",
        resample_output=False,
        output_sample_rate="48000",
    ):
        cache = cache_dir()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = (
            cache / f"comfy_to_reaper_{stamp}_{int(time.time() * 1000) % 1000:03d}.wav"
        )
        source_rate = int(audio["sample_rate"])
        target_rate = int(output_sample_rate) if resample_output else source_rate
        waveform = audio["waveform"].detach().float()
        if waveform.ndim == 3:
            waveform = waveform[0]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if target_rate != source_rate:
            try:
                import torchaudio

                waveform = torchaudio.functional.resample(
                    waveform, source_rate, target_rate
                )
            except Exception:
                length = max(
                    1, int(round(waveform.shape[-1] * target_rate / source_rate))
                )
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0),
                    size=length,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)
        waveform = waveform.cpu()
        samples = waveform.numpy().T
        sf.write(str(output), samples, target_rate, subtype="PCM_24")
        manifest = cache / "comfy_to_reaper.json"
        outputs = []
        request_job_id = None
        request = cache / "reaper_to_comfy.json"
        if request.is_file():
            try:
                request_job_id = json.loads(request.read_text(encoding="utf-8")).get(
                    "job_id"
                )
            except Exception:
                pass
        if manifest.is_file():
            try:
                previous = json.loads(manifest.read_text(encoding="utf-8"))
                if request_job_id and previous.get("job_id") == request_job_id:
                    outputs = list(previous.get("outputs", []))
            except Exception:
                outputs = []
        outputs.append({"path": str(output), "label": label})
        position = None
        length = None
        if request.is_file():
            try:
                request_data = json.loads(request.read_text(encoding="utf-8"))
                position = request_data.get("position")
                length = request_data.get("length")
            except Exception:
                pass
        payload = {
            "status": "complete",
            "job_id": request_job_id,
            "outputs": outputs,
            "path": str(output),
            "label": label,
        }
        if position is not None:
            payload["position"] = position
        if length is not None:
            payload["length"] = length
        payload = json.dumps(payload, ensure_ascii=False, indent=2)
        manifest.write_text(payload, encoding="utf-8")
        # Keep a user-level fallback so REAPER can import results even when
        # the project was unsaved before the send action created its pointer.
        fallback = Path.home() / "REAPER_AI_CACHE"
        fallback.mkdir(parents=True, exist_ok=True)
        fallback_manifest = fallback / "comfy_to_reaper.json"
        if fallback_manifest.resolve() != manifest.resolve():
            fallback_manifest.write_text(payload, encoding="utf-8")
        return {
            "ui": {"text": [f"REAPER output ready: {output}"]},
            "result": (str(manifest),),
        }


class ReaperTextInput:
    DESCRIPTION = "Reads text from the current REAPER manifest. Inputs: field (text, lyrics, chords or arrangement), optional prefix_text and refresh_token. Output: STRING text."
    """Read lyrics/text prepared by a REAPER action as a STRING output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "field": (
                    ["text", "lyrics", "chords", "arrangement"],
                    {"default": "text"},
                ),
                "prefix_text": ("STRING", {"multiline": True, "default": ""}),
                "refresh_token": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"
    CATEGORY = "REAPER AI/Inputs"

    @classmethod
    def IS_CHANGED(cls, field="text", prefix_text="", refresh_token=""):
        """Invalidate ComfyUI's execution cache when REAPER publishes a new job."""
        candidates = []
        try:
            candidates.append(cache_dir() / "reaper_to_comfy.json")
        except Exception:
            pass
        fallback = Path.home() / "REAPER_AI_CACHE" / "reaper_to_comfy.json"
        if fallback not in candidates:
            candidates.append(fallback)
        signature = [
            str(refresh_token or ""),
            str(field or "text"),
            str(prefix_text or ""),
        ]
        for path in candidates:
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    signature.extend(
                        [str(data.get("job_id", "")), str(path.stat().st_mtime_ns)]
                    )
                    break
                except Exception:
                    signature.append(str(path.stat().st_mtime_ns))
        return "|".join(signature)

    def load(self, field="text", prefix_text="", refresh_token=""):
        data = request_data()
        field_map = {
            "lyrics": "lyrics_text",
            "chords": "chords_text",
            "arrangement": "arrangement_text",
        }
        selected_key = field_map.get(field, "text")
        text = str(data.get(selected_key, "") or "")
        if not text.strip():
            source = str(data.get("source", ""))
            if field == "chords" and source == "chords":
                text = str(data.get("chords", data.get("text", "")) or "")
            elif field == "lyrics" and source == "regions":
                text = str(data.get("text", "") or "")
            elif field == "arrangement" and source == "arrangement":
                text = str(data.get("text", "") or "")
        text_path = (
            data.get("lyrics_text_path") if field == "lyrics" else data.get("text_path")
        )
        if text_path:
            path = Path(str(text_path)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"REAPER text file not found: {path}")
            text = path.read_text(encoding="utf-8-sig")
        prefix_text = str(prefix_text or "").strip()
        if prefix_text and text.strip():
            text = prefix_text + "\n\n" + text
        elif prefix_text:
            text = prefix_text
        return (text,)


class ReaperMidiToScore:
    DESCRIPTION = "Converts REAPER MIDI into a YuE2 ABC score. Inputs: mode, vocal_track, instrument_track, optional comma-separated instrument_tracks, score_abc and settings. Output: score_abc STRING."
    """Convert a MIDI file prepared by REAPER into YuE2 score and lyrics strings."""

    @classmethod
    def INPUT_TYPES(cls):
        track_choices = ["auto"] + [str(index) for index in range(1, 65)]
        instrument_choices = ["auto", "none"] + [str(index) for index in range(1, 65)]
        return {
            "required": {
                "mode": (["melody", "full"], {"default": "full"}),
                "vocal_track": (track_choices, {"default": "auto"}),
                "instrument_track": (instrument_choices, {"default": "auto"}),
            },
            "optional": {
                "score_abc": ("STRING", {"multiline": True, "default": ""}),
                "without_sections": ("BOOLEAN", {"default": False}),
                "instrument_tracks": ("STRING", {"default": "", "multiline": False}),
                "refresh_token": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("score_abc",)
    FUNCTION = "load"
    CATEGORY = "REAPER AI/YuE2"

    def load(
        self,
        mode,
        vocal_track,
        instrument_track,
        score_abc="",
        without_sections=False,
        instrument_tracks="",
        refresh_token="",
    ):
        data = request_data()
        # Lyrics remain a separate REAPER Text Input output, but are still
        # passed internally to YuE2 so its auto vocal-track selection can use
        # syllable density and phrase timing.
        lyrics_hint = str(data.get("lyrics_text", data.get("text", "")) or "")
        midi_raw = data.get("midi_path")
        if not midi_raw:
            raise FileNotFoundError("REAPER MIDI input request has no midi_path")
        source = Path(str(midi_raw)).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"REAPER MIDI file not found: {source}")
        try:
            from folder_paths import get_input_directory

            ensure_yue2_path()
            from yue2_comfy.load_midi import YuE2LoadMidi
        except ImportError as error:
            raise RuntimeError(
                "YuE2-ComfyUI must be installed for REAPER MIDI to Score"
            ) from error
        if str(instrument_tracks or "").strip():
            instrument_track = str(instrument_tracks).strip()
        input_dir = Path(get_input_directory())
        input_dir.mkdir(parents=True, exist_ok=True)
        temp_name = f"reaper_{int(time.time() * 1000)}_{source.name}"
        temp_path = input_dir / temp_name
        shutil.copyfile(source, temp_path)
        try:
            result = YuE2LoadMidi().load(
                temp_name,
                mode,
                vocal_track,
                instrument_track,
                score_abc=score_abc,
                lyrics=lyrics_hint,
                without_sections=without_sections,
            )
            score_result = result["result"][0]
            score_result = (
                score_result.replace(r"\_", "_").replace(r"\^", "^").replace(r"\=", "=")
            )
            return (score_result,)
        finally:
            temp_path.unlink(missing_ok=True)


class ReaperScoreOutput:
    DESCRIPTION = "Saves a YuE2 ABC score and optional lyrics, and exports the score to MIDI. Inputs: score_abc, optional lyrics, label and export_midi. Outputs: abc_path, midi_path and manifest_path."
    """Save an ABC score, optional lyrics, and a YuE2-compatible MIDI export."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"score_abc": ("STRING", {"multiline": True})},
            "optional": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "label": ("STRING", {"default": "YuE2 score"}),
                "export_midi": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("abc_path", "midi_path", "manifest_path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "REAPER AI/YuE2"

    def save(self, score_abc, lyrics="", label="YuE2 score", export_midi=True):
        cache = cache_dir()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        abc_path = cache / f"reacomfy_score_{stamp}.abc"
        abc_path.write_text(str(score_abc), encoding="utf-8")
        lyrics_path = None
        if lyrics.strip():
            lyrics_path = cache / f"reacomfy_score_{stamp}.lyrics.txt"
            lyrics_path.write_text(str(lyrics), encoding="utf-8")
        midi_path = None
        midi_error = None
        if export_midi and str(score_abc).strip():
            try:
                if not str(score_abc).lstrip().startswith("X:1"):
                    raise ValueError(
                        "score_abc input is not YuE2 ABC (expected an X:1 header); connect YuE2 Generate Song's score_abc output"
                    )
                ensure_yue2_path()
                from yue2_comfy.midi.export import midi_of

                midi_path = cache / f"reacomfy_score_{stamp}.mid"
                midi_path.write_bytes(midi_of(str(score_abc)))
            except Exception as error:
                midi_error = str(error)
        manifest = cache / "reacomfy_score.json"
        payload = {
            "status": "complete",
            "label": label,
            "abc_path": str(abc_path),
            "midi_path": str(midi_path) if midi_path else None,
            "lyrics_path": str(lyrics_path) if lyrics_path else None,
        }
        if midi_error:
            payload["midi_error"] = midi_error
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        message = f"REAPER score ready: {abc_path}"
        if midi_path:
            message += f"; MIDI: {midi_path}"
        elif midi_error:
            message += f"; MIDI export failed: {midi_error}"
        return {
            "ui": {"text": [message]},
            "result": (
                str(abc_path),
                str(midi_path) if midi_path else "",
                str(manifest),
            ),
        }


class ReaperMidiOutput:
    DESCRIPTION = "Returns MIDI from any ComfyUI workflow to REAPER. Inputs: midi_path, optional notes_json and label. Output: manifest_path."
    """Copy a MIDI file produced by any ComfyUI workflow into the REAPER cache."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"midi_path": ("STRING", {"default": ""})},
            "optional": {
                "notes_json": ("STRING", {"default": ""}),
                "label": ("STRING", {"default": "ComfyUI MIDI"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "REAPER AI/YuE2"

    def save(self, midi_path, notes_json="", label="ComfyUI MIDI"):
        source = Path(str(midi_path or "")).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"MIDI file not found: {source}")
        if source.suffix.lower() not in (".mid", ".midi", ".kar", ".rmi"):
            raise ValueError("midi_path must point to a .mid, .midi, .kar or .rmi file")
        cache = cache_dir()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = (
            cache
            / f"reacomfy_midi_{stamp}_{int(time.time() * 1000) % 1000:03d}{source.suffix.lower()}"
        )
        shutil.copy2(str(source), str(output))
        notes_output = None
        notes_source = Path(str(notes_json or "")).expanduser()
        if notes_source.is_file():
            notes_output = cache / f"reacomfy_notes_{stamp}.json"
            shutil.copy2(str(notes_source), str(notes_output))
        manifest = cache / "reacomfy_midi.json"
        payload = {
            "status": "complete",
            "label": label,
            "midi_path": str(output),
            "notes_json": str(notes_output) if notes_output else None,
        }
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "ui": {"text": [f"REAPER MIDI ready: {output}"]},
            "result": (str(manifest),),
        }


class ReaperVideoOutput:
    DESCRIPTION = "Copies a video file produced by any ComfyUI workflow into the REAPER project cache. Input: video_path and optional label. Output: manifest_path."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"video_path": ("STRING", {"default": ""})},
            "optional": {"label": ("STRING", {"default": "ComfyUI Video"})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "REAPER AI/Video"

    def save(self, video_path, label="ComfyUI Video"):
        source = Path(str(video_path or "")).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Video file not found: {source}")
        allowed = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv")
        if source.suffix.lower() not in allowed:
            raise ValueError("video_path must point to a supported video file")
        cache = cache_dir()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = (
            cache
            / f"reacomfy_video_{stamp}_{int(time.time() * 1000) % 1000:03d}{source.suffix.lower()}"
        )
        shutil.copy2(str(source), str(output))
        manifest = cache / "reacomfy_video.json"
        payload = {"status": "complete", "label": label, "video_path": str(output)}
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "ui": {"text": [f"REAPER video ready: {output}"]},
            "result": (str(manifest),),
        }


class ReaperTextOutput:
    DESCRIPTION = "Exports subtitles or plain text from ComfyUI for REAPER. Accepts SRT/VTT timestamps or plain lines and imports them as note items, project markers, or regions."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"text": ("STRING", {"multiline": True, "default": ""})},
            "optional": {
                "mode": (["items", "regions", "markers"], {"default": "items"}),
                "track_name": ("STRING", {"default": "TTS Subtitles"}),
                "label": ("STRING", {"default": "ComfyUI Text"}),
                "start_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -86400.0, "max": 86400.0},
                ),
                "default_duration": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.05, "max": 300.0},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_path", "text")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "REAPER AI/Outputs"

    @staticmethod
    def _timecode(value):
        value = str(value).strip().replace(",", ".")
        parts = value.split(":")
        try:
            if len(parts) == 3:
                return (
                    float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
                )
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            return float(value)
        except ValueError:
            return None

    @classmethod
    def _segments(cls, raw, offset, default_duration):
        source = str(raw or "")
        candidate = Path(source.strip()).expanduser()
        if candidate.is_file():
            source = candidate.read_text(encoding="utf-8-sig", errors="replace")
        lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments, index = [], 0
        while index < len(lines):
            line = lines[index].strip()
            if not line or line.upper() == "WEBVTT":
                index += 1
                continue
            timing = line
            if (
                "-->" not in timing
                and index + 1 < len(lines)
                and "-->" in lines[index + 1]
            ):
                index += 1
                timing = lines[index].strip()
            if "-->" in timing:
                left, right = [
                    part.strip().split()[0] for part in timing.split("-->", 1)
                ]
                start, end = cls._timecode(left), cls._timecode(right)
                if start is not None and end is not None:
                    index += 1
                    body = []
                    while index < len(lines) and lines[index].strip():
                        body.append(lines[index].strip())
                        index += 1
                    text = "\n".join(body).strip()
                    if text:
                        segments.append(
                            (
                                max(0.0, start + offset),
                                max(start + offset + 0.05, end + offset),
                                text,
                            )
                        )
                    continue
            index += 1
        if segments:
            return segments
        plain = [line.strip() for line in lines if line.strip()]
        return [
            (offset + i * default_duration, offset + (i + 1) * default_duration, line)
            for i, line in enumerate(plain)
        ]

    def save(
        self,
        text,
        mode="items",
        track_name="TTS Subtitles",
        label="ComfyUI Text",
        start_offset=0.0,
        default_duration=4.0,
    ):
        cache, stamp = cache_dir(), time.strftime("%Y%m%d_%H%M%S")
        segments = self._segments(
            text, float(start_offset), max(0.05, float(default_duration))
        )
        text_path = cache / f"reacomfy_text_{stamp}.txt"
        text_path.write_text(str(text or ""), encoding="utf-8")
        segments_path = cache / f"reacomfy_text_{stamp}.tsv"
        with segments_path.open("w", encoding="utf-8", newline="\n") as handle:
            for start, end, value in segments:
                handle.write(
                    f"{start:.6f}\t{end:.6f}\t{json.dumps(value, ensure_ascii=False)[1:-1]}\n"
                )
        manifest = cache / "reacomfy_text.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mode": str(mode),
                    "track_name": str(track_name),
                    "label": str(label),
                    "text_path": str(text_path),
                    "segments_path": str(segments_path),
                    "segment_count": len(segments),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "ui": {
                "text": [f"REAPER text ready: {len(segments)} segment(s) | {manifest}"]
            },
            "result": (str(manifest), str(text or "")),
        }


class ReaperChordsToABC:
    DESCRIPTION = "Merges chord text into an existing YuE2 ABC score. Inputs: score_abc and chords. Output: score_abc STRING."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score_abc": ("STRING", {"multiline": True}),
                "chords": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("score_abc",)
    FUNCTION = "merge"
    CATEGORY = "REAPER AI/YuE2"

    def merge(self, score_abc, chords):
        names = []
        for line in str(chords or "").splitlines():
            value = line.split("|", 1)[-1].strip() if line.strip() else ""
            for token in value.replace(",", " ").split():
                token = token.strip("[](){}<>;:")
                if token and __import__("re").match(
                    r"^[A-Ga-g](?:#|b)?(?:m|min|maj|dim|aug|sus|add)?\d*(?:/[A-Ga-g](?:#|b)?)?$",
                    token,
                ):
                    names.append(token)
        output, active, index = [], False, 0
        for line in str(score_abc).splitlines():
            if line.startswith("V:"):
                active = not active
            if active and index < len(names) and "|" in line:
                parts = line.split("|")
                rebuilt = []
                for part_index, part in enumerate(parts):
                    clean_part = __import__("re").sub(r'"[^"]*"', "", part)
                    if index < len(names) and clean_part.strip():
                        clean_part = (
                            '"' + names[index].replace('"', "") + '"' + clean_part
                        )
                        index += 1
                    rebuilt.append(clean_part)
                    if part_index < len(parts) - 1:
                        rebuilt.append("|")
                line = "".join(rebuilt)
            line = line.replace(r"\_", "_").replace(r"\^", "^").replace(r"\=", "=")
            output.append(line)
        return ("\n".join(output),)


class ReaperChordsOnlyToABC:
    DESCRIPTION = "Builds a valid two-voice YuE2 ABC score from sectioned chords without MIDI. Inputs: chords, optional bpm and key. Output: score_abc STRING."
    """Build a valid YuE2 two-voice ABC score from sectioned chord text alone."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"chords": ("STRING", {"multiline": True})},
            "optional": {
                "bpm": ("INT", {"default": 92, "min": 1, "max": 300}),
                "key": ("STRING", {"default": "C"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("score_abc",)
    FUNCTION = "build"
    CATEGORY = "REAPER AI/YuE2"

    @staticmethod
    def _chord(value):
        import re

        token = str(value).strip().strip('"[](){}<>;,|')
        token = token.replace(":min", "m").replace(":maj", "")
        token = re.sub(r"maj(?=\d|$)", "", token, flags=re.IGNORECASE)
        token = re.sub(r"min(?=\d|$)", "m", token, flags=re.IGNORECASE)
        if token.endswith("M"):
            token = token[:-1]
        if re.fullmatch(
            r"[A-Ga-g](?:##|bb|#|b)?(?:m|dim|aug|sus2|sus4|add\d+|7|6|9|11|13)?(?:/[A-Ga-g](?:#|b)?)?",
            token,
        ):
            return token[0].upper() + token[1:]
        return None

    def build(self, chords, bpm=92, key="C"):
        import re

        sections = []
        current_name, current = None, []
        for raw in str(chords or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            section = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
            if section:
                if current_name and current:
                    sections.append((current_name, current))
                current_name, current = section.group(1).strip(), []
                line = section.group(2).strip()
            for value in re.split(r"[\s,|]+", line):
                chord = self._chord(value)
                if chord:
                    current.append(chord)
        if current_name and current:
            sections.append((current_name, current))
        if not sections:
            raise ValueError(
                "No chord symbols found. Use sectioned text such as [Verse] Fm Bbm Cm."
            )
        key = str(key or "C").strip() or "C"
        header = [
            "X:1",
            "T:",
            "M:4/4",
            "L:1/16",
            f"Q:1/4={int(bpm)}",
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
            'V: Ins clef=treble name="Ins Melody" snm="Inst."',
            f"K:{key}",
        ]
        output = list(header)
        for name, values in sections:
            output.append("% " + name.lower())
            for offset in range(0, len(values), 4):
                group = values[offset : offset + 4]
                output.append("V: Vocal")
                output.append("|".join('"' + chord + '"z16' for chord in group) + "|")
                output.append("V: Ins")
                output.append("|".join("Z" for _ in group) + "|")
        return ("\n".join(output),)


def _models_subdir(name):
    """Return ComfyUI/models/<name> (or ~/.cache/reacomfy/<name> outside ComfyUI)."""
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / name
    except Exception:
        base = Path.home() / ".cache" / "reacomfy" / name
    base.mkdir(parents=True, exist_ok=True)
    return base


def _hf_snapshot(repo_id, local_dir, allow_patterns=None):
    """Download a Hugging Face repo once into local_dir and return its path."""
    local_dir = Path(local_dir)
    marker = local_dir / ".reacomfy_complete"
    if marker.is_file():
        return local_dir
    _ensure_modules({"huggingface_hub": "huggingface_hub"}, "model downloads")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id, local_dir=str(local_dir), allow_patterns=allow_patterns
    )
    marker.write_text(repo_id, encoding="utf-8")
    return local_dir


def _resolve_device(device):
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _interrupt_checker():
    try:
        import comfy.model_management as model_management

        return model_management.throw_exception_if_processing_interrupted
    except Exception:
        return lambda: None


def _resample(waveform, source_rate, target_rate):
    """Resample [..., T] audio with torchaudio, falling back to linear interpolation."""
    source_rate, target_rate = int(source_rate), int(target_rate)
    if source_rate == target_rate:
        return waveform
    try:
        import torchaudio.functional as AF

        return AF.resample(waveform, source_rate, target_rate)
    except Exception:
        shape = waveform.shape
        flat = waveform.reshape(1, -1, shape[-1])
        size = max(1, round(shape[-1] * target_rate / source_rate))
        out = torch.nn.functional.interpolate(
            flat, size=size, mode="linear", align_corners=False
        )
        return out.reshape(*shape[:-1], size)


def _overlap_chunks(total, chunk, overlap):
    """Yield (start, end) windows of `chunk` samples that overlap by `overlap`."""
    if total <= chunk:
        yield 0, total
        return
    hop = max(1, chunk - overlap)
    start = 0
    while True:
        end = min(total, start + chunk)
        yield start, end
        if end >= total:
            return
        start += hop


def _process_in_chunks(signal, sample_rate, chunk_seconds, overlap_seconds, fn):
    """Run fn on overlapping chunks of a 1-D tensor and crossfade the results."""
    total = signal.shape[-1]
    if chunk_seconds <= 0:
        out = fn(signal)
        return _fit_length(out, total)
    chunk = max(1, int(chunk_seconds * sample_rate))
    overlap = min(int(overlap_seconds * sample_rate), chunk // 2)
    result = torch.zeros(total, dtype=torch.float32)
    weight = torch.zeros(total, dtype=torch.float32)
    check_interrupt = _interrupt_checker()
    for start, end in _overlap_chunks(total, chunk, overlap):
        check_interrupt()
        piece = _fit_length(fn(signal[start:end]), end - start)
        window = torch.ones(end - start)
        if overlap > 0:
            ramp = torch.linspace(0.0, 1.0, overlap + 2)[1:-1]
            if start > 0:
                window[:overlap] = ramp[: min(overlap, end - start)]
            if end < total:
                n = min(overlap, end - start)
                window[-n:] = torch.minimum(window[-n:], ramp.flip(0)[-n:])
        result[start:end] += piece * window
        weight[start:end] += window
    return result / weight.clamp_min(1e-6)


def _fit_length(samples, length):
    samples = torch.as_tensor(samples).detach().float().cpu().reshape(-1)
    if samples.shape[-1] >= length:
        return samples[:length]
    return torch.nn.functional.pad(samples, (0, length - samples.shape[-1]))


def _ensure_modules(requirements, purpose):
    """Install missing Python modules into ComfyUI's own environment.

    requirements: {import_name: pip_spec}. REAPER users should never need a
    terminal, so missing optional modules are installed on first use with the
    interpreter that runs ComfyUI (sys.executable). Installed packages that
    are already satisfied (e.g. torch) are never upgraded.
    """
    import importlib
    import subprocess

    missing = []
    for module, spec in requirements.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(spec)
    if not missing:
        return
    print(f"[ReaComfy] Installing modules for {purpose}: {' '.join(missing)}")
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    completed = subprocess.run(
        command + missing, capture_output=True, text=True, timeout=1800
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-1500:]
        raise RuntimeError(
            f"ReaComfy could not install {', '.join(missing)} for {purpose}:\n{tail}"
        )
    importlib.invalidate_caches()


def _download_github_zip(owner_repo, target, marker):
    """Download a GitHub repository's main branch as a zip into target."""
    import io
    import tempfile
    import urllib.request
    import zipfile

    url = f"https://codeload.github.com/{owner_repo}/zip/refs/heads/main"
    print(f"[ReaComfy] Downloading {owner_repo} code into {target}")
    with urllib.request.urlopen(url, timeout=300) as response:
        payload = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="reacomfy_code_", dir=str(target.parent)))
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(staging)
        roots = [p for p in staging.iterdir() if p.is_dir()]
        if len(roots) != 1 or not (roots[0] / marker).exists():
            raise RuntimeError(f"Unexpected {owner_repo} archive layout")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(roots[0]), str(target))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------
# Managed ZONOS3 runtime: ZONOS3 pins Python 3.12 and its own torch, so
# ReaComfy builds a private environment for it on first use (uv downloads
# Python 3.12 itself) and keeps one worker process with the model loaded.
# --------------------------------------------------------------------------

ZONOS3_TORCH = ("torch==2.9.1", "torchaudio==2.9.1")
ZONOS3_DEPS = [
    "descript-audio-codec==1.0.0",
    "transformers>=4.56.0,<=4.57.3",
    "huggingface_hub",
    "pyyaml",
    "omegaconf",
    "numpy",
    "soundfile",
    "tokenizers",
    "kernels>=0.12.1",
    "zyphra-textnorm==0.2.0",
    "lmdb>=1.4",
    "fastapi",
    "pydantic>=2",
    "pyicu-wheels>=2.14,<2.16",
]
ZONOS3_SETUP_VERSION = "1"

ZONOS3_WORKER = r'''
"""ReaComfy's ZONOS3 worker: loads the model once, then serves requests.

Protocol: one JSON object per stdin line {"request": GenerateRequest, "out": wav};
answers are stdout lines starting with "@@REACOMFY " followed by JSON.
"""
import json
import sys
import traceback


def say(payload):
    sys.stdout.write("@@REACOMFY " + json.dumps(payload) + "\n")
    sys.stdout.flush()


def main():
    device, dtype_name, bundle = sys.argv[1], sys.argv[2], sys.argv[3]
    import io
    import wave

    import soundfile as sf
    import torch

    import zonos3.tts.audio as zaudio
    from zonos3.generation import SAMPLE_RATE
    from zonos3.reference.lm import Zonos3LM
    from zonos3.reference.synth import ReferenceAssets, synthesize
    from zonos3.tts.audio import write_pcm16_wav
    from zonos3.utils import release

    original = zaudio._transcode_audio_bytes_to_wav

    def to_wav(audio_bytes):
        # ReaComfy sends PCM WAV; decode it without needing ffmpeg.
        try:
            data, rate = sf.read(io.BytesIO(audio_bytes), dtype="int16", always_2d=True)
        except Exception:
            return original(audio_bytes)
        out = io.BytesIO()
        with wave.open(out, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(data.mean(axis=1).astype("int16").tobytes())
        return out.getvalue()

    zaudio._transcode_audio_bytes_to_wav = to_wav
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    lm = Zonos3LM.load(release.RELEASE_LM, device=torch.device(device), dtype=dtype)
    assets = ReferenceAssets.load(steer_bundle=bundle or None, accent_adapters=None, laci_heads={})
    assets.grow(lm)
    if release.RELEASE_TEXT_ROWS:
        from zonos3.models.text_extra import TextExtraRows

        lm.append_text_rows(TextExtraRows.load(release.RELEASE_TEXT_ROWS))
    say({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            result = synthesize(lm, job["request"], assets=assets, device=torch.device(device), dtype=dtype)
            write_pcm16_wav(job["out"], result.pcm, SAMPLE_RATE)
            say({"ok": True, "seed": result.seed})
        except Exception:
            say({"ok": False, "error": traceback.format_exc()[-3000:]})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say({"ok": False, "fatal": True, "error": traceback.format_exc()[-3000:]})
        raise
'''


class _Zonos3Runtime:
    """Builds the private ZONOS3 environment and owns the worker process."""

    process = None
    process_key = None

    @staticmethod
    def base():
        return _models_subdir("zonos3")

    @classmethod
    def python(cls):
        venv = cls.base() / "venv"
        if os.name == "nt":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"

    @classmethod
    def _run(cls, command, log, timeout=3600):
        import subprocess

        log.write("\n$ " + " ".join(str(c) for c in command) + "\n")
        log.flush()
        completed = subprocess.run(
            [str(c) for c in command],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            cwd=str(cls.base()),
        )
        if completed.returncode != 0:
            log.flush()
            tail = Path(log.name).read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"ZONOS3 setup step failed:\n{tail}")

    @classmethod
    def torch_index(cls):
        if sys.platform == "darwin":
            return None
        if torch.cuda.is_available():
            return "https://download.pytorch.org/whl/cu128"
        return "https://download.pytorch.org/whl/cpu"

    @classmethod
    def ensure_installed(cls):
        base = cls.base()
        home = base / "ZONOS3"
        marker = base / f".setup_v{ZONOS3_SETUP_VERSION}"
        if marker.is_file() and cls.python().is_file() and home.is_dir():
            return home
        print(
            "[ReaComfy] First ZONOS3 run: preparing its own Python 3.12 environment "
            "(several GB, can take 10-30 minutes). Progress: " + str(base / "setup.log")
        )
        if not (home / "model" / "pyproject.toml").is_file():
            _download_github_zip("Zyphra/ZONOS3", home, "model/pyproject.toml")
        _ensure_modules({"uv": "uv"}, "the ZONOS3 environment")
        uv = [sys.executable, "-m", "uv"]
        with open(base / "setup.log", "a", encoding="utf-8") as log:
            if not cls.python().is_file():
                cls._run(uv + ["venv", "--python", "3.12", str(base / "venv")], log)
            pip = uv + ["pip", "install", "--python", str(cls.python())]
            index = cls.torch_index()
            torch_cmd = pip + list(ZONOS3_TORCH)
            if index:
                torch_cmd += ["--index-url", index]
            cls._run(torch_cmd, log)
            cls._run(pip + ZONOS3_DEPS, log)
            # fusealign-hub (speech editing only) is not on PyPI: skip deps.
            cls._run(pip + ["--no-deps", str(home / "model")], log)
        (base / "reacomfy_zonos3_worker.py").write_text(ZONOS3_WORKER, encoding="utf-8")
        marker.write_text("ok", encoding="utf-8")
        return home

    @classmethod
    def worker(cls, device, timeout):
        import subprocess

        home = cls.ensure_installed()
        script = cls.base() / "reacomfy_zonos3_worker.py"
        if not script.is_file():
            script.write_text(ZONOS3_WORKER, encoding="utf-8")
        device = _resolve_device(device)
        dtype = "bf16" if device == "cuda" else "f32"
        key = (device, dtype)
        process = cls.process
        if process is not None and process.poll() is None and cls.process_key == key:
            return process
        cls.stop()
        bundle = home / ZONOS3_BUNDLE
        log = open(cls.base() / "worker.log", "a", encoding="utf-8")
        print("[ReaComfy] Starting ZONOS3 (first start downloads ~9 GB of weights)")
        process = subprocess.Popen(
            [
                str(cls.python()),
                str(script),
                device,
                dtype,
                str(bundle) if bundle.is_file() else "",
            ],
            cwd=str(home),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        cls.process, cls.process_key = process, key
        reply = cls._read(process, max(int(timeout), 7200))
        if not reply.get("ready"):
            cls.stop()
            raise RuntimeError("ZONOS3 failed to start:\n" + reply.get("error", ""))
        return process

    @classmethod
    def _read(cls, process, timeout):
        import threading

        result = {}

        def reader():
            for line in process.stdout:
                if line.startswith("@@REACOMFY "):
                    result.update(json.loads(line[len("@@REACOMFY ") :]))
                    return

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            cls.stop()
            raise RuntimeError(f"ZONOS3 did not answer within {timeout} s")
        if not result:
            log = cls.base() / "worker.log"
            tail = (
                log.read_text(encoding="utf-8", errors="replace")[-2000:]
                if log.is_file()
                else ""
            )
            cls.stop()
            raise RuntimeError("ZONOS3 worker exited:\n" + tail)
        return result

    @classmethod
    def generate(cls, request, device, timeout):
        import tempfile

        process = cls.worker(device, timeout)
        out = Path(tempfile.mkdtemp(prefix="reacomfy_zonos3_")) / "out.wav"
        try:
            process.stdin.write(
                json.dumps({"request": request, "out": str(out)}) + "\n"
            )
            process.stdin.flush()
            reply = cls._read(process, timeout)
            if not reply.get("ok"):
                raise RuntimeError("ZONOS3 error:\n" + reply.get("error", ""))
            samples, rate = sf.read(str(out), dtype="float32", always_2d=True)
            return torch.from_numpy(samples.mean(axis=1)), int(rate), reply.get("seed")
        finally:
            shutil.rmtree(out.parent, ignore_errors=True)

    @classmethod
    def stop(cls):
        process, cls.process = cls.process, None
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=10)
            except Exception:
                process.kill()


# --------------------------------------------------------------------------
# DBLM-SE-1B (Yorch233, Apache-2.0): generative denoise + dereverb, 16 kHz.
# --------------------------------------------------------------------------

_DBLM_CACHE = {}


DBLM_SE_CODE_ZIP = "https://codeload.github.com/Yorch233/DBLM-SE/zip/refs/heads/main"
DBLM_SE_DEPS = "einops tabulate accelerate vector-quantize-pytorch transformers pyyaml"


def _download_dblm_code(target):
    """Fetch the DBLM-SE code (Apache-2.0) from GitHub into target."""
    _download_github_zip("Yorch233/DBLM-SE", target, "src/modeling_dblm.py")


def _dblm_code_dir():
    candidates = []
    configured = os.environ.get("DBLM_SE_REPO", "")
    if configured:
        candidates.append(Path(configured).expanduser())
    default = _models_subdir("dblm_se") / "DBLM-SE"
    candidates.append(default)
    candidates.append(Path(__file__).resolve().parent.parent / "DBLM-SE")
    for candidate in candidates:
        if (candidate / "src" / "modeling_dblm.py").is_file():
            return candidate
    try:
        _download_dblm_code(default)
        return default
    except Exception as error:
        raise RuntimeError(
            f"Could not download the DBLM-SE code ({error}). Clone it once "
            "(it is Apache-2.0):\n"
            f'  git clone https://github.com/Yorch233/DBLM-SE "{default}"\n'
            "or set DBLM_SE_REPO to an existing checkout."
        ) from error


class _RotaryPositionalEmbeddings(torch.nn.Module):
    """Same maths and buffer as torchtune.modules.RotaryPositionalEmbeddings.

    XCodec2 only needs this one class from torchtune, which pins torch/torchao
    versions that often clash with ComfyUI; it is used when torchtune is missing.
    """

    def __init__(self, dim, max_seq_len=4096, base=10_000):
        super().__init__()
        self.dim, self.base, self.max_seq_len = dim, base, max_seq_len
        theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.register_buffer("theta", theta, persistent=False)
        self._build(max_seq_len)

    def _build(self, length):
        seq = torch.arange(length, dtype=self.theta.dtype, device=self.theta.device)
        angles = torch.einsum("i,j->ij", seq, self.theta).float()
        cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x, *, input_pos=None):
        seq_len = x.size(1)
        if input_pos is None and seq_len > self.cache.shape[0]:
            self._build(seq_len)
        rope = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        shaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope = rope.view(-1, shaped.size(1), 1, shaped.size(3), 2)
        out = torch.stack(
            [
                shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1],
                shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1],
            ],
            -1,
        )
        return out.flatten(3).type_as(x)


def _torchtune_standin():
    try:
        import torchtune.modules  # noqa: F401

        return {}
    except Exception:
        import types

        package = types.ModuleType("torchtune")
        modules = types.ModuleType("torchtune.modules")
        modules.RotaryPositionalEmbeddings = _RotaryPositionalEmbeddings
        package.modules = modules
        package.__path__ = []
        return {"torchtune": package, "torchtune.modules": modules}


class _IsolatedImport:
    """Import DBLM-SE's generic top-level packages (src, library) without
    clobbering same-named packages from other custom nodes."""

    NAMES = ("src", "library")

    def __init__(self, repo, names=None, modules=None):
        self.repo = str(repo)
        if names is not None:
            self.NAMES = tuple(names)
        self.modules = dict(modules or {})  # pre-bound stand-ins (e.g. mamba_ssm)

    def _owned(self):
        return {
            key: module
            for key, module in sys.modules.items()
            if key.split(".")[0] in self.NAMES
        }

    def __enter__(self):
        self.saved = self._owned()
        for key in self.saved:
            del sys.modules[key]
        self.cwd = os.getcwd()
        os.chdir(self.repo)  # DBLM-SE reads config/models.yaml relative to cwd
        sys.path.insert(0, self.repo)
        # Bind each top-level package to this checkout explicitly. Folders
        # without __init__.py would otherwise lose to any regular package of
        # the same name elsewhere on sys.path (common with many custom nodes).
        import importlib.util
        import types

        sys.modules.update(self.modules)
        for name in self.NAMES:
            folder = Path(self.repo) / name
            if not folder.is_dir():
                continue
            init = folder / "__init__.py"
            if init.is_file():
                spec = importlib.util.spec_from_file_location(
                    name, init, submodule_search_locations=[str(folder)]
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            else:
                module = types.ModuleType(name)
                module.__path__ = [str(folder)]
                module.__package__ = name
                sys.modules[name] = module
        return self

    def __exit__(self, *exc):
        os.chdir(self.cwd)
        try:
            sys.path.remove(self.repo)
        except ValueError:
            pass
        for key in self._owned():
            del sys.modules[key]
        sys.modules.update(self.saved)
        return False


def _load_dblm(device):
    cached = _DBLM_CACHE.get(device)
    if cached is not None:
        return cached
    _ensure_modules(
        {
            "einops": "einops",
            "tabulate": "tabulate",
            "accelerate": "accelerate",
            "vector_quantize_pytorch": "vector-quantize-pytorch",
            "transformers": "transformers",
            "yaml": "pyyaml",
            "safetensors": "safetensors",
            "huggingface_hub": "huggingface_hub",
        },
        "DBLM-SE",
    )
    repo = _dblm_code_dir()
    weights = _hf_snapshot(
        "Yorch233/DBLM-SE-1B", _models_subdir("dblm_se") / "DBLM-SE-1B"
    )
    wavlm = sorted((weights / "WavLM").glob("*.pt"))
    codec = sorted((weights / "Xcodec2").glob("*.safetensors"))
    if not wavlm or not codec or not (weights / "Wav2vec").is_dir():
        raise RuntimeError(
            f"DBLM-SE-1B download looks incomplete in {weights}. Delete the folder "
            "and run again."
        )
    torch_device = torch.device(device)
    try:
        standin = _torchtune_standin()
        with _IsolatedImport(
            repo, names=("src", "library", *sorted(standin)), modules=standin
        ):
            import library  # noqa: F401  registers WavLM and XCodec2
            from library.register import ModelRegister
            from safetensors.torch import load_file
            from src.common.config import read_config_from_yaml
            from src.modeling_dblm import DBLMForSE
            from src.sde import DDBM_VPSDE

            config = read_config_from_yaml(str(weights / "config.yml"))
            encoder = ModelRegister.fetch("WavLM")(
                model_path=str(wavlm[0]), output_layer=6, device=torch_device
            )
            codec_model = ModelRegister.fetch("XCodec2")(
                model_paths={
                    "wav2vec": str(weights / "Wav2vec"),
                    "codec": str(codec[0]),
                },
                codec_dim=2048,
                semantic_dim=1024,
                device=torch_device,
            )
            model = DBLMForSE(
                in_dim=config.latent_dim,
                d_model=config.model_dim,
                nhead=config.num_head,
                num_layers=config.num_layers,
            )
            state = load_file(str(weights / "model.safetensors"))
            # The checkpoint was saved from a torch.compile wrapper.
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state)
            model.to(torch_device).eval()
            sde = DDBM_VPSDE(
                beta=config.beta, t_max=config.t_max, t_min=0, device=torch_device
            )
    except ImportError as error:
        raise RuntimeError(
            f"DBLM-SE needs an extra module ({error.name}). Install into the ComfyUI "
            f"environment: python -m pip install {DBLM_SE_DEPS}"
        ) from error
    _DBLM_CACHE.clear()
    _DBLM_CACHE[device] = (encoder, codec_model, model, sde)
    return _DBLM_CACHE[device]


class ReaperDBLMSE:
    DESCRIPTION = (
        "Generative speech denoise + dereverb with DBLM-SE-1B (Yorch233, Apache-2.0). "
        "The model rebuilds speech from WavLM features through an XCodec2 decoder, so "
        "badly damaged dialogue comes back clean, but the result is resynthesized at "
        "16 kHz. Chain REAPER ClearVoice Super Resolution or LavaSR after it for 48 kHz. "
        "Needs the DBLM-SE code checkout and ~11 GB of weights (auto-download); a CUDA "
        "GPU with about 8 GB or more is recommended."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "steps": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 30,
                        "tooltip": "Diffusion-bridge sampling steps. 3-10 is typical; more is slower.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "1.0 = only the restored voice; lower values mix the (resampled) dry signal back in.",
                    },
                ),
                "output_rate": (
                    ["original", "16000"],
                    {
                        "default": "original",
                        "tooltip": "original resamples the 16 kHz result back to the input rate (no new highs).",
                    },
                ),
            },
            "optional": {
                "chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 1.0,
                        "tooltip": "Process long files in chunks to bound VRAM. 0 = whole file at once.",
                    },
                ),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "enhance"
    CATEGORY = "REAPER AI/Enhancement"

    def enhance(
        self,
        audio,
        steps=5,
        strength=1.0,
        output_rate="original",
        chunk_seconds=12.0,
        device="auto",
        keep_model_loaded=True,
    ):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER DBLM-SE requires an AUDIO input")
        waveform = audio["waveform"].detach().float().cpu()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or waveform.shape[0] != 1:
            raise ValueError("REAPER DBLM-SE supports one [1,C,T] audio at a time")
        device = _resolve_device("cpu" if device == "cpu" else device)
        if device == "mps":
            device = "cpu"
        encoder, codec_model, model, sde = _load_dblm(device)
        torch_device = torch.device(device)
        mono = _resample(waveform[0].mean(dim=0), sample_rate, 16000)
        peak = float(mono.abs().max())
        scale = 1.0 / peak if peak > 0 else 1.0

        def run(chunk):
            x = (chunk * scale).to(torch_device).reshape(1, 1, -1)
            with torch.no_grad():
                latents = encoder(x)
                solver = sde.get_sde_solver(
                    model_fn=lambda xt, t: model(
                        xt,
                        torch.full(
                            (xt.shape[0],), t, device=xt.device, dtype=torch.float32
                        ),
                        return_logits=False,
                    )
                )
                clean = solver.sampling(x=latents, num_steps=int(steps))
                tokens = torch.argmax(model.get_logits(clean), dim=1)
                out = codec_model.token2audio(tokens.unsqueeze(0))
            return out.float().cpu() / scale

        try:
            restored = _process_in_chunks(mono, 16000, float(chunk_seconds), 1.0, run)
        finally:
            if not keep_model_loaded:
                _DBLM_CACHE.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if output_rate == "16000":
            out_rate = 16000
            dry = mono
        else:
            out_rate = sample_rate
            restored = _resample(restored, 16000, sample_rate)
            dry = waveform[0].mean(dim=0)
        restored = _fit_length(restored, dry.shape[-1])
        amount = float(strength)
        mixed = dry + amount * (restored - dry) if amount < 1.0 else restored
        return ({"waveform": mixed.reshape(1, 1, -1), "sample_rate": out_rate},)


# --------------------------------------------------------------------------
# ZONOS3-Mini (Zyphra; code MIT, weights Apache-2.0): multilingual TTS with
# voice cloning and steering controls (emotion, pace, accent, texture...).
# The runtime pins its own Python 3.12 / torch stack, so ReaComfy talks to it
# out of process: either a running ZONOS3 server (any backend, fastest) or
# the plain-PyTorch reference in its own venv.
# --------------------------------------------------------------------------

ZONOS3_LANGUAGES = [
    "uk", "en_us", "en_gb", "ru", "pl", "cs", "sk", "bg", "ro", "hu", "de",
    "fr_fr", "es", "it", "pt_br", "nl", "da", "no", "sv", "fi", "el", "tr",
    "he", "ar", "ka", "hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta",
    "te", "ur", "id", "ms", "tl", "th", "vi", "ja", "ko", "cmn",
]  # fmt: skip

# Studio-bundle emotion controls and the doses Zyphra recommends for the
# release LM ("angry and happy at x2-3, sad at x4-6, calm at x4, scared at x8,
# confused at x2"). emotion_strength scales these.
ZONOS3_EMOTIONS = {
    "none": None,
    "angry": ("emotions.angry", 2.5),
    "happy": ("emotions.happy", 2.5),
    "sad": ("emotions.sad", 5.0),
    "calm": ("emotions.serene", 4.0),
    "scared": ("emotions.scared", 8.0),
    "confused": ("emotions.confused", 2.0),
}

ZONOS3_ACCENTS = {
    "none": None,
    "american": {"accents_en.american": 1.0},
    "british": {"accents_en.british": 1.0},
    "scottish": {"accents_en.scottish": 1.0},
    "irish": {"accents_en.irish": 1.0},
    "welsh": {"accents_en.welsh": 1.0},
    "australian": {"accents_en.australian": 1.0},
    "indian english": {"accents_en.indian": 1.0},
    "hindi": {"accents_l1.hin": 1.0},
    "spanish": {"accents_l1.spa": 1.0},
    "french": {"accents_l1.fra": 1.0},
    "portuguese": {"accents_l1.por": 1.0},
    "german": {"accents_l1.deu": 1.0},
    "vietnamese": {"accents_l1.vie": 1.0},
    "polish": {"accents_l1.pol": 1.0},
    "mandarin": {"mandarin.cmn": 1.0, "mandarin_hidden.cmn": 1.0 / 3.0},
}

# Bipolar studio sliders: (negative direction, positive direction).
ZONOS3_BIPOLAR = {
    "gender": ({"identity.gender": -1}, {"identity.gender": 1}),
    "age": ({"identity.age": -1}, {"identity.age": 1}),
    "pitch": ({"pitch.pitch": -1}, {"pitch.pitch": 1}),
    "intonation": ({"pitch.pitch_range": -1}, {"pitch.pitch_range": 1}),
    "room": ({"room.clean": -1}, {"room.clean": 1}),
    "projection": ({"delivery.whisper": 1}, {"delivery.loud": 1}),
    "pace": ({"delivery.tempo": -1}, {"delivery.tempo": 1}),
    "chatty": ({"delivery.conversational": -1}, {"delivery.conversational": 1}),
    "liveliness": ({"expressive.expressive": -1}, {"expressive.expressive": 1}),
    "hesitation": ({"disfluency.disfluent": -1}, {"disfluency.disfluent": 1}),
    "valence": ({"mood.valence": -1}, {"mood.valence": 1}),
    "arousal": ({"mood.arousal": -1}, {"mood.arousal": 1}),
}

ZONOS3_TEXTURE = {
    "deep": "texture.deep",
    "bright": "texture.shrill",
    "husky": "texture.husky",
    "raspy": "texture.raspy",
    "soft": "texture.soft",
    "authority": "texture.authoritative",
}

ZONOS3_PRESETS = {
    "none": {},
    "BBC newsreader": {"accent": ("british", 1.5), "authority": 3, "room": 2},
    "Elderly Scottish narrator": {
        "age": 1.5,
        "accent": ("scottish", 1.5),
        "deep": 2,
        "intonation": -1,
    },
    "Gravelly old man": {"gender": -1, "age": 2, "raspy": 3, "deep": 2},
    "Chatty podcaster": {"chatty": 3, "room": 1, "liveliness": 4},
    "Late-night radio": {"deep": 2, "room": 3, "pace": -0.8, "projection": -0.6},
    "Phone call": {"room": -2},
}

ZONOS3_LM = "hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini"
ZONOS3_BUNDLE = "assets/steering/studio_bundle_mect60k_x10.json"


def zonos3_controls(
    sliders=None, texture=None, emotion="none", emotion_strength=1.0,
    accent="none", accent_strength=1.0, preset="none", extra=None,
):  # fmt: skip
    """Translate ReaComfy sliders into ZONOS3 raw steering doses."""
    sliders = dict(sliders or {})
    texture = dict(texture or {})
    preset_values = dict(ZONOS3_PRESETS.get(preset, {}))
    preset_accent = preset_values.pop("accent", None)
    for key, value in preset_values.items():
        if key in ZONOS3_TEXTURE:
            texture[key] = texture.get(key, 0.0) or value
        else:
            sliders[key] = sliders.get(key, 0.0) or value
    controls = {}

    def add(mapping, dose):
        for direction, weight in mapping.items():
            controls[direction] = controls.get(direction, 0.0) + weight * dose

    for key, value in sliders.items():
        value = float(value or 0.0)
        if value == 0.0 or key not in ZONOS3_BIPOLAR:
            continue
        negative, positive = ZONOS3_BIPOLAR[key]
        add(positive if value > 0 else negative, abs(value))
    for key, value in texture.items():
        value = float(value or 0.0)
        if value > 0.0 and key in ZONOS3_TEXTURE:
            add({ZONOS3_TEXTURE[key]: 1.0}, value)
    if emotion in ZONOS3_EMOTIONS and ZONOS3_EMOTIONS[emotion] is not None:
        direction, dose = ZONOS3_EMOTIONS[emotion]
        if float(emotion_strength) > 0:
            add({direction: 1.0}, dose * float(emotion_strength))
    if accent == "none" and preset_accent:
        accent, accent_strength = preset_accent
    if ZONOS3_ACCENTS.get(accent) and float(accent_strength) > 0:
        add(ZONOS3_ACCENTS[accent], float(accent_strength))
    for direction, dose in (extra or {}).items():
        add({str(direction): 1.0}, float(dose))
    return {k: round(v, 4) for k, v in controls.items() if abs(v) > 1e-6}


def zonos3_split_text(text, mode):
    text = str(text or "").strip()
    if not text:
        return []
    if mode == "whole text":
        return [text]
    if mode == "paragraphs":
        parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    else:  # lines
        parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n")]
    return [p for p in parts if p]


def _zonos3_server_generate(url, request, token, timeout):
    import urllib.error
    import urllib.request

    body = dict(request, stream=False, format="wav")
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v2/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            seed = response.headers.get("X-Seed")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"ZONOS3 server error {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach the ZONOS3 server at {url} ({error.reason}). Start it first "
            "(see README: ZONOS3-Mini), or switch backend to 'reference (venv)'."
        ) from error
    import io

    samples, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    return torch.from_numpy(samples.mean(axis=1)), int(rate), seed


def _zonos3_reference_generate(python, home, request, device, dtype, timeout):
    import subprocess
    import tempfile

    python = python or os.environ.get("ZONOS3_PYTHON", "")
    home = home or os.environ.get("ZONOS3_HOME", "")
    if not python or not home:
        raise RuntimeError(
            "The 'reference (venv)' backend needs the ZONOS3 checkout and its Python: "
            "set zonos3_home / zonos3_python on the node, or the ZONOS3_HOME and "
            "ZONOS3_PYTHON environment variables (see README: ZONOS3-Mini)."
        )
    workdir = Path(tempfile.mkdtemp(prefix="reacomfy_zonos3_"))
    try:
        request_file = workdir / "request.json"
        output = workdir / "out.wav"
        request_file.write_text(json.dumps(request), encoding="utf-8")
        command = [
            python, "-m", "zonos3.reference", "--model", ZONOS3_LM,
            "--request", str(request_file), "-o", str(output),
            "--dtype", dtype,
        ]  # fmt: skip
        if device != "auto":
            command += ["--device", device]
        bundle = Path(home) / ZONOS3_BUNDLE
        if request.get("controls") and bundle.is_file():
            command += ["--steer-bundle", str(bundle)]
        completed = subprocess.run(
            command, cwd=home, capture_output=True, text=True, timeout=timeout
        )
        if completed.returncode != 0 or not output.is_file():
            tail = (completed.stderr or completed.stdout or "")[-1500:]
            raise RuntimeError(f"ZONOS3 reference failed:\n{tail}")
        samples, rate = sf.read(str(output), dtype="float32", always_2d=True)
        seed = None
        for token in (completed.stdout or "").split(","):
            if token.strip().startswith("seed "):
                seed = token.strip()[5:]
        return torch.from_numpy(samples.mean(axis=1)), int(rate), seed
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class ReaperZonos3TTS:
    DESCRIPTION = (
        "ZONOS3-Mini text-to-speech (Zyphra; code MIT, weights Apache-2.0, 43 languages "
        "incl. Ukrainian). Clones the voice of the optional reference AUDIO (5-30 s of "
        "clean speech, e.g. the selected REAPER item) and steers emotion, pace, "
        "projection, accent, texture and room. Runs out of process: a ZONOS3 server "
        "(fast) or the reference in its own Python venv. Output: 44.1 kHz mono."
    )

    @classmethod
    def INPUT_TYPES(cls):
        def bipolar(lo, hi, tip):
            return (
                "FLOAT",
                {"default": 0.0, "min": lo, "max": hi, "step": 0.1, "tooltip": tip},
            )

        def unipolar(hi, tip):
            return (
                "FLOAT",
                {"default": 0.0, "min": 0.0, "max": hi, "step": 0.1, "tooltip": tip},
            )

        return {
            "required": {
                "text": (
                    "STRING",
                    {"multiline": True, "default": "Привіт! Це тест голосу ZONOS3."},
                ),
                "language": (ZONOS3_LANGUAGES, {"default": "uk"}),
                "backend": (
                    ["automatic", "server", "reference (venv)"],
                    {
                        "default": "automatic",
                        "tooltip": "automatic: ReaComfy installs ZONOS3 in its own environment on first use and keeps it loaded (nothing to set up). server: an existing ZONOS3 server. reference: zonos3.reference from your own venv.",
                    },
                ),
                "emotion": (list(ZONOS3_EMOTIONS), {"default": "none"}),
                "emotion_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "1.0 = Zyphra's recommended dose for the chosen emotion.",
                    },
                ),
                "pace": bipolar(-2, 2, "slower <-> faster (-1 = x1.45 longer)."),
                "projection": bipolar(-1.5, 2, "whisper <-> loud."),
                "valence": bipolar(-4, 4, "Mood: negative <-> positive."),
                "arousal": bipolar(-4, 4, "Mood: calm <-> excited."),
                "accent": (list(ZONOS3_ACCENTS), {"default": "none"}),
                "accent_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4294967295,
                        "tooltip": "0 = random. Keep a seed to get the same take again.",
                    },
                ),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "speaker_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Server backend: a voice id from GET /v2/speakers, used when no reference_audio is connected.",
                    },
                ),
                "preset": (list(ZONOS3_PRESETS), {"default": "none"}),
                "chatty": bipolar(-4, 4, "careful reading <-> spontaneous."),
                "liveliness": bipolar(-8, 8, "flat <-> lively intonation."),
                "hesitation": bipolar(-8, 8, "fluent <-> hesitant."),
                "gender": bipolar(-2, 2, "masculine <-> feminine (changes the voice)."),
                "age": bipolar(-2, 2, "younger <-> older."),
                "pitch": bipolar(-2, 2, "lower <-> higher."),
                "intonation": bipolar(-2, 2, "monotone <-> varied."),
                "room": bipolar(-2, 4, "phone <-> studio."),
                "deep": unipolar(4, "Deeper, huskier."),
                "bright": unipolar(4, "Shriller."),
                "husky": unipolar(4, "Huskier."),
                "raspy": unipolar(4, "Raspy (changes the voice the most)."),
                "soft": unipolar(4, "Softer and quieter."),
                "authority": unipolar(4, "More authoritative."),
                "extra_controls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": 'Advanced: JSON of raw bundle doses, e.g. {"emotions.relieved": 3}.',
                    },
                ),
                "split": (
                    ["lines", "paragraphs", "whole text"],
                    {
                        "default": "lines",
                        "tooltip": "Generate each line/paragraph separately (stable on long scripts) and join them.",
                    },
                ),
                "gap_ms": ("INT", {"default": 350, "min": 0, "max": 5000, "step": 50}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.1, "max": 1.5, "step": 0.05},
                ),
                "speaker_cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Closer to the reference voice (>1 doubles compute). Cannot be combined with laci.",
                    },
                ),
                "laci": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Alignment-guarded decoding: fewer skipped/repeated words (server needs LACI heads).",
                    },
                ),
                "server_url": ("STRING", {"default": "http://127.0.0.1:1919"}),
                "access_token": ("STRING", {"default": ""}),
                "zonos3_home": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Reference backend: ZONOS3 checkout (or env ZONOS3_HOME).",
                    },
                ),
                "zonos3_python": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Reference backend: python of the ZONOS3 venv (or env ZONOS3_PYTHON).",
                    },
                ),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "timeout_s": ("INT", {"default": 900, "min": 30, "max": 7200}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "speak"
    CATEGORY = "REAPER AI/Voice"

    def build_request(self, text, language, controls, seed, kwargs, speaker):
        request = {
            "text": text,
            "language": language,
            "temperature": float(kwargs.get("temperature", 0.75)),
        }
        if int(seed) > 0:
            request["seed"] = int(seed)
        request.update(speaker)
        if controls:
            request["controls"] = controls
            request["speaker_mode"] = "embedding"
        cfg = float(kwargs.get("speaker_cfg_scale", 1.0))
        if kwargs.get("laci"):
            request["laci"] = True
        elif cfg > 1.0:
            request["speaker_cfg_scale"] = cfg
        return request

    def speak(
        self,
        text,
        language="uk",
        backend="automatic",
        emotion="none",
        emotion_strength=1.0,
        pace=0.0,
        projection=0.0,
        valence=0.0,
        arousal=0.0,
        accent="none",
        accent_strength=1.0,
        seed=0,
        reference_audio=None,
        speaker_id="",
        preset="none",
        extra_controls="",
        split="lines",
        gap_ms=350,
        server_url="http://127.0.0.1:1919",
        access_token="",
        zonos3_home="",
        zonos3_python="",
        device="auto",
        timeout_s=900,
        **kwargs,
    ):
        import base64
        import io

        parts = zonos3_split_text(text, split)
        if not parts:
            raise ValueError("REAPER ZONOS3 TTS: text is empty")
        sliders = {
            "pace": pace,
            "projection": projection,
            "valence": valence,
            "arousal": arousal,
        }
        for key in ZONOS3_BIPOLAR:
            if key in kwargs:
                sliders[key] = kwargs[key]
        texture = {key: kwargs.get(key, 0.0) for key in ZONOS3_TEXTURE}
        extra = {}
        if str(extra_controls or "").strip():
            try:
                extra = json.loads(extra_controls)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"extra_controls is not valid JSON: {error}"
                ) from error
            if not isinstance(extra, dict):
                raise ValueError(
                    'extra_controls must be a JSON object like {"emotions.relieved": 3}'
                )
        controls = zonos3_controls(
            sliders, texture, emotion, emotion_strength, accent, accent_strength,
            preset, extra,
        )  # fmt: skip

        speaker = {}
        if reference_audio is not None and "waveform" in reference_audio:
            ref = reference_audio["waveform"].detach().float().cpu()
            ref = ref.reshape(-1, ref.shape[-1]).mean(dim=0)
            ref_rate = int(reference_audio["sample_rate"])
            ref = ref[: ref_rate * 60]  # the speaker encoder reads up to 60 s
            buffer = io.BytesIO()
            sf.write(buffer, ref.numpy(), ref_rate, format="WAV", subtype="PCM_16")
            speaker["speaker_audio_base64"] = base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")
            speaker["speaker_audio_name"] = "reaper_reference.wav"
        elif str(speaker_id or "").strip():
            speaker["speaker_embedding_id"] = str(speaker_id).strip()

        check_interrupt = _interrupt_checker()
        pieces, rate, seeds = [], 44100, []
        dtype = "bf16" if _resolve_device(device) == "cuda" else "f32"
        for part in parts:
            check_interrupt()
            request = self.build_request(
                part, language, controls, seed, kwargs, speaker
            )
            if backend == "automatic":
                samples, rate, used_seed = _Zonos3Runtime.generate(
                    request, device, int(timeout_s)
                )
            elif backend == "server":
                samples, rate, used_seed = _zonos3_server_generate(
                    server_url, request, access_token, int(timeout_s)
                )
            else:
                samples, rate, used_seed = _zonos3_reference_generate(
                    zonos3_python, zonos3_home, request, device, dtype, int(timeout_s)
                )
            seeds.append(str(used_seed))
            if pieces and int(gap_ms) > 0:
                pieces.append(torch.zeros(int(rate * int(gap_ms) / 1000)))
            pieces.append(samples)
        audio_out = torch.cat(pieces).reshape(1, 1, -1)
        info = json.dumps(
            {
                "language": language,
                "segments": len(parts),
                "seeds": seeds,
                "controls": controls,
                "voice": "reference"
                if "speaker_audio_base64" in speaker
                else (speaker_id or "default"),
            },
            ensure_ascii=False,
        )
        return ({"waveform": audio_out, "sample_rate": rate}, info)


# --------------------------------------------------------------------------
# NVIDIA Nemotron RE-USE (OpenMDW-1.1, commercial use allowed): universal
# speech enhancement (noise, reverb, clipping, codec, bandwidth) at 8-48 kHz.
# The research variants nvidia/RE-USE and Real-time_RE-USE are non-commercial
# and deliberately not offered here.
# --------------------------------------------------------------------------

REUSE_REPO = "nvidia/NVIDIA-NemotronLabs-RE-USE"
_REUSE_CACHE = {}


class _MambaFallback(torch.nn.Module):
    """Pure-PyTorch Mamba (v1) with mamba_ssm's parameter names.

    Used when the CUDA-only mamba_ssm package is missing (Windows, macOS, CPU).
    Numerically the same selective scan as mamba_ssm's reference path, just
    slower; the checkpoint loads unchanged.
    """

    def __init__(
        self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
        conv_bias=True, bias=False, **_ignored,
    ):  # fmt: skip
        super().__init__()
        import math

        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = torch.nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = torch.nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=conv_bias,
        )  # fmt: skip
        self.x_proj = torch.nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )
        self.dt_proj = torch.nn.Linear(self.dt_rank, self.d_inner, bias=True)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = torch.nn.Parameter(torch.log(a))
        self.D = torch.nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = torch.nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states):
        batch, length, _ = hidden_states.shape
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2)
        x = torch.nn.functional.silu(x)
        dt, b, c = torch.split(
            self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = torch.nn.functional.softplus(self.dt_proj(dt))  # [B, L, D]
        a = -torch.exp(self.A_log.float())  # [D, N]
        y = self._scan(x.float(), dt.float(), a, b.float(), c.float())
        y = y + x.float() * self.D.float()
        y = y.to(hidden_states.dtype) * torch.nn.functional.silu(z)
        return self.out_proj(y)

    @staticmethod
    def _scan(x, dt, a, b, c):
        """Exact selective scan, one step at a time (stable for any decay)."""
        batch, length, dim = x.shape
        state = x.new_zeros(batch, dim, a.shape[-1])
        outputs = []
        for t in range(length):
            dt_t = dt[:, t, :, None]
            state = (
                torch.exp(dt_t * a) * state + dt_t * b[:, t, None, :] * x[:, t, :, None]
            )
            outputs.append((state * c[:, t, None, :]).sum(-1))
        return torch.stack(outputs, dim=1)


def _reuse_files():
    root = _hf_snapshot(REUSE_REPO, _models_subdir("nemotron_reuse") / "RE-USE")
    recipes = sorted((root / "recipes").glob("*.yaml"))
    if not (root / "models").is_dir() or not recipes:
        raise RuntimeError(
            f"RE-USE download looks incomplete in {root}. Delete the folder and run again."
        )
    return root, recipes[0]


def _load_reuse(device, backend):
    key = (device, backend)
    if key in _REUSE_CACHE:
        return _REUSE_CACHE[key]
    import types

    _ensure_modules(
        {
            "einops": "einops",
            "yaml": "pyyaml",
            "safetensors": "safetensors",
            "huggingface_hub": "huggingface_hub",
        },
        "Nemotron RE-USE",
    )
    import yaml
    from safetensors.torch import load_file

    root, recipe = _reuse_files()
    cfg = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    use_native = backend == "mamba_ssm" or (backend == "auto" and device == "cuda")
    native = None
    if use_native:
        try:
            import mamba_ssm as native
        except Exception:
            if backend == "mamba_ssm":
                raise RuntimeError(
                    "mamba_ssm is not installed (Linux + CUDA only). Use mamba backend "
                    "'pytorch' or install it: python -m pip install mamba-ssm"
                )
            native = None
    shim = native or types.SimpleNamespace(Mamba=_MambaFallback)
    importer = _IsolatedImport(
        root, names=("models", "utils", "mamba_ssm"), modules={"mamba_ssm": shim}
    )
    with importer:
        from models.generator_SEMamba_time_d4 import SEMamba
        from models.stfts import mag_phase_istft, mag_phase_stft

        model = SEMamba(cfg)
    state = load_file(str(root / "model.safetensors"))
    model.load_state_dict(state)
    model.to(torch.device(device)).eval()
    entry = (model, cfg, mag_phase_stft, mag_phase_istft, native is not None)
    _REUSE_CACHE.clear()
    _REUSE_CACHE[key] = entry
    return entry


def _make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


class ReaperNemotronReUSE:
    DESCRIPTION = (
        "NVIDIA Nemotron RE-USE universal speech enhancement (OpenMDW-1.1, commercial "
        "use allowed): removes noise and reverb, repairs clipping, codec and packet-loss "
        "damage at the source rate (8-48 kHz), keeps the main speaker. Optional bandwidth "
        "extension. 9.6M params; weights (~40 MB) download on first use. Uses mamba_ssm on "
        "Linux+CUDA, otherwise a slower built-in PyTorch path (fine on a GPU, very slow on CPU)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "1.0 = fully enhanced; lower values blend the dry signal back in.",
                    },
                ),
                "bandwidth_extension": (
                    ["off", "32000", "44100", "48000"],
                    {
                        "default": "off",
                        "tooltip": "Resample to this rate first so the model regenerates the missing highs (e.g. 8/16 kHz phone or Zoom audio).",
                    },
                ),
                "channels": (
                    ["per channel", "mono"],
                    {"default": "per channel"},
                ),
            },
            "optional": {
                "chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.5,
                        "tooltip": "NVIDIA's long-file recipe uses 5 s chunks with 50% overlap. 0 = whole file.",
                    },
                ),
                "mamba_backend": (
                    ["auto", "mamba_ssm", "pytorch"],
                    {
                        "default": "auto",
                        "tooltip": "auto uses mamba_ssm when it is installed on CUDA, otherwise the portable PyTorch scan.",
                    },
                ),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("speech", "removed")
    FUNCTION = "enhance"
    CATEGORY = "REAPER AI/Enhancement"

    def enhance(
        self,
        audio,
        strength=1.0,
        bandwidth_extension="off",
        channels="per channel",
        chunk_seconds=5.0,
        mamba_backend="auto",
        device="auto",
        keep_model_loaded=True,
    ):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER Nemotron RE-USE requires an AUDIO input")
        waveform = audio["waveform"].detach().float().cpu()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError("Audio waveform must be [B,C,T]")
        device = _resolve_device(device)
        model, cfg, stft, istft, _native = _load_reuse(device, mamba_backend)
        torch_device = torch.device(device)

        work_rate = sample_rate
        if bandwidth_extension != "off":
            work_rate = int(bandwidth_extension)
        work_rate = min(max(work_rate, 8000), 48000)
        dry = waveform.mean(dim=1, keepdim=True) if channels == "mono" else waveform
        dry = _resample(dry, sample_rate, work_rate)

        stft_cfg = cfg["stft_cfg"]
        base = stft_cfg["sampling_rate"]
        n_fft = _make_even(stft_cfg["n_fft"] * work_rate // base)
        hop = _make_even(stft_cfg["hop_size"] * work_rate // base)
        win = _make_even(stft_cfg["win_size"] * work_rate // base)
        compress = cfg["model_cfg"]["compress_factor"]

        def run(chunk):
            x = chunk.to(torch_device).reshape(1, -1)
            with torch.no_grad():
                mag, pha, _ = stft(
                    x, n_fft=n_fft, hop_size=hop, win_size=win,
                    compress_factor=compress, center=True, addeps=False,
                )  # fmt: skip
                amp, phase, _ = model(mag, pha)
                # NVIDIA's fix for the "strange sweep artifact": silence frames
                # whose decoded magnitude is mostly zero.
                linear = torch.expm1(torch.relu(amp))
                zero_portion = (linear == 0).sum(1) / linear.shape[1]
                amp[:, :, (zero_portion > 0.5)[0]] = 0
                out = istft(amp, phase, n_fft, hop, win, compress)
            return out.float().cpu()

        wet = torch.empty_like(dry)
        try:
            for b in range(dry.shape[0]):
                for c in range(dry.shape[1]):
                    wet[b, c] = _process_in_chunks(
                        dry[b, c], work_rate, float(chunk_seconds),
                        float(chunk_seconds) * 0.5, run,
                    )  # fmt: skip
        finally:
            if not keep_model_loaded:
                _REUSE_CACHE.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        amount = float(strength)
        speech = dry + amount * (wet - dry) if amount < 1.0 else wet
        removed = dry - speech
        return (
            {"waveform": speech.contiguous(), "sample_rate": work_rate},
            {"waveform": removed.contiguous(), "sample_rate": work_rate},
        )


NODE_CLASS_MAPPINGS = {
    "REAPERInput": ReaperInput,
    "REAPERReferenceInput": ReaperReferenceInput,
    "REAPEREQMatch": ReaperEQMatch,
    "REAPERDynamicEQMatch": ReaperDynamicEQMatch,
    "REAPERDynamicVoiceLeveler": ReaperDynamicVoiceLeveler,
    "REAPERClearVoiceEnhance": ReaperClearVoiceEnhance,
    "REAPERClearVoiceSuperResolution": ReaperClearVoiceSuperResolution,
    "REAPERAudioInpaint": ReaperAudioInpaint,
    "REAPERRoomMatchHybrid": ReaperRoomMatchHybrid,
    "REAPERLavaSR": ReaperLavaSR,
    "REAPERVoiceIsolator": ReaperVoiceIsolator,
    "REAPERVideoInput": ReaperVideoInput,
    "REAPEROutput": ReaperOutput,
    "REAPERTextInput": ReaperTextInput,
    "REAPERMidiToScore": ReaperMidiToScore,
    "REAPERScoreOutput": ReaperScoreOutput,
    "REAPERMidiOutput": ReaperMidiOutput,
    "REAPERVideoOutput": ReaperVideoOutput,
    "REAPERTextOutput": ReaperTextOutput,
    "REAPERChordsToABC": ReaperChordsToABC,
    "REAPERChordsOnlyToABC": ReaperChordsOnlyToABC,
    "REAPERNemotronReUSE": ReaperNemotronReUSE,
    "REAPERZonos3TTS": ReaperZonos3TTS,
    "REAPERDBLMSE": ReaperDBLMSE,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "REAPERInput": "REAPER Input",
    "REAPERReferenceInput": "REAPER Reference Input",
    "REAPEREQMatch": "REAPER EQ Match",
    "REAPERDynamicEQMatch": "REAPER Dynamic Voice Match",
    "REAPERDynamicVoiceLeveler": "REAPER Dynamic Voice Leveler",
    "REAPERClearVoiceEnhance": "REAPER ClearVoice Enhance (MossFormer2)",
    "REAPERClearVoiceSuperResolution": "REAPER ClearVoice Super Resolution (48 kHz)",
    "REAPERAudioInpaint": "REAPER Audio Inpaint / De-Click Repair",
    "REAPERRoomMatchHybrid": "REAPER Room Match Hybrid (Lapel + Ambient)",
    "REAPERLavaSR": "REAPER LavaSR v2 (48 kHz Speech Restoration)",
    "REAPERVoiceIsolator": "REAPER Voice Isolator (Inworld)",
    "REAPERVideoInput": "REAPER Video Input",
    "REAPEROutput": "REAPER Output",
    "REAPERTextInput": "REAPER Text Input",
    "REAPERMidiToScore": "REAPER MIDI → YuE2 Score",
    "REAPERScoreOutput": "REAPER Score Output",
    "REAPERMidiOutput": "REAPER MIDI Output",
    "REAPERVideoOutput": "REAPER Video Output",
    "REAPERTextOutput": "REAPER Text Output (SRT / Notes / Markers)",
    "REAPERChordsToABC": "REAPER Chords → ABC",
    "REAPERChordsOnlyToABC": "REAPER Chords → ABC (Chord-only)",
    "REAPERNemotronReUSE": "REAPER Nemotron RE-USE (Universal Speech Enhancement)",
    "REAPERZonos3TTS": "REAPER ZONOS3 TTS (Voice Clone + Emotion/Accent)",
    "REAPERDBLMSE": "REAPER DBLM-SE (Generative Denoise + Dereverb)",
}
