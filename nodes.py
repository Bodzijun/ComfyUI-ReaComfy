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
            def normalize(array, norm=np.inf, axis=0, threshold=None, fill=None, **kwargs):
                values = np.asarray(array)
                if threshold is None: threshold = 1e-10
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
    raise FileNotFoundError("REAPER input request not found in: " + ", ".join(str(path) for path in candidates))


class ReaperInput:
    DESCRIPTION = "Loads the audio file prepared by REAPER. Inputs: optional refresh_token. Outputs: AUDIO and source_path."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"refresh_token": ("STRING", {"default": ""})}}

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
        return {"required": {}, "optional": {"refresh_token": ("STRING", {"default": ""})}}

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
        return {"required": {
            "target_audio": ("AUDIO",),
            "reference_audio": ("AUDIO",),
            "match_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_correction_db": ("FLOAT", {"default": 6.0, "min": 0.5, "max": 18.0, "step": 0.5}),
            "smoothing_bins": ("INT", {"default": 9, "min": 3, "max": 127, "step": 2}),
            "low_frequency_hz": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 20000.0, "step": 10.0}),
            "high_frequency_hz": ("FLOAT", {"default": 16000.0, "min": 1000.0, "max": 24000.0, "step": 100.0}),
            "output_ceiling_db": ("FLOAT", {"default": -1.0, "min": -6.0, "max": -0.1, "step": 0.1}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "match"
    CATEGORY = "REAPER AI/Enhancement"

    def match(self, target_audio, reference_audio, match_strength=0.65, max_correction_db=6.0,
              smoothing_bins=9, low_frequency_hz=80.0, high_frequency_hz=16000.0,
              output_ceiling_db=-1.0):
        if target_audio is None or reference_audio is None:
            raise ValueError("REAPER EQ Match requires target_audio and reference_audio")
        target = target_audio["waveform"].float()
        reference = reference_audio["waveform"].float()
        target_rate = int(target_audio["sample_rate"])
        reference_rate = int(reference_audio["sample_rate"])
        if target.ndim == 2: target = target.unsqueeze(0)
        if reference.ndim == 2: reference = reference.unsqueeze(0)
        batch, channels, target_length = target.shape
        reference = reference.reshape(-1, reference.shape[-1])
        if reference_rate != target_rate:
            try:
                import torchaudio
                reference = torchaudio.functional.resample(reference, reference_rate, target_rate)
            except Exception:
                length = max(1, int(round(reference.shape[-1] * target_rate / reference_rate)))
                reference = torch.nn.functional.interpolate(reference.unsqueeze(1), size=length, mode="linear", align_corners=False).squeeze(1)
        target_flat = target.reshape(-1, target_length)
        n_fft, hop = 2048, 512
        window = torch.hann_window(n_fft, device=target.device, dtype=target.dtype)
        target_spec = torch.stft(target_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        reference_spec = torch.stft(reference.to(target.device), n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        eps = 1e-7
        target_db = 20.0 * torch.log10(target_spec.abs().clamp_min(eps)).mean(dim=2).mean(dim=0)
        reference_db = 20.0 * torch.log10(reference_spec.abs().clamp_min(eps)).mean(dim=2).mean(dim=0)
        correction = reference_db - target_db
        kernel = max(3, int(smoothing_bins) | 1)
        kernel = min(kernel, correction.numel() if correction.numel() % 2 else correction.numel() - 1)
        if kernel >= 3:
            correction = torch.nn.functional.avg_pool1d(correction.view(1, 1, -1), kernel, 1, kernel // 2).view(-1)[:correction.numel()]
        frequencies = torch.linspace(0.0, target_rate / 2.0, correction.numel(), device=target.device)
        mask = (frequencies >= float(low_frequency_hz)) & (frequencies <= min(float(high_frequency_hz), target_rate / 2.0))
        # Match spectral shape, not the reference file's overall loudness.
        # Without removing this broadband offset, max_correction_db can act
        # like a gain control across nearly the entire signal.
        if bool(mask.any()):
            correction = correction - correction[mask].mean()
        correction = correction.mul(float(match_strength)).clamp(-float(max_correction_db), float(max_correction_db))
        correction = torch.where(mask, correction, torch.zeros_like(correction))
        gain = torch.pow(torch.tensor(10.0, device=target.device), correction / 20.0)
        matched = torch.istft(target_spec * gain.view(1, -1, 1), n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, length=target_length)
        matched = self._apply_peak_ceiling(matched, output_ceiling_db)
        return ({"waveform": matched.reshape(batch, channels, target_length), "sample_rate": target_rate},)


class ReaperDynamicEQMatch(ReaperEQMatch):
    DESCRIPTION = "Dynamically stabilizes a voice recording against a reference spectrum. Uses slow STFT correction, voice gating and limited EQ moves to reduce interview mic-distance and angle changes."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "target_audio": ("AUDIO",),
            "reference_audio": ("AUDIO",),
            "match_strength": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_correction_db": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 12.0, "step": 0.5}),
            "smoothing_bins": ("INT", {"default": 21, "min": 3, "max": 127, "step": 2}),
            "temporal_smoothing": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 0.99, "step": 0.01}),
            "voice_gate_db": ("FLOAT", {"default": -35.0, "min": -80.0, "max": -6.0, "step": 1.0}),
            "low_frequency_hz": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 20000.0, "step": 10.0}),
            "high_frequency_hz": ("FLOAT", {"default": 16000.0, "min": 1000.0, "max": 24000.0, "step": 100.0}),
            "output_ceiling_db": ("FLOAT", {"default": -1.0, "min": -6.0, "max": -0.1, "step": 0.1}),
        }}

    def match(self, target_audio, reference_audio, match_strength=0.45, max_correction_db=3.0,
              smoothing_bins=21, temporal_smoothing=0.88, voice_gate_db=-35.0,
              low_frequency_hz=80.0, high_frequency_hz=16000.0,
              output_ceiling_db=-1.0):
        if target_audio is None or reference_audio is None:
            raise ValueError("REAPER Dynamic EQ Match requires target_audio and reference_audio")
        target = target_audio["waveform"].float()
        reference = reference_audio["waveform"].float()
        target_rate = int(target_audio["sample_rate"])
        reference_rate = int(reference_audio["sample_rate"])
        if target.ndim == 2: target = target.unsqueeze(0)
        if reference.ndim == 2: reference = reference.unsqueeze(0)
        batch, channels, target_length = target.shape
        reference = reference.reshape(-1, reference.shape[-1])
        if reference_rate != target_rate:
            try:
                import torchaudio
                reference = torchaudio.functional.resample(reference, reference_rate, target_rate)
            except Exception:
                length = max(1, int(round(reference.shape[-1] * target_rate / reference_rate)))
                reference = torch.nn.functional.interpolate(reference.unsqueeze(1), size=length, mode="linear", align_corners=False).squeeze(1)

        target_flat = target.reshape(-1, target_length)
        target_analysis = target_flat.mean(dim=0, keepdim=True)
        reference_analysis = reference.mean(dim=0, keepdim=True).to(target.device)
        n_fft, hop = 2048, 512
        window = torch.hann_window(n_fft, device=target.device, dtype=target.dtype)
        target_spec = torch.stft(target_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        analysis_spec = torch.stft(target_analysis, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        reference_spec = torch.stft(reference_analysis, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        eps = 1e-7
        target_db = 20.0 * torch.log10(analysis_spec.abs().clamp_min(eps)).squeeze(0)
        reference_db = 20.0 * torch.log10(reference_spec.abs().clamp_min(eps)).squeeze(0)
        reference_energy = 20.0 * torch.log10(reference_spec.abs().pow(2).mean(dim=1).sqrt().clamp_min(eps)).squeeze(0)
        gate = reference_energy.max() + float(voice_gate_db)
        voiced = reference_energy >= gate
        if int(voiced.sum()) == 0: voiced = torch.ones_like(reference_energy, dtype=torch.bool)
        reference_profile = reference_db[:, voiced].mean(dim=1)

        bins = max(3, int(smoothing_bins) | 1)
        bins = min(bins, target_db.shape[0] if target_db.shape[0] % 2 else target_db.shape[0] - 1)
        smoothed_target = target_db
        if bins >= 3:
            smoothed_target = torch.nn.functional.avg_pool1d(
                target_db.transpose(0, 1).unsqueeze(1), bins, 1, bins // 2
            ).squeeze(1).transpose(0, 1)
        correction = reference_profile.unsqueeze(1) - smoothed_target
        frequencies = torch.linspace(0.0, target_rate / 2.0, correction.shape[0], device=target.device)
        mask = (frequencies >= float(low_frequency_hz)) & (frequencies <= min(float(high_frequency_hz), target_rate / 2.0))
        if bool(mask.any()):
            correction = correction - correction[mask].mean(dim=0, keepdim=True)
        alpha = float(temporal_smoothing)
        if alpha > 0.0 and correction.shape[1] > 1:
            filtered = correction.clone()
            for frame in range(1, correction.shape[1]):
                filtered[:, frame] = alpha * filtered[:, frame - 1] + (1.0 - alpha) * correction[:, frame]
            correction = filtered
        correction = correction.mul(float(match_strength)).clamp(-float(max_correction_db), float(max_correction_db))
        correction = torch.where(mask.unsqueeze(1), correction, torch.zeros_like(correction))
        gain = torch.pow(torch.tensor(10.0, device=target.device), correction / 20.0)
        matched = torch.istft(target_spec * gain.unsqueeze(0), n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, length=target_length)
        matched = self._apply_peak_ceiling(matched, output_ceiling_db)
        return ({"waveform": matched.reshape(batch, channels, target_length), "sample_rate": target_rate},)


class ReaperDynamicVoiceLeveler:
    DESCRIPTION = "Offline dialogue leveler. Smoothly stabilizes quiet and loud speech while leaving gated pauses unchanged, then applies a transparent peak safety correction."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "target_level_db": ("FLOAT", {"default": -20.0, "min": -36.0, "max": -10.0, "step": 0.5}),
            "voice_gate_db": ("FLOAT", {"default": -45.0, "min": -80.0, "max": -12.0, "step": 1.0}),
            "max_gain_db": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 18.0, "step": 0.5}),
            "max_cut_db": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 18.0, "step": 0.5}),
            "window_ms": ("FLOAT", {"default": 300.0, "min": 80.0, "max": 1000.0, "step": 10.0}),
            "attack_ms": ("FLOAT", {"default": 120.0, "min": 20.0, "max": 1000.0, "step": 10.0}),
            "release_ms": ("FLOAT", {"default": 650.0, "min": 100.0, "max": 2000.0, "step": 25.0}),
            "ceiling_db": ("FLOAT", {"default": -1.0, "min": -6.0, "max": -0.1, "step": 0.1}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "level"
    CATEGORY = "REAPER AI/Enhancement"

    def level(self, audio, target_level_db=-20.0, voice_gate_db=-45.0, max_gain_db=6.0,
              max_cut_db=6.0, window_ms=300.0, attack_ms=120.0, release_ms=650.0,
              ceiling_db=-1.0):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER Dynamic Voice Leveler requires an AUDIO input")
        waveform = audio["waveform"].float()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2: waveform = waveform.unsqueeze(0)
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
        desired = torch.where(active, float(target_level_db) - level_db, torch.zeros_like(level_db))
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
                alpha_attack = torch.exp(torch.tensor(-hop_seconds / attack_tau, device=waveform.device, dtype=waveform.dtype))
                alpha_release = torch.exp(torch.tensor(-hop_seconds / release_tau, device=waveform.device, dtype=waveform.dtype))
                alpha = torch.where(rising, alpha_attack, alpha_release)
                smoothed[:, index] = alpha * smoothed[:, index - 1] + (1.0 - alpha) * desired[:, index]
        gain_db = torch.nn.functional.interpolate(smoothed.unsqueeze(1), size=length, mode="linear", align_corners=True).squeeze(1)
        gain = torch.pow(torch.tensor(10.0, device=waveform.device, dtype=waveform.dtype), gain_db / 20.0).unsqueeze(1)
        leveled = waveform * gain
        ceiling = torch.pow(torch.tensor(10.0, device=waveform.device, dtype=waveform.dtype), torch.tensor(float(ceiling_db), device=waveform.device, dtype=waveform.dtype) / 20.0)
        peak = leveled.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
        safety_gain = torch.minimum(torch.ones_like(peak), ceiling / peak)
        leveled = leveled * safety_gain
        return ({"waveform": leveled.reshape(batch, channels, length), "sample_rate": sample_rate},)


class ReaperClearVoiceEnhance:
    DESCRIPTION = "Full-band speech enhancement using ClearVoice MossFormer2_SE_48K. Requires the optional clearvoice package; models are downloaded on first use."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "model": (["MossFormer2_SE_48K", "FRCRN_SE_16K", "MossFormerGAN_SE_16K"], {"default": "MossFormer2_SE_48K"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }, "optional": {
            "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
        }}

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
                arr = output.detach().cpu().numpy() if torch.is_tensor(output) else np.asarray(output)
                arr = np.squeeze(arr)
                sf.write(str(result), arr.astype(np.float32), sample_rate)
            enhanced, out_rate = sf.read(str(result), always_2d=True, dtype="float32")
            out = torch.from_numpy(enhanced.T).unsqueeze(0)
            if float(strength) < 1.0:
                dry = waveform
                if out.shape[-1] != dry.shape[-1]:
                    out = torch.nn.functional.interpolate(out, size=dry.shape[-1], mode="linear", align_corners=False)
                reference_np = dry[0].mean(dim=0).numpy()
                enhanced_np = out[0].mean(dim=0).numpy()
                step = 8; ref_ds, cand_ds = reference_np[::step], enhanced_np[::step]
                max_lag = min(int(0.08 * sample_rate / step), max(1, dry.shape[-1] // (4 * step)))
                best_lag, best_score = 0, -1e30
                for lag in range(-max_lag, max_lag + 1):
                    if lag > 0: a, b = ref_ds[lag:], cand_ds[:-lag]
                    elif lag < 0: a, b = ref_ds[:lag], cand_ds[-lag:]
                    else: a, b = ref_ds, cand_ds
                    if a.size < 32 or b.size < 32: continue
                    score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                    if score > best_score: best_lag, best_score = lag, score
                if best_lag > 0:
                    shift = best_lag * step; aligned = np.empty_like(enhanced_np); aligned[:-shift] = enhanced_np[shift:]; aligned[-shift:] = enhanced_np[-1]; enhanced_np = aligned
                elif best_lag < 0:
                    shift = -best_lag * step; aligned = np.empty_like(enhanced_np); aligned[:shift] = enhanced_np[0]; aligned[shift:] = enhanced_np[:-shift]; enhanced_np = aligned
                # Keep the actual dry waveform in the partial blend. The
                # magnitude-only spectral mix can turn model residuals into
                # audible digital noise.
                mixed = band_limited_model_mix(reference_np, enhanced_np, strength, int(out_rate), 0)
                out = torch.from_numpy(mixed.astype("float32")).view(1, 1, -1)
            return ({"waveform": out, "sample_rate": int(out_rate)},)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class ReaperClearVoiceSuperResolution:
    DESCRIPTION = "Restores speech bandwidth with ClearVoice MossFormer2_SR_48K. Best used after denoising; outputs 48 kHz audio."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "preserve_low_hz": ("INT", {"default": 6000, "min": 0, "max": 12000, "step": 250}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "restore"
    CATEGORY = "REAPER AI/Enhancement"

    def restore(self, audio, strength=1.0, preserve_low_hz=6000, artifact_guard=0.5):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER ClearVoice Super Resolution requires an AUDIO input")
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
            raise ValueError("ClearVoice Super Resolution currently supports one audio batch at a time")
        dry = waveform.mean(dim=1)
        if source_rate != 48000:
            import torchaudio
            dry_48 = torchaudio.functional.resample(dry, source_rate, 48000)
        else:
            dry_48 = dry
        enhancer = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
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
                enhanced_np = np.interp(new_axis, old_axis, enhanced_np).astype("float32")
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
                score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                if score > best_score:
                    best_lag, best_score = lag, score
            if best_lag > 0:
                aligned = np.empty_like(enhanced_np); aligned[:-best_lag * step] = enhanced_np[best_lag * step:]; aligned[-best_lag * step:] = enhanced_np[-1]
                enhanced_np = aligned
            elif best_lag < 0:
                advance = -best_lag * step
                aligned = np.empty_like(enhanced_np); aligned[:advance] = enhanced_np[0]; aligned[advance:] = enhanced_np[:-advance]
                enhanced_np = aligned
            enhanced = torch.from_numpy(enhanced_np).view(1, 1, -1)
            # Keep the actual dry waveform in the partial blend. This avoids
            # the synthetic noise introduced by magnitude-only mixing.
            reference_np = dry_48[0].numpy()
            mixed = band_limited_model_mix(reference_np, enhanced_np, strength, 48000, int(preserve_low_hz))
            enhanced = torch.from_numpy(mixed.astype("float32")).view(1, 1, -1)
        return ({"waveform": enhanced, "sample_rate": 48000},)


class ReaperAudioInpaint:
    DESCRIPTION = "CPU audio repair for hard clicks and short mouth/saliva noises. Uses separate impulse and short spectral-burst detectors, protects sustained speech fricatives, and reconstructs only isolated damaged regions from neighboring samples."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "mode": (["auto", "de_click", "inpaint"], {"default": "auto"}),
            "sensitivity": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 20.0, "step": 0.5}),
            "mouth_sensitivity": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 20.0, "step": 0.5}),
            "speech_protect": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_repair_ms": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 15.0, "step": 0.5}),
            "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "repair_info")
    FUNCTION = "repair"
    CATEGORY = "REAPER AI/Restoration"

    def repair(self, audio, mode="auto", sensitivity=8.0, mouth_sensitivity=8.0,
               speech_protect=0.80, max_repair_ms=8.0, strength=0.75):
        if audio is None or "waveform" not in audio:
            raise ValueError("REAPER Audio Inpaint requires an AUDIO input")
        try:
            from scipy.ndimage import (binary_closing, gaussian_filter1d,
                                       maximum_filter1d, median_filter,
                                       uniform_filter1d)
        except ImportError as error:
            raise RuntimeError("REAPER Audio Inpaint requires scipy. Install it with: pip install scipy") from error
        import numpy as np
        waveform = audio["waveform"].detach().float().cpu()
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or waveform.shape[0] != 1:
            raise ValueError("REAPER Audio Inpaint currently supports one audio batch at a time")
        data = waveform[0].numpy().copy()
        channels, length = data.shape
        if length < 16:
            return ({"waveform": waveform, "sample_rate": sample_rate}, json.dumps({"repaired_regions": 0, "mode": mode}))
        mono = data.mean(axis=0)
        # Detector 1: isolated, sub-3 ms waveform discontinuities.  The small
        # median residual is deliberately kept independent of the mouth path.
        med = median_filter(mono, size=9, mode="nearest")
        residual = mono - med
        robust = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
        peak = max(float(np.max(np.abs(mono))), 1e-6)
        mode_factor = 1.50 if mode == "de_click" else (0.90 if mode == "inpaint" else 1.0)
        threshold = max(robust * float(sensitivity) * mode_factor, peak * 0.15 * mode_factor)
        local_rms = np.sqrt(np.maximum(gaussian_filter1d(mono * mono, max(1.0, 0.002 * sample_rate)), 1e-12))
        impulsive_ratio = np.abs(residual) / np.maximum(local_rms, peak * 0.002)
        ratio_limit = 3.0 if mode == "de_click" else (1.6 if mode == "inpaint" else 2.5)
        hard_candidate = (np.abs(residual) > threshold) & (impulsive_ratio > ratio_limit)

        # Detector 2: mouth/saliva clicks are often 2-15 ms noisy bursts rather
        # than single-sample spikes.  Compare short high-band energy and its
        # flux with a 20 ms local baseline.  This is a light-weight spectral
        # novelty proxy and avoids an STFT/ISTFT repair path.
        hp_sigma = max(1.0, 0.00012 * sample_rate)
        high_band = mono - gaussian_filter1d(mono, hp_sigma, mode="nearest")
        short_window = max(3, int(0.0015 * sample_rate))
        long_window = max(short_window + 2, int(0.020 * sample_rate))
        short_hf = np.sqrt(np.maximum(uniform_filter1d(high_band * high_band, short_window, mode="nearest"), 1e-12))
        long_hf = np.sqrt(np.maximum(uniform_filter1d(high_band * high_band, long_window, mode="nearest"), 1e-12))
        hf_flux = np.abs(np.diff(high_band, prepend=high_band[0]))
        short_flux = uniform_filter1d(hf_flux, short_window, mode="nearest")
        long_flux = uniform_filter1d(hf_flux, long_window, mode="nearest")
        energy_ratio = short_hf / np.maximum(long_hf, peak * 0.001)
        flux_ratio = short_flux / np.maximum(long_flux, peak * 0.0002)
        mouth_score = np.sqrt(np.maximum(energy_ratio, 0.0) * np.maximum(flux_ratio, 0.0))
        score_center = float(np.median(mouth_score))
        score_mad = max(float(np.median(np.abs(mouth_score - score_center)) * 1.4826), 0.03)
        mouth_limit = score_center + score_mad * (1.5 + 0.32 * float(mouth_sensitivity))
        mouth_seed = ((mouth_score > mouth_limit)
                      & (energy_ratio > 1.35)
                      & (flux_ratio > 1.20)
                      & (short_hf > peak * 0.0025))
        mouth_expand = max(3, int(0.0015 * sample_rate))
        mouth_candidate = maximum_filter1d(mouth_seed.astype(np.uint8), size=mouth_expand, mode="nearest") > 0
        mouth_candidate = binary_closing(mouth_candidate, structure=np.ones(max(2, int(0.00075 * sample_rate)), dtype=bool))

        candidate = hard_candidate | mouth_candidate
        groups = []
        start = None
        for index, flag in enumerate(candidate):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                groups.append((start, index)); start = None
        if start is not None: groups.append((start, length))
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
            allowed_samples = max_samples if is_mouth else min(max_samples, hard_max_samples)
            if end - begin > allowed_samples or begin < 2 or end >= length - 2:
                continue
            context_start = max(0, begin - speech_context)
            context_end = min(length, end + speech_context)
            left_hf = float(np.mean(hf_active[context_start:begin])) if begin > context_start else 0.0
            right_hf = float(np.mean(hf_active[end:context_end])) if context_end > end else 0.0
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
                u = (span.astype(np.float64) - float(left - 1)) / float(right - left + 1)
                h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
                h10 = u**3 - 2.0 * u**2 + u
                h01 = -2.0 * u**3 + 3.0 * u**2
                h11 = u**3 - u**2
                repaired_values = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
                # Fade the correction to zero at the boundaries so the
                # repair cannot introduce a new discontinuity or LF thump.
                taper = 0.5 - 0.5 * np.cos(2.0 * np.pi * u)
                repaired_values = data[channel, span] + (repaired_values - data[channel, span]) * taper
                if float(strength) >= 1.0:
                    data[channel, span] = repaired_values
                elif float(strength) > 0.0:
                    data[channel, span] += (repaired_values - data[channel, span]) * float(strength)
            repaired += 1
            repaired_hard += int(is_hard)
            repaired_mouth += int(is_mouth)
        input_peak = max(float(np.max(np.abs(waveform[0].numpy()))), 1e-8)
        output_peak = float(np.max(np.abs(data)))
        if output_peak > input_peak:
            data *= input_peak / output_peak
        result = torch.from_numpy(data).unsqueeze(0)
        info = json.dumps({"mode": mode, "sample_rate": sample_rate,
                           "repaired_regions": repaired,
                           "repaired_hard_clicks": repaired_hard,
                           "repaired_mouth_clicks": repaired_mouth,
                           "speech_protected_regions": protected_regions,
                           "sensitivity": float(sensitivity),
                           "mouth_sensitivity": float(mouth_sensitivity),
                           "speech_protect": float(speech_protect),
                           "max_repair_ms": float(max_repair_ms)}, indent=2)
        return ({"waveform": result, "sample_rate": sample_rate}, info)


class ReaperRoomMatchHybrid:
    DESCRIPTION = "Hybrid room-profile estimator requiring two simultaneous recordings: a dry-ish lavalier and an original reverberant ambient microphone. Aligns both recordings, estimates a regularized transfer response, extracts early reflections, synthesizes a stable late tail, and returns an IR plus room metrics. Use denoise or light dereverb only on the lavalier; keep ambient reverberation intact. Optional stereo_ir preserves a practical L/R ambient balance; mono is the default. With one recording, use a dereverb chain or an existing IR instead. This is an acoustic approximation, not a measured swept-sine IR."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lapel_audio": ("AUDIO",),
            "ambient_audio": ("AUDIO",),
            "stereo_ir": ("BOOLEAN", {"default": False}),
            "ir_length_sec": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 5.0, "step": 0.05}),
            "max_alignment_ms": ("FLOAT", {"default": 250.0, "min": 20.0, "max": 1000.0, "step": 10.0}),
            "early_reflections_ms": ("FLOAT", {"default": 120.0, "min": 20.0, "max": 300.0, "step": 5.0}),
            "tail_floor_db": ("FLOAT", {"default": -48.0, "min": -80.0, "max": -20.0, "step": 1.0}),
            "tail_rt60_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            "early_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 12.0, "step": 0.5}),
            "late_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 12.0, "step": 0.5}),
            "ir_high_cut_hz": ("INT", {"default": 12000, "min": 4000, "max": 20000, "step": 500}),
            "apply_ir": ("BOOLEAN", {"default": False}),
            "mix": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio", "room_ir", "profile_json", "rt60_s", "pre_delay_ms", "drr_db")
    FUNCTION = "match"
    CATEGORY = "REAPER AI/Room Acoustics"

    def match(self, lapel_audio, ambient_audio, stereo_ir=False, ir_length_sec=1.5,
              max_alignment_ms=250.0, early_reflections_ms=120.0,
              tail_floor_db=-48.0, tail_rt60_scale=1.0, early_gain_db=0.0,
              late_gain_db=0.0, ir_high_cut_hz=12000,
              apply_ir=False, mix=0.35):
        if not lapel_audio or not ambient_audio or "waveform" not in lapel_audio or "waveform" not in ambient_audio:
            raise ValueError("REAPER Room Match Hybrid requires lapel_audio and ambient_audio")
        import numpy as np
        try:
            from scipy.signal import fftconvolve, butter, sosfilt
            from scipy.ndimage import gaussian_filter1d
        except ImportError as error:
            raise RuntimeError("REAPER Room Match Hybrid requires scipy") from error

        sr = int(lapel_audio["sample_rate"])
        x = lapel_audio["waveform"].detach().float().cpu().mean(dim=1).reshape(-1).numpy()
        ambient_wave = ambient_audio["waveform"].detach().float().cpu()
        ambient_stereo = None
        if bool(stereo_ir) and ambient_wave.ndim >= 3 and ambient_wave.shape[1] >= 2:
            ambient_stereo = ambient_wave[0, :2, :].numpy()
        y = ambient_wave.mean(dim=1).reshape(-1).numpy()
        if int(ambient_audio["sample_rate"]) != sr:
            ay = torch.from_numpy(y).view(1, 1, -1)
            target = max(1, round(y.size * sr / int(ambient_audio["sample_rate"])))
            y = torch.nn.functional.interpolate(ay, size=target, mode="linear", align_corners=False).reshape(-1).numpy()
            if ambient_stereo is not None:
                ast = torch.from_numpy(ambient_stereo).unsqueeze(0)
                ambient_stereo = torch.nn.functional.interpolate(ast, size=target, mode="linear", align_corners=False)[0].numpy()
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        x -= x.mean(); y -= y.mean()
        x /= max(np.sqrt(np.mean(x * x)), 1e-8)
        y /= max(np.sqrt(np.mean(y * y)), 1e-8)
        fft_n = 1 << max(12, (2 * n - 1).bit_length())
        corr = np.fft.irfft(np.fft.rfft(y, fft_n) * np.conj(np.fft.rfft(x, fft_n)), fft_n)
        corr = np.concatenate((corr[-(n - 1):], corr[:n]))
        lags = np.arange(-(n - 1), n)
        limit = min(n - 1, int(sr * float(max_alignment_ms) / 1000.0))
        valid = np.abs(lags) <= limit
        lag = int(lags[valid][np.argmax(np.abs(corr[valid]))])
        if lag > 0:
            y, x = y[lag:], x[:-lag]
        elif lag < 0:
            x, y = x[-lag:], y[:lag]
        n = min(x.size, y.size); x, y = x[:n], y[:n]

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
            xs, ys = x[start:start + frame], y[start:start + frame]
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
            raise ValueError("Room Match could not find usable speech frames in the two recordings")
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
        peak = int(np.argmax(np.abs(response[:frame // 2])))
        response = np.roll(response, -peak)
        ir_len = min(frame, max(64, int(float(ir_length_sec) * sr)))
        ir = response[:ir_len].copy()
        # A speech-derived transfer estimate contains circular FFT residue and
        # diffuse estimation noise. Keep measured early reflections, then force
        # a monotonic late decay and gate the residual noise floor.
        early_len = min(ir_len, max(1, int(float(early_reflections_ms) * sr / 1000.0)))
        peak_level = max(float(np.max(np.abs(ir[:max(1, early_len)]))), 1e-8)
        if ir_len > early_len:
            late = ir[early_len:]
            late_t = np.arange(late.size, dtype=np.float64) / sr
            # Fit the raw Schroeder decay before shaping, then use that
            # measured RT60 to form the synthetic late tail. The fallback is
            # only used when speech does not provide a stable decay slope.
            raw_energy = np.cumsum((late[::-1] ** 2))[::-1]
            raw_db = 10.0 * np.log10(np.maximum(raw_energy / max(raw_energy[0], 1e-12), 1e-12))
            raw_fit = (raw_db < -5.0) & (raw_db > -35.0)
            raw_rt60 = 0.0
            if np.count_nonzero(raw_fit) > 8:
                raw_slope = np.polyfit(late_t[raw_fit], raw_db[raw_fit], 1)[0]
                if raw_slope < 0:
                    raw_rt60 = float(max(0.15, min(8.0, -60.0 / raw_slope)))
            tau = (raw_rt60 * float(tail_rt60_scale)) / np.log(1000.0) if raw_rt60 > 0 else 0.18 * float(tail_rt60_scale)
            late *= np.exp(-late_t / max(tau, 0.03))
            ir[early_len:] = late
        sos = butter(2, min(cutoff * 1.5, 0.48 * sr), btype="lowpass", fs=sr, output="sos")
        ir = sosfilt(sos, ir)
        # Use a smoothed RMS gate instead of zeroing individual samples. A
        # sample-wise gate creates the crunchy/dither-like fade heard in IRs.
        floor = peak_level * (10.0 ** (float(tail_floor_db) / 20.0))
        rms = np.sqrt(np.maximum(gaussian_filter1d(ir * ir, max(1.0, 0.010 * sr)), 1e-12))
        gate_pos = np.clip((rms - floor * 0.35) / max(floor * 1.15, 1e-8), 0.0, 1.0)
        gate = gate_pos * gate_pos * (3.0 - 2.0 * gate_pos)
        ir *= gate
        # Room tails should not regain energy after they have decayed. Apply a
        # monotonic RMS envelope while retaining the fine reflection texture.
        if ir_len > early_len:
            # Do not anchor the tail to its first sample: a near-zero sample
            # at the early/late boundary would otherwise mute the whole tail.
            tail_rms = np.sqrt(np.maximum(gaussian_filter1d(ir[early_len:] ** 2, max(1.0, 0.012 * sr)), 1e-12))
            anchor_len = min(tail_rms.size, max(1, int(0.020 * sr)))
            anchor = max(float(np.max(tail_rms[:anchor_len])), 1e-8)
            tail_tau = (raw_rt60 * float(tail_rt60_scale)) / np.log(1000.0) if raw_rt60 > 0 else 0.18 * float(tail_rt60_scale)
            envelope = anchor * np.exp(-np.arange(tail_rms.size, dtype=np.float64) / max(tail_tau * sr, 1.0))
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

        ir_channels = np.stack([ir, ir], axis=0) if ambient_stereo is not None else ir[None, :]
        if ambient_stereo is not None:
            ch_rms = np.sqrt(np.mean(ambient_stereo[:, :n] ** 2, axis=1) + 1e-12)
            ch_rms /= max(float(np.mean(ch_rms)), 1e-8)
            ir_channels *= ch_rms[:, None]
            ir_channels /= max(float(np.max(np.abs(ir_channels))), 1e-8)
        source = torch.from_numpy(x.astype("float32")).view(1, 1, -1)
        ir_audio = torch.from_numpy(ir_channels.astype("float32")).view(1, ir_channels.shape[0], -1)
        output = source
        if apply_ir:
            wet_channels = np.stack([fftconvolve(x, ir_channels[ch], mode="full")[:x.size] for ch in range(ir_channels.shape[0])], axis=0)
            wet_channels /= max(float(np.max(np.abs(wet_channels))), 1e-8)
            if ir_channels.shape[0] == 2:
                dry_channels = np.repeat(x[None, :], 2, axis=0)
                output = torch.from_numpy(((1.0 - float(mix)) * dry_channels + float(mix) * wet_channels).astype("float32")).view(1, 2, -1)
            else:
                output = torch.from_numpy(((1.0 - float(mix)) * x + float(mix) * wet_channels[0]).astype("float32")).view(1, 1, -1)
        profile = json.dumps({"sample_rate": sr, "alignment_lag_ms": 1000.0 * lag / sr,
                              "pre_delay_ms": pre_delay_ms, "rt60_s": rt60, "drr_db": drr_db,
                              "ir_length_sec": float(ir_length_sec), "method": "hybrid_transfer_plus_decay"}, indent=2)
        return ({"waveform": output, "sample_rate": sr}, {"waveform": ir_audio, "sample_rate": sr}, profile, rt60, pre_delay_ms, drr_db)


class ReaperLavaSR:
    DESCRIPTION = "LavaSR v2 speech bandwidth extension. The model internally analyzes audio at 16 kHz, discards the original band above about 8 kHz, synthesizes a new high band, and returns 48 kHz audio. strength blends the result with dry audio; cutoff_hz controls the model's low/high reconstruction crossover. This is synthesis, not recovery of the original high-frequency samples."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "denoise": ("BOOLEAN", {"default": False}),
            "batch": ("BOOLEAN", {"default": False}),
            "cutoff_hz": ("INT", {"default": 0, "min": 0, "max": 20000, "step": 100}),
        }}

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
            raise ValueError("REAPER LavaSR currently supports one audio batch at a time")
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
            input_audio, _ = model.load_audio(str(source), input_sr=16000, cutoff=cutoff)
            restored = model.enhance(input_audio, enhance=True, denoise=bool(denoise), batch=bool(batch))
            restored = restored.detach().float().cpu().reshape(1, 1, -1)
            dry = waveform.mean(dim=1, keepdim=True)
            if sample_rate != 48000:
                dry = torch.nn.functional.interpolate(dry, size=max(1, round(dry.shape[-1] * 48000 / sample_rate)), mode="linear", align_corners=False)
            length = min(dry.shape[-1], restored.shape[-1])
            dry, restored = dry[..., :length], restored[..., :length]
            output = dry + (restored - dry) * float(strength)
            return ({"waveform": output, "sample_rate": 48000},)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


_VOICE_ISOLATOR_CACHE = {}
_VOICE_ISOLATOR_FILES = {"offline": "denoiser.safetensors", "streaming": "denoiser_realtime.safetensors"}


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
        return {"required": {
            "audio": ("AUDIO",),
            "mode": (["offline", "streaming"], {"default": "offline",
                     "tooltip": "offline: best quality, sees 8 s chunks. streaming: causal realtime model."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "1.0 = fully isolated voice; lower values blend the dry signal back in."}),
            "channels": (["per channel", "mono"], {"default": "per channel",
                         "tooltip": "per channel keeps stereo/multichannel layout; mono mixes down first (faster)."}),
        }, "optional": {
            "lookahead_ms": (["20", "0", "40"], {"default": "20",
                             "tooltip": "Streaming mode only: trained look-ahead of the realtime model."}),
            "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("voice", "background")
    FUNCTION = "isolate"
    CATEGORY = "REAPER AI/Enhancement"

    def isolate(self, audio, mode="offline", strength=1.0, channels="per channel",
                lookahead_ms="20", device="auto", keep_model_loaded=True):
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
                check_interrupt = model_management.throw_exception_if_processing_interrupted
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
        return {"required": {}, "optional": {"refresh_token": ("STRING", {"default": ""}), "max_duration_s": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0}), "limit_duration": ("BOOLEAN", {"default": False})}}

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
            raise RuntimeError("This ComfyUI version does not provide the native VIDEO type") from error
        try:
            from woosh.utils.videoio import extract_video_frames
            end_time = float(max_duration_s) if limit_duration else None
            frames, rate, _pts = extract_video_frames(str(path), start_time=0, end_time=end_time)
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
                "output_sample_rate": (["44100", "48000", "96000"], {"default": "48000"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "REAPER AI/Outputs"

    def save(self, audio, label="ComfyUI result", resample_output=False, output_sample_rate="48000"):
        cache = cache_dir()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = cache / f"comfy_to_reaper_{stamp}_{int(time.time() * 1000) % 1000:03d}.wav"
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
                waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
            except Exception:
                length = max(1, int(round(waveform.shape[-1] * target_rate / source_rate)))
                waveform = torch.nn.functional.interpolate(waveform.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0)
        waveform = waveform.cpu()
        samples = waveform.numpy().T
        sf.write(str(output), samples, target_rate, subtype="PCM_24")
        manifest = cache / "comfy_to_reaper.json"
        outputs = []
        request_job_id = None
        request = cache / "reaper_to_comfy.json"
        if request.is_file():
            try:
                request_job_id = json.loads(request.read_text(encoding="utf-8")).get("job_id")
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
        payload = {"status": "complete", "job_id": request_job_id, "outputs": outputs, "path": str(output), "label": label}
        if position is not None: payload["position"] = position
        if length is not None: payload["length"] = length
        payload = json.dumps(payload, ensure_ascii=False, indent=2)
        manifest.write_text(payload, encoding="utf-8")
        # Keep a user-level fallback so REAPER can import results even when
        # the project was unsaved before the send action created its pointer.
        fallback = Path.home() / "REAPER_AI_CACHE"
        fallback.mkdir(parents=True, exist_ok=True)
        fallback_manifest = fallback / "comfy_to_reaper.json"
        if fallback_manifest.resolve() != manifest.resolve():
            fallback_manifest.write_text(payload, encoding="utf-8")
        return {"ui": {"text": [f"REAPER output ready: {output}"]}, "result": (str(manifest),)}


class ReaperTextInput:
    DESCRIPTION = "Reads text from the current REAPER manifest. Inputs: field (text, lyrics, chords or arrangement), optional prefix_text and refresh_token. Output: STRING text."
    """Read lyrics/text prepared by a REAPER action as a STRING output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {
            "field": (["text", "lyrics", "chords", "arrangement"], {"default": "text"}),
            "prefix_text": ("STRING", {"multiline": True, "default": ""}),
            "refresh_token": ("STRING", {"default": ""}),
        }}

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
        signature = [str(refresh_token or ""), str(field or "text"), str(prefix_text or "")]
        for path in candidates:
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    signature.extend([str(data.get("job_id", "")), str(path.stat().st_mtime_ns)])
                    break
                except Exception:
                    signature.append(str(path.stat().st_mtime_ns))
        return "|".join(signature)

    def load(self, field="text", prefix_text="", refresh_token=""):
        data = request_data()
        field_map = {"lyrics": "lyrics_text", "chords": "chords_text", "arrangement": "arrangement_text"}
        selected_key = field_map.get(field, "text")
        text = str(data.get(selected_key, "") or "")
        if not text.strip():
            source = str(data.get("source", ""))
            if field == "chords" and source == "chords": text = str(data.get("chords", data.get("text", "")) or "")
            elif field == "lyrics" and source == "regions": text = str(data.get("text", "") or "")
            elif field == "arrangement" and source == "arrangement": text = str(data.get("text", "") or "")
        text_path = data.get("lyrics_text_path") if field == "lyrics" else data.get("text_path")
        if text_path:
            path = Path(str(text_path)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"REAPER text file not found: {path}")
            text = path.read_text(encoding="utf-8-sig")
        prefix_text = str(prefix_text or "").strip()
        if prefix_text and text.strip(): text = prefix_text + "\n\n" + text
        elif prefix_text: text = prefix_text
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

    def load(self, mode, vocal_track, instrument_track, score_abc="", without_sections=False, instrument_tracks="", refresh_token=""):
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
            raise RuntimeError("YuE2-ComfyUI must be installed for REAPER MIDI to Score") from error
        if str(instrument_tracks or "").strip():
            instrument_track = str(instrument_tracks).strip()
        input_dir = Path(get_input_directory())
        input_dir.mkdir(parents=True, exist_ok=True)
        temp_name = f"reaper_{int(time.time() * 1000)}_{source.name}"
        temp_path = input_dir / temp_name
        shutil.copyfile(source, temp_path)
        try:
            result = YuE2LoadMidi().load(temp_name, mode, vocal_track, instrument_track,
                                         score_abc=score_abc,
                                         lyrics=lyrics_hint,
                                         without_sections=without_sections)
            score_result = result["result"][0]
            score_result = score_result.replace(r'\_', '_').replace(r'\^', '^').replace(r'\=', '=')
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
            "optional": {"lyrics": ("STRING", {"multiline": True, "default": ""}),
                         "label": ("STRING", {"default": "YuE2 score"}),
                         "export_midi": ("BOOLEAN", {"default": True})},
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
                    raise ValueError("score_abc input is not YuE2 ABC (expected an X:1 header); connect YuE2 Generate Song's score_abc output")
                ensure_yue2_path()
                from yue2_comfy.midi.export import midi_of
                midi_path = cache / f"reacomfy_score_{stamp}.mid"
                midi_path.write_bytes(midi_of(str(score_abc)))
            except Exception as error:
                midi_error = str(error)
        manifest = cache / "reacomfy_score.json"
        payload = {"status": "complete", "label": label, "abc_path": str(abc_path),
                   "midi_path": str(midi_path) if midi_path else None,
                   "lyrics_path": str(lyrics_path) if lyrics_path else None}
        if midi_error:
            payload["midi_error"] = midi_error
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        message = f"REAPER score ready: {abc_path}"
        if midi_path:
            message += f"; MIDI: {midi_path}"
        elif midi_error:
            message += f"; MIDI export failed: {midi_error}"
        return {"ui": {"text": [message]},
                "result": (str(abc_path), str(midi_path) if midi_path else "", str(manifest))}


class ReaperMidiOutput:
    DESCRIPTION = "Returns MIDI from any ComfyUI workflow to REAPER. Inputs: midi_path, optional notes_json and label. Output: manifest_path."
    """Copy a MIDI file produced by any ComfyUI workflow into the REAPER cache."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"midi_path": ("STRING", {"default": ""})},
            "optional": {"notes_json": ("STRING", {"default": ""}),
                         "label": ("STRING", {"default": "ComfyUI MIDI"})},
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
        output = cache / f"reacomfy_midi_{stamp}_{int(time.time() * 1000) % 1000:03d}{source.suffix.lower()}"
        shutil.copy2(str(source), str(output))
        notes_output = None
        notes_source = Path(str(notes_json or "")).expanduser()
        if notes_source.is_file():
            notes_output = cache / f"reacomfy_notes_{stamp}.json"
            shutil.copy2(str(notes_source), str(notes_output))
        manifest = cache / "reacomfy_midi.json"
        payload = {"status": "complete", "label": label, "midi_path": str(output),
                   "notes_json": str(notes_output) if notes_output else None}
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ui": {"text": [f"REAPER MIDI ready: {output}"]}, "result": (str(manifest),)}


class ReaperVideoOutput:
    DESCRIPTION = "Copies a video file produced by any ComfyUI workflow into the REAPER project cache. Input: video_path and optional label. Output: manifest_path."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING", {"default": ""})},
                "optional": {"label": ("STRING", {"default": "ComfyUI Video"})}}

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
        output = cache / f"reacomfy_video_{stamp}_{int(time.time() * 1000) % 1000:03d}{source.suffix.lower()}"
        shutil.copy2(str(source), str(output))
        manifest = cache / "reacomfy_video.json"
        payload = {"status": "complete", "label": label, "video_path": str(output)}
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ui": {"text": [f"REAPER video ready: {output}"]}, "result": (str(manifest),)}


class ReaperTextOutput:
    DESCRIPTION = "Exports subtitles or plain text from ComfyUI for REAPER. Accepts SRT/VTT timestamps or plain lines and imports them as note items, project markers, or regions."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "default": ""})},
                "optional": {"mode": (["items", "regions", "markers"], {"default": "items"}),
                             "track_name": ("STRING", {"default": "TTS Subtitles"}),
                             "label": ("STRING", {"default": "ComfyUI Text"}),
                             "start_offset": ("FLOAT", {"default": 0.0, "min": -86400.0, "max": 86400.0}),
                             "default_duration": ("FLOAT", {"default": 4.0, "min": 0.05, "max": 300.0})}}
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
            if len(parts) == 3: return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            if len(parts) == 2: return float(parts[0]) * 60.0 + float(parts[1])
            return float(value)
        except ValueError: return None

    @classmethod
    def _segments(cls, raw, offset, default_duration):
        source = str(raw or "")
        candidate = Path(source.strip()).expanduser()
        if candidate.is_file(): source = candidate.read_text(encoding="utf-8-sig", errors="replace")
        lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments, index = [], 0
        while index < len(lines):
            line = lines[index].strip()
            if not line or line.upper() == "WEBVTT": index += 1; continue
            timing = line
            if "-->" not in timing and index + 1 < len(lines) and "-->" in lines[index + 1]: index += 1; timing = lines[index].strip()
            if "-->" in timing:
                left, right = [part.strip().split()[0] for part in timing.split("-->", 1)]
                start, end = cls._timecode(left), cls._timecode(right)
                if start is not None and end is not None:
                    index += 1; body = []
                    while index < len(lines) and lines[index].strip(): body.append(lines[index].strip()); index += 1
                    text = "\n".join(body).strip()
                    if text: segments.append((max(0.0, start + offset), max(start + offset + 0.05, end + offset), text))
                    continue
            index += 1
        if segments: return segments
        plain = [line.strip() for line in lines if line.strip()]
        return [(offset + i * default_duration, offset + (i + 1) * default_duration, line) for i, line in enumerate(plain)]

    def save(self, text, mode="items", track_name="TTS Subtitles", label="ComfyUI Text", start_offset=0.0, default_duration=4.0):
        cache, stamp = cache_dir(), time.strftime("%Y%m%d_%H%M%S")
        segments = self._segments(text, float(start_offset), max(0.05, float(default_duration)))
        text_path = cache / f"reacomfy_text_{stamp}.txt"; text_path.write_text(str(text or ""), encoding="utf-8")
        segments_path = cache / f"reacomfy_text_{stamp}.tsv"
        with segments_path.open("w", encoding="utf-8", newline="\n") as handle:
            for start, end, value in segments: handle.write(f"{start:.6f}\t{end:.6f}\t{json.dumps(value, ensure_ascii=False)[1:-1]}\n")
        manifest = cache / "reacomfy_text.json"
        manifest.write_text(json.dumps({"status": "complete", "mode": str(mode), "track_name": str(track_name), "label": str(label), "text_path": str(text_path), "segments_path": str(segments_path), "segment_count": len(segments)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ui": {"text": [f"REAPER text ready: {len(segments)} segment(s) | {manifest}"]}, "result": (str(manifest), str(text or ""))}


class ReaperChordsToABC:
    DESCRIPTION = "Merges chord text into an existing YuE2 ABC score. Inputs: score_abc and chords. Output: score_abc STRING."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"score_abc": ("STRING", {"multiline": True}), "chords": ("STRING", {"multiline": True})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("score_abc",)
    FUNCTION = "merge"
    CATEGORY = "REAPER AI/YuE2"
    def merge(self, score_abc, chords):
        names = []
        for line in str(chords or "").splitlines():
            value = line.split("|", 1)[-1].strip() if line.strip() else ""
            for token in value.replace(",", " ").split():
                token = token.strip('[](){}<>;:')
                if token and __import__('re').match(r"^[A-Ga-g](?:#|b)?(?:m|min|maj|dim|aug|sus|add)?\d*(?:/[A-Ga-g](?:#|b)?)?$", token):
                    names.append(token)
        output, active, index = [], False, 0
        for line in str(score_abc).splitlines():
            if line.startswith("V:"): active = not active
            if active and index < len(names) and "|" in line:
                parts = line.split("|")
                rebuilt = []
                for part_index, part in enumerate(parts):
                    clean_part = __import__('re').sub(r'"[^"]*"', '', part)
                    if index < len(names) and clean_part.strip():
                        clean_part = '"' + names[index].replace('"', '') + '"' + clean_part
                        index += 1
                    rebuilt.append(clean_part)
                    if part_index < len(parts) - 1: rebuilt.append("|")
                line = "".join(rebuilt)
            line = line.replace(r'\_', '_').replace(r'\^', '^').replace(r'\=', '=')
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
        if token.endswith("M"): token = token[:-1]
        if re.fullmatch(r"[A-Ga-g](?:##|bb|#|b)?(?:m|dim|aug|sus2|sus4|add\d+|7|6|9|11|13)?(?:/[A-Ga-g](?:#|b)?)?", token):
            return token[0].upper() + token[1:]
        return None

    def build(self, chords, bpm=92, key="C"):
        import re
        sections = []
        current_name, current = None, []
        for raw in str(chords or "").splitlines():
            line = raw.strip()
            if not line: continue
            section = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
            if section:
                if current_name and current: sections.append((current_name, current))
                current_name, current = section.group(1).strip(), []
                line = section.group(2).strip()
            for value in re.split(r"[\s,|]+", line):
                chord = self._chord(value)
                if chord: current.append(chord)
        if current_name and current: sections.append((current_name, current))
        if not sections:
            raise ValueError("No chord symbols found. Use sectioned text such as [Verse] Fm Bbm Cm.")
        key = str(key or "C").strip() or "C"
        header = ["X:1", "T:", "M:4/4", "L:1/16", f"Q:1/4={int(bpm)}",
                  'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
                  'V: Ins clef=treble name="Ins Melody" snm="Inst."', f"K:{key}"]
        output = list(header)
        for name, values in sections:
            output.append("% " + name.lower())
            for offset in range(0, len(values), 4):
                group = values[offset:offset + 4]
                output.append("V: Vocal")
                output.append("|".join('"' + chord + '"z16' for chord in group) + "|")
                output.append("V: Ins")
                output.append("|".join("Z" for _ in group) + "|")
        return ("\n".join(output),)


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
}
