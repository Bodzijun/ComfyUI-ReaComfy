# ComfyUI-ReaComfy — REAPER Bridge nodes (ReaComfy 0.6.0)

ComfyUI custom nodes that exchange audio, MIDI, text and video with the active
REAPER project. They are the ComfyUI half of
[ReaComfy](#reaper-side): the REAPER scripts send material to ComfyUI and import
the results back to the timeline. Works on Windows, macOS and Linux.

Discussion, updates and support: [ReaComfy thread on the Cockos REAPER forum](https://forum.cockos.com/showthread.php?t=311369).

## Installation

**ComfyUI-Manager / Comfy Registry:** search for `ReaComfy` and install.

**Manually:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Bodzijun/ComfyUI-ReaComfy
python -m pip install -r ComfyUI-ReaComfy/requirements.txt
```

Use the Python that runs ComfyUI (for the portable Windows build:
`python_embeded\python.exe -m pip ...`; for a venv, activate it first). Restart
ComfyUI afterwards.

## Exchange folder

The nodes read and write a small file protocol in an exchange folder:

1. `REAPER_AI_CACHE` environment variable, when set;
2. otherwise `~/REAPER_AI_CACHE` (`%USERPROFILE%\REAPER_AI_CACHE` on Windows,
   `$HOME/REAPER_AI_CACHE` on macOS/Linux). When it contains
   `current_project.txt`, the folder named there (the current REAPER project's
   cache) is used.

## Optional modules and model installation

Nothing has to be installed by hand. Nodes that need extra Python modules
(ClearVoice, LavaSR, RE-USE) install them into ComfyUI's own Python the first
time you run them, in any ComfyUI install (portable, desktop or venv); ZONOS3
builds its own private environment. Model weights download on first use. The
first run of such a node therefore takes a few minutes; watch the ComfyUI
console for `[ReaComfy] Installing ...` lines. Your ComfyUI's torch and numpy
are never upgraded or downgraded.

### Inworld Voice Isolator

`REAPER Voice Isolator (Inworld)` separates the foreground voice from noise,
music and room background using
[inworld/voice-isolator](https://huggingface.co/inworld/voice-isolator)
(code and weights Apache-2.0). It is installed with `requirements.txt`
(`inworld-voice-isolation`); the weights (~256 MB offline, ~233 MB realtime)
download from Hugging Face on first use and are checksum-verified.
For offline machines, put `denoiser.safetensors` and/or
`denoiser_realtime.safetensors` into `ComfyUI/models/voice_isolator/`.

- `mode` — `offline` (best quality, 8 s chunks with 2 s overlap) or `streaming`
  (the causal realtime model; same result as live processing).
- `strength` — 1.0 returns only the voice; lower values blend the dry signal back.
- `channels` — `per channel` keeps the stereo/multichannel layout; `mono` mixes
  down first and is faster.
- `lookahead_ms` — streaming only: 0, 20 or 40 ms.
- `device` — `auto` picks CUDA, then Apple `mps` (PyTorch 2.6+), then CPU.
- Outputs: `voice` and `background`; `voice + background` equals the input, so
  the background can be rebalanced under the cleaned voice in REAPER.

The model runs at 48 kHz internally; other rates are resampled in and out, and
the outputs keep the input rate and length. Version 1 removes noise only; it
does not restore missing bandwidth (chain `REAPER LavaSR v2` or
`REAPER ClearVoice Super Resolution` after it for that).

### ClearVoice / MossFormer2

Used by `REAPER ClearVoice Enhance` and `REAPER ClearVoice Super Resolution`.
ClearVoice installs itself on first use (without its pinned numpy/torch
versions, so ComfyUI's own packages are left alone).

The `MossFormer2_SE_48K` and `MossFormer2_SR_48K` checkpoints are downloaded
automatically on first use. The enhancement node is intended for noisy speech;
the super-resolution node is best used after enhancement and always returns
48 kHz audio. Project: https://github.com/modelscope/ClearerVoice-Studio

### LavaSR v2

`REAPER LavaSR v2` is the recommended 48 kHz bandwidth-extension node for
speech that was cloned, compressed, or processed at a lower rate. It downloads
the LavaSR v2 checkpoint from Hugging Face on first use and returns 48 kHz
audio. LavaSR and its vocos fork install themselves on first use (from GitHub
archives, no git needed; Torch is never reinstalled).

Use `denoise` only when the input is noisy; for a cleaner chain, denoise first
with a dedicated 48 kHz enhancer and then run LavaSR. `strength` blends the
reconstructed high band with the dry signal. `cutoff_hz = 0` uses the model's
automatic cutoff. Important: the current LavaSR v2 API internally resamples
the working signal to 16 kHz (roughly an 8 kHz source bandwidth), then
synthesizes a new upper band and outputs 48 kHz. It does not preserve the
original high-frequency samples. Use `MossFormer2_SR_48K` instead when the
source already contains valuable full-band information. Project:
https://github.com/ysharma3501/LavaSR

### NVIDIA Nemotron RE-USE

`REAPER Nemotron RE-USE (Universal Speech Enhancement)` uses
[nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE)
(OpenMDW-1.1, "ready for commercial use"). One small model (9.6M parameters)
removes noise and reverb and repairs clipping, codec and packet-loss damage at
the source rate (8–48 kHz) while keeping the main speaker. The research
variants `nvidia/RE-USE` and `nvidia/Real-time_RE-USE` are non-commercial and
are not used here.

Everything is automatic: the code and weights (~40 MB) download on first use
into `ComfyUI/models/nemotron_reuse/`, and missing Python modules (`einops`,
`pyyaml`, `safetensors`, `huggingface_hub`) are installed into ComfyUI's own
Python. `mamba-ssm` (Linux + CUDA only) is used when it is already installed.

Without `mamba-ssm` (Windows, macOS, CPU) the node uses a built-in PyTorch
version of the Mamba scan: same weights and maths, slower. On an NVIDIA GPU
without `mamba-ssm` (e.g. Windows) expect several seconds per 5 s chunk; on
CPU it takes minutes per chunk, so use it only for short clips.
`mamba_backend` forces either path.

- `strength` — 1.0 = fully enhanced; lower values blend the dry signal back in.
- `bandwidth_extension` — resample to 32/44.1/48 kHz first so the model
  regenerates the missing highs of phone/Zoom/16 kHz recordings.
- `chunk_seconds` — NVIDIA's long-file recipe (5 s, 50 % overlap).
- Outputs `speech` and `removed` (speech + removed = input).

### ZONOS3-Mini TTS

`REAPER ZONOS3 TTS (Voice Clone + Emotion/Accent)` speaks text with
[Zyphra ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) (code MIT,
weights Apache-2.0, 43 languages including Ukrainian and Russian). Connect a
`reference_audio` (5–30 s of clean speech, for example the selected REAPER
item) to clone that voice, or give a `speaker_id` from the server's default
voices.

Controls (Zyphra's studio steering bundle): emotion (angry, happy, sad, calm,
scared, confused) with a strength where 1.0 is Zyphra's recommended dose; pace;
projection (whisper ↔ loud); mood valence and arousal; read ↔ chatty; flat ↔
lively; hesitation; accent (British, American, Scottish, Irish, Welsh,
Australian, Indian English, Hindi, Spanish, French, Portuguese, German,
Vietnamese, Polish, Mandarin) with strength; gender, age, pitch, intonation,
room (phone ↔ studio); texture (deep, bright, husky, raspy, soft, authority);
character presets; and `extra_controls` for raw bundle doses as JSON. Each line
of the text is generated separately (`split`) and joined with `gap_ms` of
silence. Output is 44.1 kHz mono; `info` returns the seeds used.

Nothing to install or launch: with `backend = automatic` (default) the first
run builds a private ZONOS3 environment in `ComfyUI/models/zonos3/` (its own
Python 3.12 and PyTorch via `uv`, the ZONOS3 code and ~9 GB of weights; 10–30
minutes, progress in `setup.log`). ZONOS3 pins its own Python/PyTorch versions,
so it never touches ComfyUI's environment. After that the model stays loaded in
a background worker and each line takes seconds. Reference audio is decoded
without FFmpeg.

Advanced alternatives: `server` talks to a ZONOS3 server you run yourself
(default `http://127.0.0.1:1919`); `reference (venv)` runs `zonos3.reference`
from your own checkout (`zonos3_home` / `zonos3_python` or `ZONOS3_HOME` /
`ZONOS3_PYTHON`).

### ZONOS3 Speech Edit (fix a slip of the tongue)

`REAPER ZONOS3 Speech Edit (Fix a Slip of the Tongue)` regenerates only the
changed words of a recorded clip in the speaker's own voice with ZONOS3-Mini's
speech-edit adapter; every other codec frame of the recording is kept exactly.
Give it the selected REAPER item (keep it short: one or two sentences), the
`corrected_text` (what the whole clip should say) and either:

- `original_text`: what the clip says now, slip included. FuseAlign finds the
  words that differ and only those are regenerated; or
- `edit_start` / `edit_end`: the part to regenerate, in seconds from the clip
  start (useful when the aligner misplaces a word).

Only the regenerated words are new: by default the original recording is kept
everywhere else (`keep_original_audio`), crossfaded and level-matched, because
ZONOS3 itself re-decodes the whole clip through its codec.

`seed` 0 is random; change it for another take. The result is the whole clip
with the fix, at the item's sample rate (`match_input_rate`). Clips longer than
16 s are edited in windows around the change and need `original_text`.

Everything installs itself on the first edit, inside the private ZONOS3
environment: FuseAlign (Zyphra, Apache-2.0, ~1.3 GB checkpoint) and the
speech-edit LoRA from `Zyphra/ZONOS3-Mini`. Needs an NVIDIA GPU.

### DeepFilterNet / Egregora

`REAPER ClearVoice` is independent from Egregora's DeepFilterNet nodes. If
Egregora is installed through ComfyUI-Manager, use its own dependency
installer. Current ReaComfy installations include a compatibility shim for
modern `torchaudio`; downgrading ComfyUI is not required. If DeepFilterNet is
reinstalled, the shim may need to be reapplied.

### YuE2

YuE2 MIDI and score nodes require the separate `YuE2-ComfyUI` custom node
pack. Install and update it through ComfyUI-Manager, then restart ComfyUI so
`REAPER MIDI → YuE2 Score` can import the YuE2 parser.

`REAPER Input` reads `reaper_to_comfy.json` and returns an AUDIO object. It invalidates its ComfyUI cache when the REAPER manifest changes, so a new source file is processed on the next queue. `REAPER Text Input` can select the generic text, lyrics, chords, or arrangement field from a combined YuE2 manifest.
`REAPER Reference Input` reads the separate `reference_path` written by ReaComfy's **Send reference audio** action and returns it as AUDIO for reference-based processing.
`REAPER Output` writes an audio file and `comfy_to_reaper.json` for REAPER to
import. Resampling is disabled by default via `resample_output`; enable it only when needed, then choose `output_sample_rate` (44100, 48000, or 96000 Hz).

Additional YuE2 helpers are available when YuE2-ComfyUI is installed:

- `REAPER Text Input` returns `text` from `text` or `text_path` in the request.
- `REAPER MIDI → YuE2 Score` reads `midi_path`, invokes YuE2's MIDI parser, and
  returns `score_abc` for the YuE2 workflow. Its optional `instrument_tracks`
  field accepts comma-separated MIDI part numbers (for example `2,3,4`) and
  merges those parts into the instrumental voice.
- `REAPER Score Output` saves ABC and optional lyrics into the project cache and exports the same score to MIDI by default. The MIDI can be imported back into REAPER with `ReaComfy - Import Score Output.lua`.
- `REAPER Chords → ABC (Chord-only)` builds a valid two-voice YuE2 score from sectioned chord text without requiring MIDI. Set BPM and key, then connect its `score_abc` output to `YuE2 Generate Song` or `REAPER Score Output`.
- `REAPER Video Input` exposes `video_path`, native `video`, `woosh_video`, `images`, and `fps`. `limit_duration` is disabled by default for compatibility; enable it when a downstream frame-based pipeline should use only `max_duration_s` seconds.
- `REAPER Audio Inpaint / De-Click Repair` removes clicks, mouth/saliva noise and short digital dropouts on the CPU, with no model to download. It whitens the audio with a short-term AR and a pitch predictor, so speech, sibilants, drums and synth waveforms are left alone while clicks stand out, and rebuilds each damaged region with high-order least-squares AR interpolation (LSAR) that keeps the pitch and texture around it. Mouth noise is replaced only above 2 kHz, so the voice underneath stays original; dropouts up to 50 ms are rebuilt full-band. `sensitivity` / `mouth_sensitivity`: lower finds more. `speech_protect`: higher keeps more natural transients. `strength`: 1.0 removes a found defect completely, 0.5 halves it. Set `output` to `removed` to hear only what was taken out. SciPy installs itself if missing.
- `REAPER EQ Match` uses deterministic STFT/iSTFT DSP to match a target recording to a reference spectrum. It includes strength, maximum correction, smoothing, and low/high frequency limits; it does not use an AI model or change timing.
- `REAPER Dynamic Voice Match` is the dialogue-oriented variant. It follows the target spectrum frame by frame with voice gating and slow temporal smoothing, helping stabilize interview and lavalier recordings whose tone changes with distance or mic angle. Use a clean speech excerpt as `reference_audio`; start with match strength 0.35–0.50 and maximum correction 2–3 dB.
- `REAPER Dynamic Voice Leveler` is an offline speech-dynamics processor. It smooths short-term voice level, leaves gated pauses mostly unchanged, limits the correction range, and applies a final peak safety stage. A practical dialogue starting point is target `-20 dB`, gate `-45 dB`, max gain/cut `6 dB`, attack `120 ms`, release `650 ms`, ceiling `-1 dB`.
- `REAPER ClearVoice Enhance` is a full-band neural speech enhancer based on ClearVoice's `MossFormer2_SE_48K` model. It is intended for modern 48 kHz dialogue cleanup and can be blended with the dry signal using `strength`. ClearVoice and the model checkpoint install themselves on first use.
- `REAPER ClearVoice Super Resolution` uses `MossFormer2_SR_48K` to restore speech bandwidth after low-rate cloning or processing. It accepts any input sample rate, internally works at 48 kHz, and outputs 48 kHz audio. ClearVoice recommends enhancement before super-resolution when the source is noisy.
- `REAPER Room Match Hybrid` estimates a usable room profile from a lavalier plus a distant/ambient microphone. It aligns the recordings, estimates a regularized transfer response over speech frames, smooths the transfer magnitude, applies a gentle high-frequency roll-off (`ir_high_cut_hz`, not a brick-wall cut), extracts early reflections, fits the late-tail shape from the measured decay/RT60, and allows tuning it with `tail_rt60_scale`. Use `early_gain_db` and `late_gain_db` to balance early reflections against the late tail independently. A soft RMS gate (`tail_floor_db`) prevents crunchy noise, while `early_reflections_ms` controls the measured early-field window. `max_alignment_ms` limits how far the node searches for the relative recording delay; it is not the IR length. The node estimates RT60/DRR/pre-delay and returns both a convolution-ready `room_ir` AUDIO output and a JSON profile. It is a hybrid acoustic approximation, not a measured swept-sine impulse response; the two recordings should be simultaneous and share the same clock as closely as possible.

Use the original, reverberant room recording for `ambient_audio`. If the lavalier is noisy, light denoise or dereverb may be applied before `lapel_audio`, but do not dereverb the ambient input before estimating the IR because that removes the reflections the node needs. With only one recording, this node cannot reliably recover the real room response; use a dereverb/denoise chain instead, or apply an existing IR.
- `REAPER LavaSR v2` reconstructs missing speech high frequencies and always outputs 48 kHz. It is intended after denoising or low-rate voice processing; use `strength` below 1.0 when a conservative blend is preferred.

### Node menu groups

The nodes are grouped in the ComfyUI menu under `REAPER AI/Inputs`,
`REAPER AI/Enhancement`, `REAPER AI/Outputs`, `REAPER AI/Video`, and
`REAPER AI/YuE2`. These are UI subfolders; the nodes remain one bridge pack
and do not need to be copied into several custom-node directories.

### REAPER Video Input outputs

- `video_path` (`STRING`) — the full source-file path. Use this with path-based video or Whisper nodes.
- `video` (`VIDEO`) — the native ComfyUI video object, for nodes that accept the standard `VIDEO` type.
- `woosh_video` (`WOOSH_VIDEO`) — a Woosh-compatible video package for the 2026 Woosh V2A/Foley pipeline.
- `images` (`IMAGE`) — decoded video frames, useful for PrismAudio and image/frame-based nodes.
- `fps` (`FLOAT`) — the frame rate associated with `images`.

`max_duration_s` affects only `woosh_video`, `images`, and `fps`, and only when `limit_duration` is enabled. The default is full-duration extraction (`limit_duration = false`). `video_path` and native `video` always refer to the complete source file. The duration of the generated audio remains controlled by the downstream model, for example Woosh Sampler's `latent_frames` setting; the input node does not change that model setting.

The request manifest can contain `midi_path`, `text_path`, or inline `text` in
addition to the audio `path`. The REAPER-side sender must create these fields.

## REAPER side

The REAPER scripts (launcher, send/import actions, Python bridge) are
distributed separately as the ReaComfy script package. Load `ReaComfy.lua` in
REAPER's action list; see that package's README.
Downloads and discussion: [REAPER forum thread](https://forum.cockos.com/showthread.php?t=311369).

## License

MIT, see [LICENSE](LICENSE). Optional models keep their own licenses.
