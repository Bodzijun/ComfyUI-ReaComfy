# Changelog

All notable changes to ReaComfy are documented here.

## [0.6.0] - 2026-11-26

### Fixed

- DBLM-SE stopped with "needs an extra module (vector_quantize_pytorch)" when the module was present but broken. Missing or broken modules are now reinstalled automatically (with an upgrade on the second try) and checked before the node continues.

### Added

- `REAPER Nemotron RE-USE (Universal Speech Enhancement)`: NVIDIA's commercial-use RE-USE model (OpenMDW-1.1) for noise, reverb, clipping, codec and bandwidth damage at 8–48 kHz, with `speech` and `removed` outputs, optional bandwidth extension and chunked processing. Runs with `mamba-ssm` on Linux + CUDA or a built-in PyTorch Mamba scan everywhere else.
- `REAPER ZONOS3 TTS (Voice Clone + Emotion/Accent)`: Zyphra ZONOS3-Mini text-to-speech (43 languages incl. Ukrainian) with voice cloning from a REAPER item and steering controls for emotion, pace, projection, mood, accent, texture and room. By default ReaComfy installs ZONOS3 into its own Python 3.12 environment on first use and keeps it loaded in a background worker (no terminal, no server to launch); an external ZONOS3 server or venv can still be used.
- `REAPER DBLM-SE (Generative Denoise + Dereverb)`: DBLM-SE-1B (Apache-2.0) generative speech restoration (16 kHz output), chunked for long files.
- Missing Python modules for the new nodes are installed automatically into ComfyUI's environment; model code and weights download on first use.
- Example workflows `reaper_nemotron_reuse.json`, `reaper_zonos3_tts.json` and `reaper_dblm_se.json`.

## [0.5.3] - 2026-11-26

### Fixed

- Removed the optional `restoration` extra (`clearvoice`) from the package metadata. ComfyUI-Manager tried to install it and failed; ClearVoice stays an optional manual install (see README).
- `scipy` (used by Audio Inpaint and Room Match) is now a regular requirement.

## [0.5.2] - 2026-11-26

### Changed

- Added the REAPER forum thread (https://forum.cockos.com/showthread.php?t=311369) to the README and package links.

## [0.5.1] - 2026-11-26

### Changed

- Reformatted `nodes.py` (one statement per line) to pass the Comfy Registry code checks. No behaviour changes.

## [0.5.0] - 2026-11-26

### Added

- `REAPER Voice Isolator (Inworld)` ComfyUI node: separates the foreground voice from noise, music and background with Inworld Voice Isolator (Apache-2.0, 48 kHz). Offline and streaming (realtime) models, strength blend, per-channel or mono processing, `voice` and `background` outputs.
- `voice_isolator_api.json` workflow preset with a strength macro in the launcher.
- The ComfyUI nodes are packaged as a standalone custom-node repository (`ComfyUI-ReaComfy`) with `pyproject.toml` for the Comfy Registry, `requirements.txt`, LICENSE and tests.

### Changed

- Cross-platform paths: all REAPER scripts now join paths with `/`, which REAPER and Lua accept on Windows, macOS and Linux. The user cache falls back to `HOME` when `USERPROFILE` is not set, and absolute POSIX paths are recognised.
- The default bridge Python is `python3` on macOS/Linux and `python` on Windows.
- Video export runs FFmpeg directly on macOS/Linux (PowerShell only on Windows) and looks for FFmpeg in REAPER `UserPlugins` and common Homebrew/system locations.
- The Python bridge looks for FFmpeg in the REAPER resource folder on all three platforms.
- Removed the machine-specific `G:\pinokio` path from the node documentation.
### Included from the former Unreleased section

- Added optional **Choose cached result when importing** mode. It lists recent cached ComfyUI audio files with timestamps, allowing a specific iteration (or several selected files) to be imported without pulling in results from later runs.
- Replaced the text-based cached-result selector in the ReaImGui launcher with a checkbox table, including multi-selection for stem imports.
- Track imports now use only the configured track name; output labels remain on takes, and a new track with the same base name is created when the destination position is occupied.
- Fixed reuse of multiple same-named output tracks: each selected result now searches all free tracks in the group before creating another one.
- Added `REAPER Dynamic Voice Match`, a voice-focused time-varying EQ match with voice gating, temporal smoothing, and limited correction for interview/lavalier recordings.
- Added `REAPER Dynamic Voice Leveler`, an offline voice-dynamics stabilizer with gating, attack/release smoothing, correction limits, and peak safety.
- Added `REAPER Reference Input` for separate target/reference audio workflows; the Send reference action now preserves the target path and writes `reference_path`.
- Added optional `REAPER Output` resampling. It is disabled by default; when enabled, choose 44100, 48000, or 96000 Hz.
- Fixed ComfyUI execution caching for `REAPER Input` and `REAPER Reference Input`: changing the REAPER manifest now invalidates the cached audio result.
- Moved Send reference audio into the main Direct ComfyUI Bridge and moved Import Video Output next to the other import controls.
- The main audio bridge button now reflects the selected input mode: direct source transfer or temporary Glue export.
- Added **Selected track** as a return destination and improved source-position fallback when a stale bridge status manifest lacks the original item position.
- Additional host-specific setup guidance and broader platform testing.

## [0.4.4] - 2026-10-18

### Fixed

- Guarded text import against corrupted edit-cursor values and out-of-range subtitle timestamps that could freeze REAPER.

## [0.4.3] - 2026-10-18

### Fixed

- Text importer now verifies created REAPER objects, selects imported items, and moves the edit cursor to the imported range for immediate visibility.

## [0.4.2] - 2026-10-18

### Fixed

- Fixed invalid Lua path escaping in `ReaComfy - Import Text Output.lua`, which prevented the script from loading.

## [0.4.1] - 2026-10-16

### Fixed

- Added a compatibility shim for ClearVoice with librosa 0.11+, restoring the removed `librosa.util.normalize` API without downgrading ComfyUI dependencies.

## [0.4.0] - 2026-10-15

### Added

- `REAPER Text Output` for exporting SRT/VTT/plain text from ComfyUI.
- REAPER import as note items, project regions, or markers.
- Importer button in the main launcher bridge block.
- Clear separation between text output import and audio result import.
- Multi-output audio import support for stem workflows.

## [0.3.0] - 2026-09-20

First public release after three development iterations.

### Added

- ReaComfy workflow launcher with resizable ReaImGui interface and persistent setup.
- API workflow presets with refresh support, macro controls, additional workflow folders, and editable ComfyUI host URL.
- Audio bridge for original-source and glue workflows, WAV/FLAC output, 48 kHz conversion, source-position return, lanes, tracks, and multi-output imports.
- Direct `REAPER Input` and `REAPER Output` nodes for editable ComfyUI pipelines.
- Video source transfer and FFmpeg export from a REAPER time selection or cutscene item.
- Configurable video and cutscene track names.
- `REAPER Video Input` outputs for path, native ComfyUI video, Woosh video, PrismAudio images, and FPS, with optional frame-duration limiting.
- Frame-duration limiting is opt-in; full-duration frame extraction is now the default for compatibility with existing video pipelines.
- Added `REAPER EQ Match`, a reference-based STFT/iSTFT correction node for controlling excessive spectral emphasis in cloned voices.
- Fixed multi-output node exchange: all `REAPER Output` files from one job are now imported, while stale files from previous jobs are ignored through per-job manifests.
- Video output import back to the REAPER timeline.
- YuE2 source export for lyrics, regions, MIDI, chords, arrangement plans, and reference audio.
- Multi-track MIDI export, section markers, chord-only ABC generation, score output, MIDI output, and score import.
- Cache cleanup and project-local cache manifests.
- Cross-platform path resolution and user configuration without personal workflow paths in the release package.
- Node descriptions and README documentation for inputs, outputs, workflow contracts, and custom presets.

### Fixed

- Windows path handling for spaces, Unicode, parentheses, and nested project folders.
- Stale manifest/result selection and cache collisions between consecutive jobs.
- Long chord and arrangement items spanning multiple REAPER regions.
- Return placement and timing for source-position imports.
