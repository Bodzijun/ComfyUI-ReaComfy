"""ComfyUI nodes for exchanging audio with the active REAPER project.

Cross-platform (Windows, macOS, Linux). The nodes use a small file protocol
in REAPER_AI_CACHE. A REAPER action can
write ``reaper_to_comfy.json`` with an input path; REAPER Output writes a
``comfy_to_reaper.json`` manifest and an audio file for the import action.
"""

__version__ = "0.5.1"

if __package__:  # skipped only when pytest walks the repository root
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
