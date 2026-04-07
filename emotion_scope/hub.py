"""
HuggingFace Hub integration for EmotionScope.

Handles:
  - Downloading pre-extracted vectors from the Hub (for zero-setup demo)
  - Uploading extracted vectors + figures to the Hub (for sharing results)
  - Auto-downloading model weights via transformers (standard HF flow)

Usage:
    from emotion_scope.hub import download_vectors, upload_results

    # Pull pre-extracted vectors (auto-downloads if missing)
    path = download_vectors("google/gemma-2-2b-it")

    # Push your results to your own HF repo
    upload_results("your-username/emotion-scope-vectors", model_name="google/gemma-2-9b-it")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from emotion_scope.config import RESULTS_DIR, VECTORS_DIR, FIGURES_DIR

# Default HF repo for pre-extracted vectors shipped by the project
DEFAULT_HUB_REPO = os.environ.get("ES_HUB_REPO", "AidanZach/EmotionScope-vectors")


def _model_slug(model_name: str) -> str:
    """Convert HF model ID to a filename-safe slug."""
    return model_name.replace("/", "_")


def download_vectors(
    model_name: str = "google/gemma-2-2b-it",
    hub_repo: str = DEFAULT_HUB_REPO,
    revision: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Download pre-extracted emotion vectors from HuggingFace Hub.

    Checks local cache first. Returns the local path to the .pt file.

    Args:
        model_name: The model these vectors were extracted from
        hub_repo: HF repo ID containing the vectors
        revision: Git revision / branch (default: main)
        force: Re-download even if local file exists

    Returns:
        Path to the local .pt file
    """
    slug = _model_slug(model_name)
    local_path = VECTORS_DIR / f"{slug}.pt"

    # If we already have it locally, skip download
    if local_path.exists() and not force:
        return local_path

    from huggingface_hub import hf_hub_download

    filename = f"vectors/{slug}.pt"

    print(f"[hub] Downloading vectors for {model_name} from {hub_repo}...")
    try:
        # local_dir_use_symlinks was removed in huggingface_hub 1.0
        # Use it on <1.0, skip it on >=1.0
        import huggingface_hub
        hf_version = tuple(int(x) for x in huggingface_hub.__version__.split(".")[:2])

        dl_kwargs = dict(
            repo_id=hub_repo,
            filename=filename,
            revision=revision,
            local_dir=str(RESULTS_DIR),
        )
        if hf_version < (1, 0):
            dl_kwargs["local_dir_use_symlinks"] = False

        downloaded = hf_hub_download(**dl_kwargs)
        print(f"[hub] Downloaded to {downloaded}")
        return Path(downloaded)
    except Exception as e:
        print(f"[hub] Download failed: {e}")
        print(f"[hub] You can extract vectors locally with:")
        print(f"  uv run python scripts/extract_all.py --model {model_name} --sweep-layers")
        raise


def upload_results(
    hub_repo: str,
    model_name: Optional[str] = None,
    include_figures: bool = True,
    private: bool = False,
    commit_message: Optional[str] = None,
):
    """
    Upload extracted vectors and figures to a HuggingFace Hub repository.

    Creates the repo if it doesn't exist. Uploads:
      - results/vectors/{model_slug}.pt
      - results/figures/*.png (if include_figures=True)
      - A README with model metadata

    Args:
        hub_repo: Target HF repo ID (e.g., "your-username/emotion-scope-results")
        model_name: Specific model to upload vectors for (default: all .pt files)
        include_figures: Also upload figures
        private: Create as private repo
        commit_message: Custom commit message
    """
    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if needed
    try:
        api.create_repo(
            repo_id=hub_repo,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
    except Exception as e:
        print(f"[hub] Note: {e}")

    # Collect files to upload
    files_to_upload = []

    if model_name:
        slug = _model_slug(model_name)
        pt_path = VECTORS_DIR / f"{slug}.pt"
        if pt_path.exists():
            files_to_upload.append((str(pt_path), f"vectors/{slug}.pt"))
        else:
            print(f"[hub] Warning: {pt_path} not found, skipping")
    else:
        # Upload all .pt files
        for pt_file in VECTORS_DIR.glob("*.pt"):
            files_to_upload.append((str(pt_file), f"vectors/{pt_file.name}"))

    if include_figures:
        for fig_file in FIGURES_DIR.glob("*.png"):
            files_to_upload.append((str(fig_file), f"figures/{fig_file.name}"))
        for fig_file in FIGURES_DIR.glob("*.svg"):
            files_to_upload.append((str(fig_file), f"figures/{fig_file.name}"))

    if not files_to_upload:
        print("[hub] No files to upload.")
        return

    # Generate a README for the Hub repo
    readme_content = _generate_hub_readme(model_name, files_to_upload)
    readme_path = RESULTS_DIR / "_hub_readme.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    files_to_upload.append((str(readme_path), "README.md"))

    # Upload
    msg = commit_message or f"Upload EmotionScope vectors"
    if model_name:
        msg += f" for {model_name}"

    print(f"[hub] Uploading {len(files_to_upload)} files to {hub_repo}...")
    for local, remote in files_to_upload:
        print(f"  {remote} ({Path(local).stat().st_size / 1024:.0f} KB)")

    for local_path, remote_path in files_to_upload:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=hub_repo,
            commit_message=msg,
        )

    # Clean up temp readme
    readme_path.unlink(missing_ok=True)

    print(f"[hub] Done. View at: https://huggingface.co/{hub_repo}")


def _generate_hub_readme(model_name: Optional[str], files: list) -> str:
    """Generate a README.md for the HF Hub repository."""
    vector_files = [f for _, f in files if f.endswith(".pt")]
    figure_files = [f for _, f in files if f.endswith(".png") or f.endswith(".svg")]

    lines = [
        "# EmotionScope — Pre-extracted Emotion Vectors",
        "",
        "Pre-extracted emotion direction vectors from [EmotionScope](https://github.com/AidanZach/EmotionScope),",
        "an open-source toolkit for probing functional emotion representations in language model residual streams.",
        "",
        "## Usage",
        "",
        "```python",
        "from emotion_scope.hub import download_vectors",
        "from emotion_scope.extract import EmotionExtractor",
        "",
        '# Download pre-extracted vectors',
        f'path = download_vectors("{model_name or "google/gemma-2-2b-it"}")',
        'saved = EmotionExtractor.load(str(path))',
        'vectors = saved["vectors"]  # Dict[str, Tensor] — 20 emotion directions',
        "```",
        "",
        "## Vectors",
        "",
    ]
    for f in vector_files:
        lines.append(f"- `{f}`")
    if figure_files:
        lines.append("")
        lines.append("## Figures")
        lines.append("")
        for f in figure_files:
            lines.append(f"- `{f}`")
    lines.extend([
        "",
        "## Reference",
        "",
        "Based on: Sofroniew, N., et al. (2026). \"Emotion Concepts and their Function in a Large Language Model.\"",
        "[Transformer Circuits Thread](https://transformer-circuits.pub/2026/emotions/index.html).",
        "",
        "## License",
        "",
        "MIT",
    ])
    return "\n".join(lines)
