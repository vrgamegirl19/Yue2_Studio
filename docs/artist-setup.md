# Artist setup preparation (integration in progress)

This is the setup layer for the upcoming Artist Trainer, not a finished release.
The Artist panel, setup actions and training worker now use the new registry
paths. Playback integration and installer support are implemented and tested on
CPU in isolated installs. Real isolated setup and GPU smoke checks are documented
below; final user review remains a release gate. These are technical checks, not
a guarantee of singer similarity or compatibility with every GPU.

## Artist panel and queue

Artist trainer sits below Style trainer. Enter/reuse model folders, then queue
Check setup, Prepare model paths / download missing, or Install separate Artist
runtime. The download action requires acceptance of model terms and confirmation;
the runtime action requires confirmation. Checks are read-only. Preparing model
paths records them only after verification, even when all files already exist.
Setup runs in the shared serial queue, with results/errors in Library → Open run.

Scan matching audio and full lyric files, review the lyrics, enter a shared style
and trigger, and save a versioned setup. Source files are not changed. Training
requires a saved setup, at least two 30–360 second recordings and explicit GPU
approval. The last selected song is held out; controls default to 500 updates,
rank 64, learning rate 0.0001, checkpoint interval 250 and alignment weight 0.08.
MMS alignment's first use may download Demucs/MMS weights. Experimental
Whisper-assisted timing instead downloads `htdemucs_ft` and Whisper large-v3,
and saves per-song timing JSON for review. Zero alignment weight skips either
timing method but not lyric conditioning. No silent clipping occurs.

The queue freezes verified model paths and the registered runtime path into the
job; the worker rechecks pinned assets before GPU work. New checkpoints embed the
shared training style. Cancellation waits for a safe training boundary and retains
completed adapter snapshots, not optimizer-resume state. Setup cancellation uses
the existing process-tree termination path; partial downloads/environments remain
for diagnosis rather than replacing a previous successful setup.

## Explicit actions

Run with the installed Studio's Python 3.12 interpreter and installed source on
the import path. `YUE2_KIT` must identify that installation, not the add-on repo.

```
python -m yue2_studio.artist_setup check
python -m yue2_studio.artist_setup download-models --confirm --accept-model-terms
python -m yue2_studio.artist_setup install-runtime --confirm
```

`check` reads files and hashes and probes an already registered runtime with
CUDA hidden and Hugging Face offline. It installs nothing, downloads nothing,
and returns nonzero when setup is incomplete. Hashing large local weights takes
time and disk bandwidth. Its success is not a GPU-memory or musical-quality test.

Model paths can be supplied using `--model`, `--vae`, `--mert`, `--encoder` and
`--regularizer`. Existing local model folders and the Style Trainer's downloaded
model/VAE registry are reused. Only the tested, hash-pinned releases are accepted.
An existing invalid folder fails validation; it is not overwritten automatically.

Missing assets download to `models/artist-cache/<kind>/<revision>` using ordinary
files, not symlinks. Exact-file requests avoid snapshot progress-library failures.
Transfers include explicit support/license files, not the
community's training scripts or entire audio corpus. The five paths publish to
`training/artist-models.json` only after every required file passes validation.
Failed downloads retain partial cache files and leave an older registry intact.
Allow substantial disk space: full base/VAE/MERT weights plus the encoder,
companion and reference pack, and later caches and runtime dependencies.

The runtime action creates a fresh `.artist-runtimes/<unique-id>` environment,
installs pinned Torch/torchaudio 2.10.0 CUDA 13.0 and supporting packages
(including stable-ts, openai-whisper, numba and llvmlite), checks
dependency consistency and CPU imports, then writes `training/artist-runtime.json`.
It does not upgrade the base Studio environment or `.venv-artist`. A source-only
`.pth` reference makes installed YuE2/Studio code available without sharing the
base environment's site-packages. Failed attempts remain available for diagnosis;
no old runtime is removed or replaced. This uses more disk space than sharing
Torch. A working Python 3.12 venv/pip installation and network access are required.

The known Demucs dependencies are installed explicitly before Demucs itself
(`--no-deps`), followed by `pip check`; this preserves the selected Torch pair.
An older registered Artist runtime may still work for MMS but needs the new
Whisper packages. Use **Install separate Artist runtime** again or follow the
manual commands in [the Artist Trainer guide](artist-trainer.md#install-whisper-timing-support).
The selected Whisper method probes its imports and the FFmpeg CLI on `PATH`
before queuing GPU work. FFmpeg is a separate system installation, not a Python
package in the Artist runtime. First-use
separator/Whisper weight downloads still happen during preparation. Passing CPU
imports does not establish alignment accuracy or GPU readiness.

Package commands use pip's system-certificate trust-store support and ignore
unrelated global index/trusted-host settings. TLS verification stays enabled.
Package failures retain the underlying command output in the setup log.

## Model sources and terms

Review the model cards/licenses before confirming downloads:

- [Official YuE2 model](https://huggingface.co/m-a-p/YuE2-3B)
- [Official VAE](https://huggingface.co/m-a-p/YuE2-Vae)
- [Official full-song MERT](https://huggingface.co/m-a-p/MERT-v2-FullSong)
- [Community encoder and NAR companion](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
- [Community reference pack](https://huggingface.co/datasets/Mothersuperior/yue2-minted-corpus)

Exact revisions and SHA256 values are recorded in `artist_setup.ASSETS`. The
community release's model card includes noncommercial terms; do not assume the
Studio code license grants rights to model weights or training recordings.
No weights, training recordings or personal run artifacts are distributed here.

MERT requires its two pinned architecture files at training time. Setup verifies
their SHA256 without executing them. The community scripts are not executed.

## Validation status

CPU tests cover explicit confirmation, local reuse, pinned transfer requests,
hash rejection, fresh runtime commands, and failure-safe registry publication.
Those unit tests mock transfers and pip commands. The full suite passes 150 tests
plus 28 subtests, alongside the Artist/Style/LoRA/Surprise UI-handler checks.
Real isolated setup tests downloaded and hash-verified the pinned encoder,
companion and reference pack, reusing verified base/VAE/MERT folders read-only.
The downloaded encoder and companion also loaded on CPU. This is not a cold
download test of all five model groups.

Real dependency checks found and fixed Windows certificate handling, including
nested pip build processes, without disabling TLS verification. The supporting
packages, Demucs, dependency consistency check and CPU imports passed, followed
by a successful complete fresh runtime installation with the corrected installer.
The fresh runtime also imported the installed Artist preparation/training modules
with CUDA uninitialized. Passing CPU imports alone does not establish musical quality.
The isolated Studio's real setup queue also completed `check`, with all files and
imports ready and a saved result receipt. No training was started.

### Real GPU preparation and training

On an RTX 5090 (32 GB), a separate fresh-runtime test encoded two complete
recordings (about 132 and 162 seconds), downloaded the first-use Demucs/MMS
weights, separated vocals and aligned the lyrics. Word timings were finite,
ordered and within each recording. Alignment confidence is not proof of a
perfect lyric match. A missing alignment resampler import was fixed and covered
by a regression test; the base Studio still does not need SciPy at module import.

The queued rank-64 test completed two updates with alignment weight 0.08 and
saved checkpoints after both updates plus the final adapter. Discovery returned
the Artist bundle and its training style. Peak allocated VRAM was 7.25 GiB.
A separate full-model backward pass on the longest reference (9,529 tokens,
no clipping, rank 64) passed with finite gradients and 8.11 GiB peak allocated
VRAM. These short tests do not establish long-run training quality or an ETA.

### Playback and restoration

Fixed-input No score/Torch playback generated baseline, Artist-bundle and
unloaded-bundle audio with 32 synthesis steps. Each was deliberately limited to
750 semantic tokens (30 seconds); the native receipts retain that truncation.
The audio has not been auditioned as a quality or singer-similarity evaluation.

An initial same-seed baseline/restored token-equality assertion failed. A separate
control also differed between two baseline-only generations before loading any
LoRA, so exact repeated audio is not claimed for this backend. The failed test
and all audio were retained. Direct checks found all 627 model parameter hashes
identical after adapter use and after an intentional exception inside the merge
context. With the same semantic tokens, synthesis changed with the companion
enabled and returned exactly to its baseline latents after unloading.
An additional Artist render through the real Studio generation API/queue also
completed, using the base Studio runtime, bundle identity guards and the same
explicit 30-second cap. Original recording/lyric hashes were verified unchanged.
The queue correctly marked that capped render `needs_review` for its token-limit
warning; the native generation result completed and saved playable audio.
