# Artist Trainer and playback — experimental

## Setup walkthrough

![Artist dataset and lyric pairing](../images/artist-trainer-dataset.png)

This workspace capture shows an example path, a local-only recovery shortcut and
a historical test banner. Those are not a supplied dataset or verification of
your PC. Use the release setup controls described below.

Artist trainer is its own sidebar entry below Style trainer. It learns an AR
adapter from full songs and lyrics; Style Trainer learns an acoustic adapter from
clips without lyric sidecars. Neither guarantees voice cloning.

1. Open Artist trainer and inspect its five model locations. They auto-fill from
   saved Artist paths, available Style model/VAE paths, or default folders under
   your own installation. A filled field does not mean the files exist.
2. Click **Check setup**. Review model terms before explicitly preparing paths /
   downloading missing models. Valid local files are reused; invalid existing
   folders are not overwritten. Install the separate Artist runtime with its own
   confirmation. These queue actions do not start training or change normal Studio Python.
3. After model preparation finishes, reload the page to display the registered
   paths, then check again. Do this before entering unsaved dataset edits. The
   check verifies files and CPU imports, not available training VRAM or quality.
4. Choose a dataset folder and its lyric suffix. `My song.wav` pairs with
   `My song.lyrics.txt` when `.lyrics.txt` is selected, or `My song.txt` when `.txt`
   is selected. Scan reads direct files, not subfolders. Use UTF-8 full lyrics,
   preferably with [Verse]/[Chorus] tags; a style caption is not a lyric file.
5. Select at least two complete 30–360 second recordings. Review lyrics against
   each song, exclude errors/unwanted songs, and confirm review. Filenames alone
   do not establish a match. Use recordings you have permission to train on.
6. Enter a name, shared style and trigger, then save the setup. No per-song style
   sidecar is required. A consistent singer/style is easier to evaluate than a
   mixed collection, but does not guarantee singer identity. The last selected
   song is held out for validation. Changed source files require a rescan/new save.
7. Select the saved setup, review training controls and explicitly confirm GPU use.
   The queue uses the saved dataset/style, not unsaved edits, and freezes the chosen
   controls and model/runtime paths. It does not coordinate other GPU applications.

The **model / vae** paths are the full PyTorch generation model and decoder.
**mert** extracts audio features; **encoder** contains the community semantic
encoder and acoustic companion; **regularizer** supplies reference songs/tokens
mixed with your dataset. Opening the page downloads nothing. Allow space for full
models, a separate runtime, caches and adapter checkpoints. See [setup details](artist-setup.md).

## Training controls and progress

![Artist training controls with example values](../images/artist-trainer-controls.png)

The screenshot shows older **800 / 200** values and predates the lyric-timing
method selector; current defaults are **500 / 250**.

| Control | Default | Meaning |
| --- | --- | --- |
| Training steps | 500 | Optimizer updates; two song examples per update, not two full dataset passes |
| Save checkpoint every | 250 | Intermediate adapter snapshots for comparison |
| Rank | 64 | Adapter capacity; higher can use more memory/storage without better results |
| Learning rate | 0.0001 | Update size; overly aggressive learning can degrade results |
| Lyric timing method | MMS | Choose the original forced aligner or experimental Whisper-assisted timing |
| Alignment weight | 0.08 | English lyric-timing supervision; 0 skips separation/alignment, not lyrics |

Click the information disclosures for supported ranges and tradeoffs. These are
starting values, not a guarantee that 500 steps is best. Artist uses complete songs,
not the Style Trainer clip-length setting. Generation GPU presets do not automatically
tune training speed, memory use or batch size.

Open **Song library → Open run** for encoding, vocal separation, lyric alignment,
validation, steps, loss and checkpoint artifacts. First-use MMS alignment downloads
Demucs/MMS weights; Whisper-assisted timing downloads fine-tuned Demucs and
Whisper large-v3 weights. Preparation can take longer than a short training run and
happens before optimizer steps. Confidence and loss are not quality scores.

### Choose MMS or Whisper-assisted timing

Both methods use the complete original song for semantic encoding and the full
lyric file to condition training. Separation is only for estimating lyric timing;
neither method changes the source recording. Both automatically proceed from
preparation to LoRA training—there is no manual review step.

| | MMS (default) | Whisper-assisted (experimental) |
| --- | --- | --- |
| Vocal separation | Demucs `htdemucs` | Fine-tuned Demucs `htdemucs_ft` |
| Timing | Forces every supplied English lyric word onto the vocal audio | Whisper large-v3 recognizes audible words; Studio matches them to the supplied lyrics |
| Missing/uncertain words | Still receive forced positions | Positions are estimated for review, but do **not** supervise timing loss |
| Diagnostics | Per-song mean forced-alignment confidence in the run log | Exact/fuzzy/estimated counts in the log and per-song timing JSON in `result/alignment` |

Whisper's exact/fuzzy counts do not prove the timestamps are correct. Review the
lyric files carefully, especially for screams, overlapping vocals or dense mixes.
The experimental method currently handles English lyrics. When alignment weight
is **0**, neither method separates vocals or runs lyric timing; lyric files are
still required for Artist training. The method choice is frozen in each run and
old MMS setups/runs continue to work.

### Install Whisper timing support

The **Install separate Artist runtime** action on the Artist Trainer page now
includes the pinned Whisper packages. It creates a fresh Artist-only environment;
normal Studio Python, model files, songs and existing runs are not upgraded.
After it completes, use **Check setup**. If you had installed the Artist runtime
before Whisper was added, choose **Install separate Artist runtime** again before
queueing Whisper training. MMS can continue using the older runtime.

Alternatively, if you manage the existing Artist environment yourself, install
the packages into the Python recorded in `training/artist-runtime.json`—not into
the normal Studio environment. On Windows PowerShell, from the Studio folder:

```powershell
$artistPython = (Get-Content -LiteralPath 'training/artist-runtime.json' -Raw | ConvertFrom-Json).python
& $artistPython -m pip install 'stable-ts==2.19.1' 'openai-whisper==20250625' 'more-itertools==10.8.0' 'numba==0.61.2' 'llvmlite==0.44.0'
& $artistPython -m pip check
ffmpeg -version
```

Then use **Check setup** and queue a run with **Whisper-assisted** selected.
Whisper also requires the [FFmpeg command-line program](https://github.com/jianfch/stable-ts)
on your system `PATH`; installing the Python packages does not install FFmpeg.
If `ffmpeg -version` is not found, install FFmpeg for your operating system and
reopen Studio so its process sees the updated `PATH`.
The first such run downloads Whisper large-v3 and `htdemucs_ft` weights with an
internet connection; allow disk space and time for both. The setup check probes
imports on CPU and does **not** download those weights or validate available GPU
memory. If the separate runtime lacks Whisper or FFmpeg, the queue rejects the new method
with an installation hint before starting training. See [Artist setup](artist-setup.md)
for the other model prerequisites and terms.

Checkpoints/final adapters live in the run's `result` folder. They contain adapter
weights, not optimizer/RNG state: exact resume is not implemented. Cancellation
waits for a safe boundary; force-closing can lose unsaved work. Keep completed
checkpoints and logs after a failure. No comparison song is generated automatically.

## Playback and portability

Completed Artist runs appear in **Style / Artist LoRA**, including Surprise me.
Selecting an adapter offers its embedded training style in both creation style
boxes; existing text requires confirmation. You can edit the result. Surprise me
can lock that style while the LLM writes new titles and lyrics.

Artist playback requires **No score**, **Torch/Torch-eager**, **quantization None**
and **AR offloading disabled**. Incompatible settings fail rather than silently
changing backend or generation mode. GGUF is unsupported. Only one Style adapter
or Artist bundle can be selected; arbitrary LoRA stacking is not supported.

An Artist bundle contains its AR safetensors adapter, sibling `manifest.json`,
and the exact pinned community NAR companion referenced by that manifest. Keep
these together when transferring a bundle. Relative companion paths resolve from
the manifest folder; absolute paths refer to that PC. A standalone safetensors file
without its manifest/companion is insufficient. Checkpoints can be selected via
the local-path control while retaining the same sibling manifest.

The queue freezes adapter, companion and base-model identities. Playback checks
those identities against the actual files before generating. Artist deltas apply
to semantic generation and acoustic-prefix conditioning; the companion applies
to acoustic synthesis. Every modified weight is restored after each stage,
including failure/cancellation paths. No encoder/MERT is needed merely to play an
already trained bundle; the base YuE2 model, VAE and companion are needed.

Strength scales the Artist AR delta. At any nonzero strength, the companion stays
at its trained full strength. Zero disables both; it is not an AR-only comparison
with an unchanged companion. None selects original YuE2. Increased strength or
longer training is not a guarantee of singer similarity or better music.

The installer accepts stock upstream 0.1.6 and the known merged Style Trainer
pipeline, backs up replaced files, and refuses unfamiliar pipeline or adapter
code. It does not download models or install Artist dependencies automatically.

## Comparing and troubleshooting

Same seed is not an exact replay guarantee. Tested No score/Torch baseline-only
repeats varied even with identical style, lyrics and settings. The exact cause
is unconfirmed. Keep original files and compare multiple baseline/LoRA pairs;
see [comparison guidance](studio.md#seeds-and-comparing-results). The companion
can change the sound even when a new AR adapter has learned little, so a difference
alone is not evidence of singer learning.

- **Missing LoRA:** refresh the list. For intermediate checkpoints use their local
  path, retaining the sibling manifest and referenced companion. Do not remove a
  run folder while using its adapter.
- **Incompatible mode:** use No score, Torch/Torch-eager, None quantization and AR
  offloading off. A generation GPU preset may have changed offloading. No GGUF.
- **Missing lyrics:** check the filename suffix and audio stem. Zero alignment
  weight does not make lyric files optional.
- **OOM:** free competing GPU workloads and review the failed log and training
  controls. Generation memory presets do not tune this trainer; saved adapter
  checkpoints do not provide exact resume.
- **Abrupt generation ending:** inspect token-limit/truncation warnings. Training
  steps and save frequency do not control generated song length.

Fresh isolated setup, full-song alignment, a two-update rank-64 checkpoint test,
a 9,529-token backward test and playback passed on an RTX 5090. Exact parameter
restoration passed; repeated sampled songs were not identical. This is not
all-hardware, long-training or singer-quality certification. [Test scope](artist-setup.md).
