# YuE2 Studio · Beta

A local web studio for **YuE2** with original songs, melody covers, an LLM writing room,
automatic song batches, experimental Style and Artist LoRA training, and detailed generation controls. Custom HTML/CSS/JavaScript UI;
no Gradio or Node build. Artist training has a separate, explicitly installed runtime.

**Beta preview:** tested on Windows with an RTX 5090. Other hardware configurations
are still being validated. GGUF and lower-VRAM presets remain experimental.
Planned first release: `v0.1.0-beta.1`.

![YuE2 Studio with the song editor, navigation and LLM writing room](images/studio-overview.png)

*Write lyrics, shape a style, and generate music in one local workspace. See the [illustrated guide](docs/studio.md) for each part of the UI.*

## 1. Install YuE2 first

Follow the **[official YuE2 installation and quick start](https://github.com/multimodal-art-projection/YuE#quick-start)**.
Make sure its example can generate a song before adding Studio. The old YuE v1 branch is
not compatible. This add-on targets **yue2-infer 0.1.6**; the tested upstream commit is
`92a73cc7652fcc1f937855e4b765e0a0edd7ff2e`.

The upstream baseline is Python 3.12 and a BF16-capable NVIDIA GPU with 24 GB VRAM.
Studio's Windows torch/cuDNN compatibility path was also tested on an RTX 5090.
Long songs can still exceed VRAM. See the [settings and performance guide](docs/settings.md).

## 2. Clone Studio inside your installed YuE2 folder

From the folder containing YuE2's `src`, `.venv`, and `models`:

```powershell
git clone https://github.com/vrgamegirl19/Yue2_Studio.git
cd Yue2_Studio
.\"Start Yue2 Studio.bat"
```

Or double-click **Start Yue2 Studio.bat** inside `Yue2_Studio`.
The launcher installs the UI into the parent YuE2 folder, backs up replaced files,
and opens **http://127.0.0.1:7862**. No pip reinstall is needed for the add-on.

```text
Your-YuE2-folder/
  .venv/
  models/
  src/yue2/
  Yue2_Studio/                 ← clone this repository here
    Start Yue2 Studio.bat
    install_studio.py
```

The installer also installs acoustic LoRA support in `src/yue2/pipeline.py` and
`src/yue2/lora.py`. It refuses an unfamiliar/custom pipeline instead of overwriting
it. The supported baseline is the upstream commit above; keep a separate copy of
any custom engine changes. See [Style Trainer setup](docs/trainer.md) and
[LoRA playback](docs/lora.md). These are Studio extensions, not upstream training APIs.

The experimental [Artist Trainer](docs/artist-trainer.md) adds full-song lyric
conditioning and AR adapters with a pinned community acoustic companion. Follow
[Artist setup](docs/artist-setup.md) for separate dependencies, model paths and
terms. Lyric timing offers the original MMS forced aligner or experimental
Whisper-assisted matching; the guide explains their differences and separate
runtime installation. Playback requires No score, Torch, no quantization and AR offloading off;
GGUF and arbitrary adapter stacking are unsupported. The installer also accepts
the known merged Style Trainer pipeline, with backups, but rejects unfamiliar
Artist adapter code. Fresh isolated setup, alignment, short GPU training and playback
tests passed on an RTX 5090; see the documented [verification limits](docs/artist-setup.md#validation-status).

The installer modifies `src/yue2/cuda_graph.py` to detect builds without compiled
Flash Attention, keeping fast CUDA graphs through cuDNN/SDPA. It also installs the
Studio package, launcher, and score/transcription helper scripts. Existing files
that change are backed up in the parent `.studio-backups/<timestamp>/` folder.
Review custom changes before installing. Stop servers on alternate ports manually
before updating; the installer checks the default port 7862.

### Linux or a different Python environment

Use the Python interpreter that already runs YuE2:

```bash
# From the Yue2_Studio clone:
../.venv/bin/python install_studio.py
../.venv/bin/python ../launch_studio.py
```

For a clone outside the YuE2 folder, use `install_studio.py --target /path/to/YuE2`,
then run `/path/to/YuE2/launch_studio.py` with its environment. Windows launchers
assume the standard nested layout. `--check` validates without installing.

## 3. Check your paths, then create

In **Advanced settings**, select your model and VAE paths and a GPU memory budget
appropriate to your hardware. Default folders are `models/YuE2-3B` and `models/YuE2-Vae`
under YOUR YuE2 root. If you used upstream's Hugging Face cache instead, enter the
Hub IDs `m-a-p/YuE2-3B` and `m-a-p/YuE2-Vae`; disable Offline model loading if downloads
are needed. Paths are resolved locally; this repository includes no weights.

- **New song:** enter style and section-tagged lyrics, then create music.
- **Style trainer:** check setup, select local songs and a shared style, save the
  setup, then explicitly queue training. Successful adapters appear in the creation
  area's Style / Artist LoRA list. Full PyTorch weights and a BF16 NVIDIA GPU are required;
  LoRAs do not work with GGUF. Missing weights can be downloaded through the trainer.
- **Artist trainer:** prepare its separate runtime/model paths, review matching
  full-song lyric files, save a setup and explicitly queue training. Defaults are
  500 steps and save every 250. Playback loads your Artist adapter and its pinned
  companion together. [Walkthrough](docs/artist-trainer.md).
- **Writing room:** choose an LLM provider, refresh its model list, enter your own
  API key or local endpoint, and generate/edit lyrics and styles.
- **Surprise me:** choose batch size, vocal gender, style, language and optional
  required profanity. Songs are written and rendered sequentially.
- **Covers:** use your own ABC, or configure the separate SheetSage2 environment
  and models using the [upstream cover guide](https://github.com/multimodal-art-projection/YuE/blob/main/docs/covers.md).
  Review the transcribed melody before rendering. Covers do not clone a singer.
- **Lyrics export:** copy section-tagged lyrics or download TXT and song-details JSON for your video workflow, from drafts, the editor, or saved runs.
- **Progress and library:** watch stages, token speed, and synthesis steps; play or
  download completed songs and inspect saved settings and logs.

## Choose your music engine

> **GGUF requires a separate audio.cpp installation.** Studio can download GGUF
> model files, but it **does not install or build the audio.cpp engine**. Install
> a Yue2-capable audio.cpp executable outside Studio, then connect it in the model
> panel. Downloading a model alone is not enough to generate with GGUF.
> **Torch users do not need audio.cpp.** [Engine installation instructions](docs/gguf.md#1-obtain-audiocpp-with-yue2-support).

Switch **Torch / GGUF** using **Music engine** above the song editor. Open
**Set up GGUF / Models** (also in the sidebar as **Music models**) to download Q4,
Q8, or BF16 bundles with their decoder and supporting files. The panel shows
sizes, progress, cancellation, verification, and automatic engine detection.
Click **Use model**, then create songs with your own lyrics/style or Surprise me.
Engine installation instructions are included; audio.cpp itself is installed
separately. [Setup and download guide](docs/gguf.md#easy-setup-from-the-main-page).

The main page also offers **GPU memory presets**: Auto, 8/12/16/24/32 GB, and
Custom. These set Torch budgets and offloading, and show separate GGUF model
suggestions. [Preset values and limits](docs/settings.md#gpu-memory-presets).

## Experimental GGUF / lower-VRAM engine

Studio also has an **audio.cpp** backend for [audio-cpp/Yue2-3B-GGUF](https://huggingface.co/audio-cpp/Yue2-3B-GGUF).
It requires a separate Yue2-capable audio.cpp executable, a main GGUF, a VAE GGUF,
and four sidecars. Q8 + F16 VAE is the default GGUF combination; Q4 is selectable.
Select **Advanced settings → Models & runtime → Inference backend → audio.cpp**, then
configure the **audio.cpp / GGUF** group. `torch` stays the overall default.

Original songs, supplied-ABC covers, Surprise me and audio downloads use the same UI.
Plan-only output and generated ABC export currently require torch. GGUF outputs are
marked for review because the CLI does not return structured truncation flags.
Q8 + F16 CUDA generation has been tested on Windows with an RTX 5090; Q4 and other
devices remain unvalidated locally. See the [setup, test steps, limitations and
memory guide](docs/gguf.md) for the measured result and its limits.

## Speed defaults

`torch` CUDA graphs, no weight quantization, AR offloading disabled, optional model
hash verification disabled, and transcription tensor exports disabled. Baseline
sampling and 32 synthesis steps remain intact. Hash verification is a provenance
tradeoff; fewer synthesis steps and shorter token limits change the result, so they
are documented options rather than hidden speed tricks. Existing saved drafts
retain their settings; use Restore defaults in Advanced settings to adopt new defaults.

For ordinary generation memory failures, **Offload autoregressive model** trades
weight-transfer time for lower synthesis memory use. Keep `torch` enabled.
**Artist LoRAs require AR offloading disabled**; free competing GPU workloads and
check the supported settings instead of enabling that option for an Artist bundle.

**Same seed does not guarantee the same song.** No score/Torch baseline-only tests
varied with identical seed, style, lyrics and settings. The exact cause is not yet
established. Retain original outputs and compare multiple baseline/LoRA pairs;
see [comparison guidance](docs/studio.md#seeds-and-comparing-results).

## Updates and shutdown

Finish your jobs and run **Stop Yue2 Studio.bat**, then:

```powershell
# Inside the Yue2_Studio clone:
git pull
.\"Start Yue2 Studio.bat"
```

The launcher installs changed files with backups. Closing the last browser tab
stops an idle server after about 15 seconds; active jobs and batches finish first.
Refreshing or another open tab keeps it running. Stop explicitly before updating.
Your runs stay under the parent `runs/studio/`; keys are entered per browser session.

## Documentation

- [Full workflow guide](docs/studio.md)
- [Every advanced setting, defaults, performance and troubleshooting](docs/settings.md)
- [Style Trainer](docs/trainer.md) · [Artist Trainer](docs/artist-trainer.md) · [LoRA playback](docs/lora.md)
- [Official YuE2 music skill and formatting](https://github.com/multimodal-art-projection/YuE/tree/main/skills/yue2-music)

## License and credits

Studio code is distributed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md) for upstream attribution and modifications.
Model weights are not included and have their own [model license](MODEL_LICENSE).
This is a community UI, not an official upstream release. LLM services use recipients'
own accounts and provider terms.

This repository contains no songs, uploads, API keys, caches or Python environments.
The server is local-only; cloning gives each user their own studio, not access to yours.

## See the Studio

Some images show earlier layouts. The new trainer, LoRA, library and appearance
captures are appended below, with captions explaining workspace-only details.

| Create a song | Create a cover |
| --- | --- |
| ![Song editor with style and lyrics](images/new-song.png) | ![Cover source upload and melody transcription](images/create-cover.png) |

| LLM Runner | Advanced settings |
| --- | --- |
| ![Provider and model selection](images/llm-runner.png) | ![Searchable engine settings](images/advanced-settings.png) |

| Writing room | Review your draft | Surprise me |
| --- | --- | --- |
| ![Describe a song idea](images/writing-room.png) | ![Edit and apply generated lyrics](images/review-draft.png) | ![Automatic song batches](images/surprise-me.png) |

### Engine and memory controls

![Main music engine selector](images/music-engine.png)

![GGUF setup and model downloads button](images/gguf-setup-button.png)

![GPU memory preset with the effective budget](images/gpu-memory-preset.png)

Click any screenshot to see it full size. The [illustrated guide](docs/studio.md) walks through each screen.

### Trainers, library and experimental controls

These screenshots show the author's workspace. Example paths/project names are
not bundled datasets. The recovery shortcuts and historical verification banner
shown in that workspace are not release controls/status. Artist defaults are now
**500 steps / save every 250**, not the older 800/200 shown below.

| Style dataset | Style training controls |
| --- | --- |
| ![Choose songs and a clip length for Style training](images/style-trainer-dataset.png) | ![Style training steps, rank, learning rate and checkpoints](images/style-trainer-controls.png) |

![Reopen a saved training setup](images/saved-training-projects.png)

| Artist dataset and lyrics | Artist training controls |
| --- | --- |
| ![Artist song folder, lyric pairing, trigger and shared style](images/artist-trainer-dataset.png) | ![Artist training controls; screenshot values are not the current defaults](images/artist-trainer-controls.png) |

![Style or Artist LoRA selection](images/lora-selector.png)

![Shape the generation experimental sliders](images/shape-generation.png)

Composition, Performance and Style influence map to existing advanced settings.
Click **?** for explanations and cautions. The current UI corrects the screenshot's
comparison hint: **same seed does not guarantee identical music**.
[Slider settings and limits](docs/settings.md#shape-the-generation).

![Song library with stars, title editing, search and date filters](images/song-library.png)

![Studio appearance presets](images/appearance-presets.png)
