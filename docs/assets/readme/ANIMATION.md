# MPSL training animation

The README media renders the existing [MPSL interactive walkthrough](https://nousphera.github.io/projects/mpsl/) into six five-second scenes. Artwork and scientific content come from `scripts/stories/mpsl_round.py`, `assets/mpsl-story.css`, and the generated MPSL project page in the Nousphera website source. Capture changes only presentation: larger labels, bounded dimensions, responsive layout, and compressed reveal timings. It does not replace the method with a new diagram.

The sequence follows [arXiv:2502.06355v3, Figure 1 and Section 3.2](https://arxiv.org/pdf/2502.06355v3): local modality tokenizers → parallel activation uploads → one shared encoder with late fusion → predictions returned to clients → local loss uploads → batch-size-weighted server loss and backward pass → cut-layer gradients returned → local tokenizer updates. Image and text keep their colors throughout. Client A and B retain their identities. The two- and three-object inputs denote samples, not batch sizes or measured predictions. The paper's post-training model construction is outside this training-step animation.

| Asset | Dimensions | Duration | Use |
|---|---:|---:|---|
| `method-light.gif`, `method-dark.gif` | 960 × 780 | 30 s | Desktop README |
| `method-light-mobile.gif`, `method-dark-mobile.gif` | 840 × 1640 | 30 s | Phone README, captured at 420 CSS px / 2× density |
| `method-light.mp4`, `method-dark.mp4` | 960 × 780 | 30 s | Downloadable video |

All exports contain 300 frames at 10 fps. GIFs loop indefinitely. Exact sizes, hashes, and encoding dimensions are recorded in `animation-manifest.json`. The narrow GIFs reflow the diagrams instead of shrinking the desktop layout. Put narrow theme sources before desktop theme sources in the README's `<picture>` element.

## Reproduce

Requirements: Node.js, the `playwright` package and a Chromium installation, ffmpeg. Serve the website checkout at a local URL, then run from this repository:

```sh
node docs/assets/readme/render-animation.cjs --url http://127.0.0.1:60664/projects/mpsl/
```

Optional environment variables: `CHROME_PATH` selects a browser executable; `FFMPEG` selects ffmpeg; `NODE_PATH` selects an existing Playwright installation. `--preview` renders six completed scenes into a temporary directory. `--layout desktop|mobile` and `--theme light|dark` limit the export. The script launches an isolated headless context and does not edit the website.

The native reveal sequence and CSS microanimations are sampled deterministically. Each scene lasts five seconds, with a reading hold after its final reveal. Global palettes and delta frames keep each GIF below 1.2 MB. MP4s use H.264, yuv420p, and fast-start metadata.

## Verification

Rendered frames were inspected in both themes and all six stages. Phone exports were checked at 360 px display width. Browser checks found no horizontal overflow or clipped artwork. `ffprobe` and Pillow confirmed dimensions, all 300 frames, 30-second duration, and infinite GIF loops. The website's five MPSL semantic tests passed, including local-only labels, transfer directions, batch weighting, and the shared server computation.

Source fingerprints at export:

```text
603260374f4a429dbfbe77d71232462a7597b772a557df73c798d71706290e4f  scripts/stories/mpsl_round.py
87d10eb8dfe594c77f23ba24aa5b370b24d46e4b51f986bab7322a8a19e3fd21  assets/mpsl-story.css
10aea792b273d6c68d2cb52b065f7de59f7e96704547fd3b1957f36f265534a8  assets/paper-story.js
f7551dd7a5750a061cdd59d428d957fa4e5a67195d9540c6bde7c6cf3001fca8  projects/mpsl/index.html
```
