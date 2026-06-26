# Unison — Project Page

A self-contained static project page for **Unison: Benchmarking Unified Multimodal
Models via Synergistic Understanding and Generation** (ICML 2026).

No build step, no dependencies — just HTML, one CSS file, and one JS file.

## Structure

```
Unison-ProjectPage/
├── index.html              # the page
├── serve.sh                # local preview server (auto-frees the port if busy)
└── static/
    ├── css/style.css       # design system (blue → rose gradient theme)
    ├── js/main.js          # nav, scroll-reveal, leaderboard tabs, copy, heatmap
    └── images/             # Unison-logo.svg (transparent) · Unison-logo.png · overview.png
```

## Preview locally

```bash
cd Unison-ProjectPage
./serve.sh            # serve on http://localhost:8000 (and open it)
./serve.sh 3000       # …or choose a port
```

`serve.sh` first checks whether the port is already in use; if it is, it stops
the process holding it and restarts cleanly — no "address already in use".
(Plain `python3 -m http.server` works too.)

## Logo

`static/images/Unison-logo.svg` is a transparent vector logo traced from the
original artwork, so it stays crisp at any size and sits cleanly on dark
backgrounds (the original `Unison-logo.png` had a baked-in white background).
The page uses the SVG everywhere; the PNG is kept only as a raster fallback.

## Deploy to GitHub Pages (`fudancvl.github.io/Unison`)

Push the **contents of this folder** to the repo / branch that serves the site,
then enable Pages on that source. For example, to serve from `/docs` on `main`:

```bash
# copy these files into <repo>/docs/ and enable Pages → main /docs
```

Or push to a `gh-pages` branch root. The page uses only relative paths, so it
works from any base URL.

## Before publishing — update these placeholders

- **arXiv link** — `index.html` uses `https://arxiv.org/abs/xxxxx` in the hero
  buttons and footer (mirrors the repo README). Replace with the real ID.
- **GitHub link** — set to `https://github.com/FudanCVL/Unison`. Adjust if the
  repo lives elsewhere.
- **Leaderboard** — numbers are transcribed from the repo `README.md`. Edit the
  `<table class="lb">` blocks in `index.html` to update scores or add models.
  `<strong>` marks the best value, `<u>` the second best; `class="na"` is a dash.
```
