# Biosignal Viewer — PhysioNet Challenge 2026

An interactive, single-page web app to browse the dataset one patient at a time:

- **Demographics** panel (age, sex, race, BMI, follow-up fields, CI label)
- **CAISR sleep staging** — hypnogram, stage-% bar, and event indices (AHI, arousal, PLMI, apnea/hypopnea/RERA)
- **PSG signals** — pick any channels; the server downsamples each to ~2500 points before sending, so the 170 MB EDFs never hit the browser
- **Hover explanations** — every event index, sleep stage, channel, and
  demographic field shows a plain-language tooltip on hover (or keyboard focus).
  Definitions live in `glossary.py` (one source of truth), served at
  `/api/glossary`.
- **Edwards branding** — the logo (`edwards_logo.png`, served from `/static/`)
  sits in the header.

Pure Python **standard library** (`http.server`) + `numpy` + `edfio` — **no Flask,
no CDN, no external network**, so it cannot touch the company firewall. The
frontend is one dependency-free `index.html` (vanilla JS + inline SVG).

## Files

| File | Purpose |
|---|---|
| `app.py` | HTTP server + JSON API (`/api/patients`, `/api/demographics`, `/api/caisr`, `/api/signals`, `/api/glossary`, `/static/*`) |
| `glossary.py` | Plain-language definitions of every metric, stage, channel role, and field (drives the hover tooltips) |
| `index.html` | Single-page UI (vanilla JS + SVG, validated Edwards palette, tooltip engine) |
| `edwards_logo.png` | Header logo, served from `/static/` (no CDN) |
| `run.sh` | Launch helper with a data-readability pre-check |

## Running (on pdmle)

Run as an account **in the `mlusers` group** (e.g. `arshia_ilaty_physio26`) — it
reads the dataset directly; `ubuntu` cannot and must not use sudo for a server.

```bash
# on pdmle, as your data-capable account:
cd ~/viewer          # or wherever you copied scripts/viewer/
bash run.sh          # -> http://127.0.0.1:8050
```

From your laptop, tunnel the port over SSH and open a browser locally (the app
binds to 127.0.0.1, so it is only reachable through the tunnel — nothing is
exposed on the network):

```bash
ssh -L 8050:127.0.0.1:8050 arshia_ilaty_physio26@AWOR-PDMLEAPP01
# then, in your laptop browser:
open http://127.0.0.1:8050
```

## Configuration

- `PHYSIONET_DATA_ROOT` — dataset root (default `/data-temp/shared-physionet26-dataset/extracted`)
- `PORT` / `HOST` — bind address (default `127.0.0.1:8050`)
- `POINTS` in `app.py` — max points/channel sent to the browser (default 2500)

## Notes

- Signal decimation keeps a min/max envelope per bucket, so spikes/artifacts
  survive downsampling rather than being averaged away.
- The channel-list call is header-only; sample data is read only for the
  channels you actually select.
- Hypnogram "Unknown" (code 9) epochs render as gaps, not a spurious low stage.
