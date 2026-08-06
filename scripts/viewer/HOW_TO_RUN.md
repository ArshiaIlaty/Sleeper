# How to run the Biosignal Viewer

The viewer is **not a hosted website** — it runs on the `pdmle` machine on demand
and is reached over an SSH tunnel. Nothing is exposed on the network or the
internet, so it stays inside IT policy. There are two roles:

- **Host** — one person starts the server on `pdmle`.
- **Viewer** — anyone with a `pdmle` account opens a tunnel and browses locally.

You only need to be the Host once; everyone else is a Viewer. If nobody is
hosting, the app is simply offline until someone runs step 1.

---

## 1. Host: start the server (do this once per session)

You need a `pdmle` account **in the `mlusers` group** (e.g.
`arshia_ilaty_physio26`) so the server can read the dataset. `ubuntu` cannot —
do **not** try to run it under `sudo`.

```bash
ssh your_account@AWOR-PDMLEAPP01          # log into pdmle
cd /data-temp/physio-viewer               # the deployed bundle
bash run.sh                               # serves on 127.0.0.1:8050
```

You should see:

```
Starting viewer on http://127.0.0.1:8050  (data: /data-temp/shared-physionet26-dataset/extracted)
Loaded 1103 patients from .../demographics.csv
Serving on http://127.0.0.1:8050  (Ctrl-C to stop)
```

Leave that terminal open — the server runs until you press **Ctrl-C**. To keep
it alive after you disconnect, run it inside `tmux` or `screen`:

```bash
tmux new -s viewer      # start a named session
bash run.sh             # ...run the server...
# detach with:  Ctrl-b then d      (reattach later with: tmux attach -t viewer)
```

**Run on a different port** (e.g. if 8050 is taken): `PORT=8061 bash run.sh`.

---

## 2. Viewer: connect from your own laptop

Open a tunnel **with your own `pdmle` account** — this forwards your laptop's
`localhost:8050` to the server's loopback port. Then open a normal browser.

```bash
ssh -L 8050:127.0.0.1:8050 your_account@AWOR-PDMLEAPP01
```

Leave that SSH session open, and in your browser go to:

```
http://127.0.0.1:8050
```

That's it. Multiple teammates can tunnel to the **same** running server at once
— each just runs the `ssh -L` line above with their own account.

> If the Host started on a custom port, match it on both sides, e.g.
> `ssh -L 8061:127.0.0.1:8061 your_account@AWOR-PDMLEAPP01` → `http://127.0.0.1:8061`.

---

## 3. One-liner (no tmux): tunnel + run in a single command

If you're both host and viewer and just want it up quickly, this opens the
tunnel and starts the server in one shot; when you Ctrl-C, both stop:

```bash
ssh -L 8050:127.0.0.1:8050 your_account@AWOR-PDMLEAPP01 \
    'cd /data-temp/physio-viewer && bash run.sh'
# then open http://127.0.0.1:8050 in your browser
```

---

## Choosing a dataset

Once the page is open, use the **Dataset** dropdown in the top-right to switch
between:

- **Standard (1,103)** — the main challenge release, read from local disk (fast).
- **Large (6,530)** — the extended release, streamed from S3. Demographics,
  staging, dynamics, and preprocessing are instant (small files); the raw PSG
  signals tab downloads that patient's full EDF (~150–450 MB) to a server cache
  on first view, so the first channel load takes a few seconds and later ones are
  instant. A note on the signals panel reminds you of this.

## Exporting a feature CSV (for training a model)

To produce the per-recording feature table your teammates can train on:

```bash
# on pdmle, as arshia_ilaty_physio26:
cd /data-temp/physio-viewer
python3 export_features.py --dataset standard --out exports/features_standard.csv
python3 export_features.py --dataset large    --out exports/features_large.csv --resume
```

Output lands in `/data-temp/physio-viewer/exports/`. The large export takes
~1–2 h (it streams 6,530 small CAISR files); run it inside `tmux` and pass
`--resume` so an interrupted run picks up where it left off. See `README.md` →
"Feature export" for the column list.

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `bash: run.sh: Permission denied` | The bundle files aren't readable by your account (owner/mode from a redeploy). Ask whoever deployed to run `sudo chmod -R a+rX /data-temp/physio-viewer` (and `chmod 755 run.sh`). |
| `ERROR: cannot read .../demographics.csv` | You're not in the `mlusers` group (e.g. logged in as `ubuntu`). Log in as a data-capable account like `arshia_ilaty_physio26`. |
| Browser shows "connection refused" | No server is running (nobody is hosting), or your tunnel port doesn't match the server port. Start the server (step 1) or fix the port. |
| `bind: Address already in use` (tunnel) | You already have a tunnel on that local port. Close the old SSH session, or pick another port on the **left** side: `ssh -L 8062:127.0.0.1:8050 ...` → open `http://127.0.0.1:8062`. |
| `Address already in use` (server) | Someone is already hosting on that port — you may not need to start your own. Otherwise use `PORT=8061 bash run.sh`. |
| Page loads but looks old (no new tabs/heatmaps) | A stale server is running old code. The host should Ctrl-C and restart `run.sh`; clear `__pycache__` if needed: `rm -rf /data-temp/physio-viewer/__pycache__`. |

---

## Why it works this way (IT-safe by design)

- The server binds to **`127.0.0.1` (loopback only)** — it is not reachable from
  the network, so no firewall exception or hosting approval is needed.
- Every teammate authenticates with **their own SSH credentials**; access rides
  on the permissions IT already granted for `pdmle`.
- The data (de-identified PhysioNet challenge recordings) **never leaves the
  machine** — only downsampled traces and summary numbers cross the tunnel to
  your browser.
- **Do not** use public tunnels (ngrok/cloudflared), bind to `0.0.0.0`, or share
  accounts — those would expose unauthenticated patient data and violate the
  data-use agreement. See the "Sharing" note in `README.md`.
