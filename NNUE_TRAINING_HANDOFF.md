# Deep Blue NNUE — complete training manual (Google Colab, free T4 GPU)

You don't need to know anything about chess engines or machine learning.
Every box below is copy-paste. Total time: **about 1 hour**, most of it waiting.

**What you're doing:** Deep Blue is a chess engine. It currently judges chess
positions using hand-written rules, and those have stopped getting better.
You're going to train a small neural network to judge positions instead, using
11.4 million positions that are **already prepared and labelled**. You don't
collect data, write code, or tune anything. You run five cells and send back
one file.

---

# Before you start — two things Yusuf must do

1. **Add you as a collaborator** on the GitHub repo
   `ysyed2206/deepblue-teammate-lean` (it's private, so you can't see it
   otherwise). You'll get an email invite — accept it.
2. Nothing else. The data is already uploaded and waiting.

# Before you start — one thing you must do

**Create a GitHub access token.** The repo is private, so Colab needs a token
to read it.

1. Go to **github.com/settings/tokens** → *Tokens (classic)* → *Generate new
   token (classic)*
2. Note: `colab`. Expiration: 7 days.
3. Tick the single checkbox **`repo`**.
4. *Generate token*, then **copy it** — it starts `ghp_...` and is shown only
   once.

Keep it in your clipboard or a notepad. Don't paste it into a chat or commit
it anywhere.

---

# Step 1 — Open Colab and turn on the GPU

1. Go to **colab.research.google.com**
2. *New notebook*
3. Menu: **Runtime → Change runtime type → Hardware accelerator → T4 GPU →
   Save**

**This step is the whole point — skip it and everything runs ~10x slower.**

Paste this into the first cell and press **Shift+Enter** to run it:

```python
!nvidia-smi
```

You should see a table mentioning **Tesla T4**. If you see
`command not found` or no table, the GPU isn't on — redo Runtime → Change
runtime type.

---

# Step 2 — Log in and get the code

New cell (press `+ Code`), paste, run. It will prompt you for the token —
paste it and press Enter. The token won't appear on screen; that's normal.

```python
from getpass import getpass
TOKEN = getpass("Paste your GitHub token: ").strip()

!git clone https://{TOKEN}@github.com/ysyed2206/deepblue-teammate-lean.git
%cd deepblue-teammate-lean
!mkdir -p data out
print("done")
```

If it says `Repository not found`, you haven't accepted the collaborator
invite yet, or the token is missing the `repo` checkbox.

---

# Step 3 — Download the training data (1.6 GB, takes a few minutes)

New cell, paste, run:

```python
REPO = "ysyed2206/deepblue-teammate-lean"
ASSETS = {
    "data/nnue_train_11.4M.npz":      "RA_kwDOUNwM8s4hIUPJ",
    "data/nnue_validation_11.4M.npz": "RA_kwDOUNwM8s4hITnP",
}
for path, asset_id in ASSETS.items():
    print("downloading", path)
    !curl -sL -H "Authorization: token {TOKEN}" \
          -H "Accept: application/octet-stream" \
          https://api.github.com/repos/{REPO}/releases/assets/{asset_id} -o {path}

!ls -lh data
```

You want to see roughly **1.5G** and **80M**. If either shows a few hundred
bytes, the download failed — it saved an error message instead. Re-check the
token and run the cell again.

---

# Step 4 — Train (this is the long one, 20–40 minutes)

New cell, paste, run. **Leave the tab open and don't let your computer sleep** —
Colab disconnects idle sessions and you'd lose the run.

```python
!python nnue_lab/train.py \
  --train-data data/nnue_train_11.4M.npz \
  --validation-data data/nnue_validation_11.4M.npz \
  --width 256 \
  --epochs 6 \
  --batch-size 8192 \
  --device cuda \
  --checkpoint out/net.pt \
  --log out/train.jsonl \
  --summary out/summary.json
```

### What good looks like

One line per epoch. The number that matters is **`validation_teacher_mae_cp`**
and it must go **DOWN**. From a real earlier run:

```
epoch 1   validation_teacher_mae_cp  144.1
epoch 2   validation_teacher_mae_cp  129.7
...
epoch 8   validation_teacher_mae_cp  114.8
```

Lower = the network is judging positions more like a strong engine. If it
rises two epochs in a row the script stops itself — that's a built-in safety
feature, not an error, and the best checkpoint is already saved.

### If it fails

**Error mentioning `sparse` or `SparseAdam`:** rerun the identical cell but
change `--device cuda` to `--device cpu`. Slower (2–3 hours) but it works.
Tell Yusuf this happened.

**`CUDA out of memory`:** change `--batch-size 8192` to `--batch-size 4096`.

**Session disconnected:** you have to start again from Step 2. Colab's free
tier does this if you go idle.

---

# Step 5 — Send the result back

New cell, paste, run:

```python
from google.colab import files
files.download("out/net.pt")
files.download("out/summary.json")
```

Send Yusuf **both files**. That's the job done.

**Do not run `quantize_scratch.py`** — that conversion step is done on Yusuf's
machine because it cross-checks the result against the engine's own loader.

---

# Optional — a second, bigger run

If that went smoothly and you still have time, ask Yusuf for the **28.5M**
dataset and run again with `--width 512 --epochs 4`. Send back both
checkpoints; they get tested against each other in real games and the better
one ships.

`--width` is the network size. Allowed values are **128, 256, 512** only —
anything else is rejected. Bigger is stronger but slower, both to train and
inside the engine.

---

# Two rules that really matter

**1. Train from scratch. Never start from someone else's network.**
The competition organiser was explicit: the weights must come from training
*you* started. Initialising from a released net and fine-tuning it counts as
shipping that net and would disqualify the team. `train.py` starts from random
values by default — so just **don't add any flag that loads an existing
checkpoint** and you're automatically fine.

**2. Don't change the architecture.**
The engine expects 768 inputs, 8 king buckets, and fixed quantisation
constants (QA 255, QB 64, scale 400). It checks these when loading and refuses
a mismatched file. `--width` is the only thing you should change.

---

# How long this really takes

From a real run in this repository: width 128, 424,253 positions, 8 epochs,
**273 seconds on a CPU** — about 12,400 positions/second. A T4 is several
times faster than that, and the bigger `--batch-size` above helps more.

| dataset | width | on a T4 | CPU fallback |
|---|---|---|---|
| 11.4M, 6 epochs | 256 | 20–40 min | 2–3 hours |
| 28.5M, 4 epochs | 512 | 1–2 hours | don't bother |

**A 5-hour window is comfortably enough**, even if the GPU path fails entirely
and you fall back to CPU.

---

# If something goes wrong

Send Yusuf: the exact cell you ran, the **full** error text (not a summary),
and `out/train.jsonl` if it exists. Don't try to fix the code — the failure
modes here are unobvious and the log says exactly what happened.
