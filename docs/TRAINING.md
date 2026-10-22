# Training

How the released recognizer was trained, on what hardware, and how long it took. The commands are in [`train.sh`](../train.sh); the data in [DATASETS.md](DATASETS.md).

## Model

| | |
|---|---|
| Input | a text line as a 48 px high "ink map" (1 = ink), with two neighbouring line heights of context above and below (144 px tall), any width |
| Network | Indra OCR 1.1-style conv CTC recognizer (full-height stem, tall-window context branch, 10 depthwise-separable conv blocks with masked squeeze-excite, width 128) + 2-layer bidirectional LSTM (hidden 128) as a zero-initialised residual + CTC head |
| Output | 212 classes per 4 px column: CTC blank + 211 characters (printable ASCII, the Bengali Unicode block, typographic marks) |
| Parameters | 1.76M |
| Loss | CTC |

The BiLSTM is added as a residual whose output projection starts at zero, so a model warm-started from the conv-only checkpoint begins exactly where that checkpoint was. Moving from 32 px to 48 px resizes the stem kernel and the context row prior of the 32 px checkpoint (`adapt` in `orus/ocr_model.py`).

## Training lineage

The released weights are the last step of one chain of warm starts. Every stage ran `orus/ocr_train.py` with AdamW (weight decay 0.01), a OneCycle schedule (5% warm-up, cosine decay), bf16 autocast, gradient clipping at 2.0, width-bucketed batches and GPU augmentation (small rotation / scale / shear, stroke thickening / thinning, blur, contrast, noise). The best checkpoint by mean validation CER on the real sets was kept.

| Stage | Starts from | Line height | Steps | Peak LR | Batch | Wall-clock | Val mean real CER |
|---|---|---|---|---|---|---|---|
| A | Indra OCR 1.1 `rec.onnx` weights | 32 px | 30,000 | 1.5e-3 | 1× | 46 min | 16.0% |
| B1 | A, + IIIT words and stitched lines | 32 px | 40,000 | 8e-4 | 1× | 31 min | 15.0% |
| B3 | B1 | 32 px | 30,000 | 4e-4 | 1× | 44 min | 14.7% |
| C (= 0.1-alpha) | B3, + teacher-labelled lines | 32 px | 12,000 | 3e-4 | 1× | 10 min | 14.65% |
| D | C, + BiLSTM 128 | 32 px | 30,000 | 8e-4 | 1× | 1 h 39 min | 14.0% |
| S1 | D adapted to 48 px, full 1.0 data mix | 48 px | 5,000 (stopped after the first evaluation) | 8e-4 | 1× | 24 min | 15.0% |
| Main | S1 | 48 px | 70,000 | 8e-4 | 2× | 5 h 46 min | 9.62% |
| Extension (**released**) | Main | 48 px | 35,000 | 3.5e-4 | 3× | 3 h 16 min | **9.34%** |
| **Total** | | | **252,000** | | | **≈ 13.3 GPU hours** | |

"Batch 1×" holds up to 96 lines and fewer when the lines are wide (a fixed pixel budget per batch); 2× and 3× scale both limits. "Val mean real CER" is greedy decoding on the validation splits of the real sources; it is not comparable across the 32 px and 48 px rows because the 48 px stages validate on more (and harder) sources. Training sampling is not seeded, so a re-run lands within a few tenths of a point.

Stage A–C data was the 0.1-alpha mix (the handwriting and printed sources plus 32 px synthetic font and stitched lines). From S1 on, every source in [DATASETS.md](DATASETS.md) was used with the weights listed there.

Checkpoint averaging of the last three evaluations (25k / 30k / 35k steps) gave 9.35%, no gain, so the final checkpoint is released as is.

## Hardware and resource use

| | |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 3060, 12 GB |
| CPU | AMD Ryzen 5 3500X, 6 cores / 6 threads |
| RAM | 16 GB |
| Storage | SATA SSD for the line stores (8.5 GB at 48 px), HDD for the raw datasets |
| Software | Python 3.10, PyTorch 2.14 (CUDA 12.6), ONNX Runtime 1.23, Linux |

- **GPU memory:** the extension stage (BiLSTM, 48 px, 3× batch) peaks at **8.1 GB reserved** (3.8 GB allocated), measured with `torch.cuda.max_memory_reserved` over a 400-step re-run of that configuration. The 32 px stages use less. A 12 GB card is enough for every stage; at 8 GB use `--area 2`.
- **Throughput:** about 3.4 steps/s in the main stage and 3.0 steps/s in the extension stage (larger batches). Data loading (PNG decode and batching in 4–5 worker processes) is the bottleneck, not the GPU: the CPU sits at 4–5 busy cores during training.
- **RAM:** each training run was kept inside a 7 GB memory limit (cgroup) and stayed within it, including the page cache of the memory-mapped line stores; the stores are read through `mmap`, so RAM use does not grow with the data size.
- **CPU-only steps (not counted above):** building the line stores and synthetic lines took several hours of CPU time on the 6 cores (estimate: 8–10 hours wall-clock in total, run alongside training); the character LM and the lexicon take a few minutes each; once the test sets are built, benchmarking all 34 sets takes a few minutes per decoding mode on CPU.
- **Not included in the 13.3 GPU hours:** Indra OCR's own pre-training (we start from its released weights), a 1-hour size-sweep run that was not kept, and the small picture-region net and handwritten-flag classifier carried over unchanged from 0.1-alpha (minutes each).

## Reproducing

```bash
pip install -r requirements-train.txt
# put the raw datasets under data/raw/ and data/bench_raw/ (see docs/DATASETS.md and the src_* functions in orus/ocr_data.py)
./train.sh data32 synth32 lm lex A B1 B3 C D data48 S1 main ext export bench
```

Outputs go to `runs/` (checkpoints, logs, benchmark JSON) and `models/` (LM, lexicon, ONNX). The `data` stage needs Indra OCR's detector for pairing printed pages with their text; it is downloaded automatically by the `indra-ocr` package.

The teacher-labelled `distill` store (500 lines, 2% of batches) is not reproduced by these scripts; leaving it out has a negligible effect (it moved the 0.1-alpha validation CER by less than 0.1 point).

## Fine-tuning on your own data

`orus/ocr_finetune.py` continues training from the released `orus_rec.pt` (on Hugging Face) on your own line stores, at a low learning rate, with optional grid-warp augmentation (`--warp`) and hard-example oversampling (`--hard`):

```bash
python orus/ocr_finetune.py --name my_ft --resume orus_rec.pt --w my_mix.json --steps 8000 --lr 2e-5
python orus/ocr.py export runs/my_ft.pt models/my_rec.onnx
```

`my_mix.json` maps store names to batch shares, e.g. `{"mydata": 0.5, "bnhtrd": 0.1, "iam": 0.1, "synth_font": 0.1, ...}`. Keep a share of the original sources in the mix, otherwise the model forgets what it is not shown. In our own experiments, restarting at a learning rate of 2e-4 knocked the model out of its converged state and made it worse on every validation set; 2e-5 is a safe starting point.
