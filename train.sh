#!/bin/sh
# Orus-OCR-1.0-alpha: full training recipe, in the order it was run (see docs/TRAINING.md for what each stage does).
#
#   ./train.sh <stage> [<stage> ...]
#
#   data32   page manifests for every source + 32 px line stores          -> data/ocr/pages, data/ocr/rec
#   synth32  word bank + synthetic 32 px lines (font, stitched words)       -> data/ocr/words, data/ocr/rec
#   lm       character 5-gram LM                                          -> models/orus_lm.npz
#   lex      word lexicon (BN + EN)                                       -> models/orus_lex.txt.gz
#   A B1 B3 C   conv-only 32 px recognizer from Indra OCR's rec.onnx (this is Orus-OCR-0.1-alpha)
#   D        C + 2-layer BiLSTM 128, 32 px
#   data48   48 px line stores + all synthetic kinds                      -> data/ocr/rec48
#   S1       D adapted to 48 px (stem / row prior resized), short warm-up
#   main     the 1.0 run: 70k steps
#   ext      extension: 35k steps at a lower LR and a larger batch        -> runs/orus_rec_10b.pt (= released weights)
#   export   fp32 ONNX                                                    -> models/orus_rec.onnx
#   bench    line benchmark (makes the test sets first)                   -> runs/bench_*.json
#
# Raw datasets go in data/raw/ and data/bench_raw/ (docs/DATASETS.md). Training needs a CUDA GPU (12 GB is enough);
# everything else runs on CPU. Peak VRAM of the extension stage is ~8 GB.
set -e
cd "$(dirname "$0")"
P=${PYTHON:-python}
T="$P -u orus/ocr_train.py rec --init indra"
mkdir -p runs models

for stage in "$@"; do
  echo "== stage $stage ($(date +%H:%M))"
  case $stage in
    data32)
      $P -u orus/ocr_data.py pages
      $P -u orus/ocr_data.py crops ;;
    synth32)
      $P -u orus/ocr_synth.py words
      for k in font stitch; do
        $P -u orus/ocr_synth.py lines $k 3000 val 0
        $P -u orus/ocr_synth.py lines $k 5000 test 0
      done
      for s in 0 1 2 3 4; do $P -u orus/ocr_synth.py lines font 50000 train $s; done
      for s in 0 1 2 3; do $P -u orus/ocr_synth.py lines stitch 50000 train $s; done ;;
    lm)  $P -u orus/ocr_lm.py build 5 6e6 ;;
    lex) $P -u orus/ocr_words.py build ;;
    A)   $T --name orus_rec_a --steps 30000 --lr 1.5e-3 --eval_every 3000 --workers 3 2>&1 | tee runs/orus_rec_a.log ;;
    B1)  $T --name orus_rec_b1 --resume runs/orus_rec_a.pt --steps 40000 --lr 8e-4 --eval_every 4000 --workers 3 2>&1 | tee runs/orus_rec_b1.log ;;
    B3)  $T --name orus_rec_b3 --resume runs/orus_rec_b1.pt --steps 30000 --lr 4e-4 --eval_every 5000 --workers 2 2>&1 | tee runs/orus_rec_b3.log ;;
    C)   $T --name orus_rec_c --resume runs/orus_rec_b3.pt --steps 12000 --lr 3e-4 --eval_every 3000 --workers 3 2>&1 | tee runs/orus_rec_c.log ;;
    D)   $T --name orus_rec_d --resume runs/orus_rec_c.pt --rnn 128 --steps 30000 --lr 8e-4 --eval_every 5000 --workers 4 2>&1 | tee runs/orus_rec_d.log ;;
    data48)
      export ORUS_H=48
      $P -u orus/ocr_data.py crops
      for k in font stitch; do
        $P -u orus/ocr_synth.py lines $k 3000 val 0
        $P -u orus/ocr_synth.py lines $k 5000 test 0
      done
      for s in 0 1 2 3 4 5 6 7 8 9; do $P -u orus/ocr_synth.py lines font 15000 train $s; $P -u orus/ocr_synth.py lines stitch 15000 train $s; done
      for s in 10 11 12 13; do $P -u orus/ocr_synth.py lines font 15000 train $s; done
      for k in num receipt rx book hdigit; do $P -u orus/ocr_synth.py lines $k 1000 val 0; done
      for s in 0 1; do for k in num receipt book; do $P -u orus/ocr_synth.py lines $k 15000 train $s; done; done
      for k in rx hdigit; do $P -u orus/ocr_synth.py lines $k 15000 train 0; done ;;
    S1)   $T --name orus_rec_10s1 --resume runs/orus_rec_d.pt --rnn 128 --height 48 --steps 25000 --lr 8e-4 --eval_every 5000 --workers 4 2>&1 | tee runs/orus_rec_10s1.log ;;
          # the released lineage stopped S1 after its first evaluation (5k steps); runs/orus_rec_10s1.pt is that checkpoint
    main) $T --name orus_rec_10 --resume runs/orus_rec_10s1.pt --rnn 128 --height 48 --steps 70000 --lr 8e-4 --eval_every 5000 --workers 5 --area 2 2>&1 | tee runs/orus_rec_10.log ;;
    ext)  $T --name orus_rec_10b --resume runs/orus_rec_10.pt --rnn 128 --height 48 --steps 35000 --lr 3.5e-4 --eval_every 5000 --workers 5 --area 3 2>&1 | tee runs/orus_rec_10b.log ;;
    export) $P orus/ocr.py export runs/orus_rec_10b.pt models/orus_rec.onnx ;;
    bench)
      $P -u orus/ocr_bench.py make
      BENCH_TTA=0.8 BENCH_TAG=_tta $P -u orus/ocr_bench.py run orus models/orus_rec.onnx
      BENCH_LEX=0 BENCH_TAG=_fast $P -u orus/ocr_bench.py run orus models/orus_rec.onnx ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
