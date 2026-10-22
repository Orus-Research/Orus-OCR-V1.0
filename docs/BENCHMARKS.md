# Benchmarks

All numbers are for the released weights (`orus_rec.onnx`, Orus-OCR-1.0-alpha) unless a column says otherwise.

## Protocol

- **Line level.** 300 lines per test set (a seeded random sample of the test split; `orus/ocr_bench.py`). Each line is cut from its page with the same crop rule the pipeline uses, so detection is not part of these numbers.
- **CER** = character edit distance / reference length, per Unicode code point after NFC normalisation (per grapheme cluster, Bangla CER would be lower). **WER** = word edit distance / reference words. Case and punctuation count. Two set-specific rules: CORD dot leaders (`.....`) are ignored on both sides because the labels drop them, and CROHME is scored without spaces because its labels have none.
- **Decoding.** *Shipped* = the default pipeline: greedy CTC, character 5-gram LM on unsure lines, word lexicon, and a re-read of lines with confidence < 0.8 at 0.8× and 1.25× width. *Fast* = greedy CTC + LM only (`--fast`, `OCR(tta=None, lex=False)`).
- **Synthetic test sets** use held-out font families, the test split of the handwritten word bank and the MNIST test digits; none of these fonts, writers or digits were seen in training.
- **Hardware.** CPU only: AMD Ryzen 5 3500X (6 cores), ONNX Runtime 1.23, 4 threads.

## Summary by criterion

Each criterion is the plain mean over its sets (`CRITERIA` in `orus/ocr_bench.py`). Lower is better.

| Criterion | Sets | 1.0-alpha shipped<br>CER / WER % | 1.0-alpha fast<br>CER / WER % | 0.1-alpha<br>CER / WER % |
|---|---|---|---|---|
| Printed Bangla | bnprinted, swapnil, bench1000, print_bn | **4.1 / 10.9** | 4.6 / 12.8 | 8.3 / 24.2 |
| Printed English | print_en, sroie, cord_en, funsd | **3.0 / 11.6** | 3.6 / 13.4 | 19.1 / 40.3 |
| Handwritten Bangla | bnhtrd, banglawriting, iiit, held-out BN | **7.1 / 22.6** | 8.5 / 29.7 | 15.9 / 43.7 |
| Handwritten English | iam, gnhk, imgur5k | **9.5 / 25.6** | 11.1 / 31.7 | 15.0 / 38.3 |
| Unseen fonts | print_bn, print_en, font_bn, font_en | **2.0 / 5.0** | 2.5 / 7.4 | 4.8 / 13.3 |
| Receipts | sroie, cord_en, receipt_bn, receipt_en | **3.5 / 9.1** | 3.6 / 10.0 | 21.9 / 50.6 |
| Prescriptions | prescription, rx_print | **2.7 / 5.5** | 3.7 / 8.8 | 4.9 / 15.2 |
| Digits | digits_bn, digits_en, hdigits, textocr_num | **10.6 / 17.0** | 11.2 / 17.7 | 27.7 / 54.3 |
| Arithmetic | arith_bn, arith_en, harith, crohme | **5.6** / 18.4 | 5.8 / **18.0** | 18.6 / 53.2 |
| Noisy images | noisy, gnhk, imgur5k, textocr_num | **16.3 / 30.8** | 17.9 / 35.0 | 30.0 / 48.7 |
| Books / newspapers | swapnil, book_bn, book_en | **3.8 / 8.5** | 4.3 / 10.9 | 7.3 / 20.4 |
| Mixed Bangla + English | mixed | **10.3 / 24.7** | 11.8 / 30.9 | 16.7 / 40.9 |

0.1-alpha is shown as it shipped (greedy + char LM). Against it, 1.0-alpha halves the handwriting CER and is 4–7× better on printed English, receipts and digits.

### Against our targets

We set ourselves the targets below for 1.0. **None is met at the criterion mean**; we report them so you can judge the gap.

| Target (CER / WER %) | 1.0-alpha | Met? |
|---|---|---|
| Printed Bangla < 2 / ≤ 5 | 4.1 / 10.9 | No. Clean book lines (swapnil 0.9 / 6.1) and unseen print fonts (print_bn 1.9 / 5.8) meet it; phone photos of printed Bangla with noisy labels (bnprinted 11.1 / 23.1) do not. |
| Printed English < 1 / ≤ 3 | 3.0 / 11.6 | No. Unseen print fonts (print_en 0.9 / 2.3) meet it; faint scanned forms (funsd 6.2 / 23.2) and receipts do not. |
| Handwritten Bangla ≈ 5 / ≈ 8 | 7.1 / 22.6 | No. BN-HTRd is close (3.2 / 10.0); BanglaWriting (8.7 / 22.6) and unseen writers (11.2 / 40.4) are far. |
| Handwritten English < 5 / ≈ 5 | 9.5 / 25.6 | No. IAM 6.3 / 17.9; in-the-wild GNHK 10.1 / 33.7 and IMGUR5K 12.0 / 25.3. |

For scale: handwritten line WER of 5–8% is beyond published results for line recognizers of any size (the best reported IAM line WER is about 4.8% with a ~100M-parameter model); Orus-OCR has 1.76M parameters.

## Per test set

| Set | What it is | Shipped CER | Shipped WER | Fast CER | Fast WER |
|---|---|---|---|---|---|
| bnhtrd | BN-HTRd, Bangla handwritten lines | 3.2 | 10.0 | 4.4 | 15.7 |
| banglawriting | BanglaWriting, Bangla handwritten lines | 8.7 | 22.6 | 9.3 | 27.0 |
| iiit | IIIT-INDIC-HW-WORDS, Bangla handwritten words | 5.1 | 17.7 | 6.7 | 28.7 |
| held-out BN | Bangla handwriting from writers and a source never trained on (221 lines) | 11.2 | 40.4 | 13.7 | 47.3 |
| iam | IAM, English handwritten lines | 6.3 | 17.9 | 8.1 | 24.5 |
| gnhk | GNHK, English handwriting, phone photos | 10.1 | 33.7 | 11.9 | 41.0 |
| imgur5k | IMGUR5K, English handwritten words in the wild | 12.0 | 25.3 | 13.4 | 29.7 |
| prescription | doctors' handwritten medicine names | 3.9 | 7.5 | 5.7 | 14.4 |
| mixed | real handwritten Bangla + English words stitched into lines | 10.3 | 24.7 | 11.8 | 30.9 |
| bnprinted | printed Bangla lines, photos / scans | 11.1 | 23.1 | 12.1 | 25.9 |
| swapnil | printed Bangla book lines | 0.9 | 6.1 | 0.9 | 5.9 |
| bench1000 | OCR-Bench1000-Bengali | 2.4 | 8.6 | 2.9 | 10.8 |
| sroie | SROIE receipts (scans) | 1.9 | 8.8 | 2.1 | 9.6 |
| cord_en | CORD receipts (photos) | 3.1 | 12.0 | 3.4 | 13.7 |
| funsd | FUNSD scanned forms | 6.2 | 23.2 | 8.0 | 27.3 |
| textocr_num | TextOCR scene-text numbers | 31.1 | 42.0 | 33.5 | 44.0 |
| crohme | CROHME 2019 handwritten arithmetic | 13.0 | 41.1 | 13.5 | 39.3 |
| print_bn / print_en | unseen print fonts, Bangla / English | 1.9 / 0.9 | 5.8 / 2.3 | 2.6 / 1.0 | 8.8 / 2.9 |
| font_bn / font_en | unseen fonts incl. handwriting-style | 2.4 / 2.6 | 5.8 / 6.1 | 3.2 / 3.0 | 9.5 / 8.4 |
| font | 0.1-alpha unseen-font set (mixed scripts) | 1.6 | 5.0 | 2.3 | 7.7 |
| digits_bn / digits_en | printed numbers, dates, prices | 5.1 / 3.5 | 7.9 / 6.5 | 5.2 / 3.6 | 8.4 / 6.5 |
| arith_bn / arith_en | printed arithmetic | 3.5 / 3.6 | 10.7 / 9.9 | 3.6 / 3.7 | 10.7 / 10.1 |
| hdigits / harith | handwritten digits / arithmetic (MNIST test digits) | 2.5 / 2.4 | 11.8 / 11.8 | 2.6 / 2.4 | 11.9 / 11.8 |
| receipt_bn / receipt_en | synthetic receipt rows | 5.3 / 3.5 | 9.0 / 6.4 | 5.5 / 3.6 | 10.0 / 6.4 |
| rx_print | printed prescription lines | 1.6 | 3.5 | 1.6 | 3.2 |
| book_bn / book_en | book / newspaper lines | 6.7 / 3.7 | 13.3 / 6.1 | 8.1 / 4.0 | 19.7 / 7.2 |
| noisy | heavily degraded printed lines | 11.9 | 22.0 | 12.8 | 25.2 |

### Where the English WER comes from

| Set | WER | WER, case-insensitive | + punctuation removed | reference words not in the lexicon |
|---|---|---|---|---|
| iam | 17.9 | 17.3 | 14.3 | 6.0% |
| gnhk | 33.7 | 31.0 | 25.5 | 14.2% |
| imgur5k | 25.3 | 22.0 | 20.3 | 15.3% |
| bnhtrd | 9.9 | 9.9 | 9.7 | 5.9% |
| held-out BN | 40.4 | 40.4 | 23.9 | 7.0% |

English loses 3–8 WER points to case and punctuation, which Bangla barely has. IAM labels put spaces around punctuation (`it ? Very`), so one missed space costs two words. On the held-out Bangla set, 16 of the 40 WER points are punctuation (dropped or detached danda `।`).

## Full pages

Indra OCR line detector + Orus recognizer, 30 test pages per set, shipped decoding:

| Pages | Page CER % | Page WER % |
|---|---|---|
| BN-HTRd (Bangla handwriting) | 7.1 | 16.3 |
| Printed Bangla book pages (swapnil) | 0.7 | 5.0 |
| IAM forms (English handwriting) | 12.7 | 34.2 |

On BN-HTRd the page CER is 7.0 with detected lines and **2.9 with ground-truth line boxes**: on dense, slanted Bangla handwriting the detector sometimes merges two lines, and this is now the largest page-level error. On printed pages the detector costs nothing (0.68 vs 0.77). IAM forms include the printed prompt and header, which the reference omits, so their page numbers overstate the error. GNHK and BanglaWriting pages (scattered text, unclear reading order) are not reported.

Language ID (bn / en / bn+en) is 90–100% correct per page, the handwritten flag 93–100%, unchanged from 0.1-alpha.

## Speed

CPU only, ONNX Runtime with 4 threads, AMD Ryzen 5 3500X, lightly loaded desktop.

| | Shipped | Fast |
|---|---|---|
| ms per handwritten line (bnhtrd, banglawriting, iam, gnhk) | 13–21 | 6–10 |
| ms per printed line (swapnil, sroie) | 2–11 | 2–5 |
| Example page (22 handwritten lines, `examples/tagore_lekhan.jpg`), median of 8 runs | 314 ms (detect 46, recognize 250, handwritten flag 18) | 142 ms |

The lexicon and the confidence-gated re-read roughly double the recognition time and give −1.4 to −1.6 CER and −6 WER points on handwriting.

## Size

| File | Size | Parameters |
|---|---|---|
| `orus_rec.onnx` (fp32) | 7.1 MB | 1.76M |
| `orus_lm.npz` | 3.2 MB | |
| `orus_lex.txt.gz` | 1.3 MB | 246k words |
| `orus_pic.onnx` | 0.16 MB | 38k |
| `orus_hand.onnx` | 0.40 MB | 98k |
| Indra `det.onnx` (downloaded separately) | 0.39 MB | 91k |

INT8 is not shipped: dynamic INT8 quantisation cost 7–10 CER points on handwriting and was slower on this CPU.

## Known weaknesses

- Scene text: TextOCR numbers 31% CER; general scene text (signboards, product labels) is out of scope for this version.
- Faint scanned forms (FUNSD 6.2% CER) and phone photos of printed Bangla with label noise (11.1%).
- Bangla handwriting from writers and sources it has never seen: 11.2% CER, 40% WER.
- Dense, slanted Bangla handwriting at page level, because of line detection (see above).
- Reading order is single-column (rows top to bottom); multi-column pages are read row by row across columns.
- Every line carries a confidence value: treat lines below 0.8 as unreliable.
