# Datasets

Orus-OCR-1.0-alpha was trained on 25 line sources: 18 public datasets (365k real text lines and words) and 7 kinds of synthetic lines (473k). This page lists every source, where it comes from, its licence, and how it was used. **No data is redistributed with this project**, so download each dataset from its original source and accept its terms.

Licences were checked on the source pages in October 2026. Where no licence is stated we say so. This is not legal advice. Several sources allow **non-commercial / research use only**, and that is the reason the model weights are released for non-commercial use (see [the licence section in the README](../README.md#licence)).

## How the data is used

- **Line crops.** `orus/ocr_data.py` turns each dataset into page manifests (`data/ocr/pages/<src>.jsonl`), then cuts every text line into a 48 px "ink map" with two lines of context above and below (Indra OCR's `crop_line` rule), stored in memory-mapped stores (`data/ocr/rec48/`). Word-level datasets give one crop per word.
- **Splits.** Official train / val / test splits are kept where a dataset has them. Otherwise the split is a deterministic hash of a writer, page or book key, so no writer or page appears in two splits. Test splits are never used for training, for the lexicon or for the language model.
- **Labels.** Text is NFC-normalised. Lines with characters outside the 211-character vocabulary (Latin, the Bengali Unicode block, common typographic marks) are dropped, never altered.
- **Mix.** Every training batch draws lines from the sources by fixed weights (`SOURCE_W` in `orus/ocr_train.py`, the "share" column below), not by dataset size.

## Real data

| Source (store name) | Link | Kind | Train lines | Share of batches | Licence |
|---|---|---|---|---|---|
| BN-HTRd (`bnhtrd`) | [Mendeley 743k6dm543](https://data.mendeley.com/datasets/743k6dm543) | Bangla handwritten pages → lines | 11,005 | 10.0% | CC BY 4.0 |
| IIIT-INDIC-HW-WORDS, Bengali (`iiit`) | [CVIT, IIIT Hyderabad](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words) | Bangla handwritten words | 82,554 | 7.0% | CC BY 4.0 (as listed on AIKosh; check the dataset README) |
| BanglaWriting (`banglawriting`) | [Mendeley r43wkvdk4w](https://data.mendeley.com/datasets/r43wkvdk4w) | Bangla handwritten pages → lines | 1,979 | 4.0% | **CC BY-NC 3.0** |
| Doctor's Handwritten Prescription BD (`prescription`) | [Mendeley zjjtptvn6f](https://data.mendeley.com/datasets/zjjtptvn6f) | handwritten medicine names (Bangladesh) | 3,120 | 3.0% | CC BY 4.0 |
| GNHK (`gnhk`) | [GoodNotes/GNHK-dataset](https://github.com/GoodNotes/GNHK-dataset) | English handwriting, phone photos | 5,991 | 5.0% | CC BY 4.0 |
| IAM, line level (`iam`) | [IAM database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), via [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) | English handwriting, clean | 6,482 | 5.0% | **IAM terms: non-commercial research** |
| IMGUR5K (`imgur5k`) | [facebookresearch/IMGUR5K-Handwriting-Dataset](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset) | English handwritten words in the wild | 164,118 | 6.0% | **CC BY-NC 4.0** |
| RIMES-2011, line level (`rimes`) | [Teklia/RIMES-2011-line](https://huggingface.co/datasets/Teklia/RIMES-2011-line) | French handwritten letters (lines with accents drop out of the vocabulary) | 3,485 | 2.5% | MIT (dataset card) |
| CROHME (`crohme`) | [Neeze/CROHME-full](https://huggingface.co/datasets/Neeze/CROHME-full) | handwritten maths; plain arithmetic expressions only | 450 | 0.5% | none stated (CROHME competition data, research) |
| Bengali OCR Synthetic, words (`bnsynthwords`) | Hugging Face "Bengali OCR Synthetic Dataset" (30k printed word images) | printed Bangla words | 25,464 | 1.5% | CC BY 4.0 |
| Bangla OCR validation, printed (`bnprinted`) | [arobin79/bangla-ocr-validation_data_printed](https://huggingface.co/datasets/arobin79/bangla-ocr-validation_data_printed) | printed Bangla lines (photos / scans); images holding more than one text row removed | 997 | 1.5% | **none stated** |
| Bangla OCR dataset, book pages (`swapnil`) | [swapnil7777/bangla-ocr-dataset](https://huggingface.co/datasets/swapnil7777/bangla-ocr-dataset) | printed Bangla pages; lines paired with the page text when the detected line count matches | 6,999 | 3.0% | **none stated** |
| bangla-ocr-1000 (`sahil`) | [sahil74/bangla-ocr-1000](https://huggingface.co/datasets/sahil74/bangla-ocr-1000) | printed Bangla book pages, paired as above; pages overlapping the `swapnil` test pages removed | 6,162 | 2.0% | **none stated** |
| SROIE 2019 (`sroie`) | [ICDAR SROIE](https://rrc.cvc.uab.es/?ch=13), via [Voxel51/scanned_receipts](https://huggingface.co/datasets/Voxel51/scanned_receipts) | printed English receipts, line boxes | 30,261 | 2.5% | ICDAR competition terms (the mirror lists CC BY 4.0) |
| CORD v2 (`cord`) | [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | printed receipts (photos); rows deskewed from word quads | 3,354 | 1.5% | CC BY 4.0 |
| FUNSD (`funsd`) | [FUNSD](https://guillaumejaume.github.io/FUNSD/), via [nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) | scanned forms; words grouped into lines | 6,830 | 1.5% | **non-commercial research** |
| TextOCR, number crops (`textocr`) | [TextOCR](https://textvqa.org/textocr/), via [MiXaiLL76/TextOCR_OCR](https://huggingface.co/datasets/MiXaiLL76/TextOCR_OCR) | scene-text numbers | 5,467 | 1.0% | CC BY 4.0 (TextOCR annotations) |
| Teacher-labelled lines (`distill`) | unlabeled pages: [arnaudstiegler/reddit_handwriting](https://huggingface.co/datasets/arnaudstiegler/reddit_handwriting) and [NoTeS-Bank](https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge) (both Apache 2.0), and a set of handwritten medical forms (images only) | English handwritten lines read by a vision-language teacher (PaddleOCR-VL) and kept only where the teacher and an earlier Orus model agree | 500 | 2.0% | see the page sources; the script that made this store is not included |

Share percentages are the configured weights; they are renormalised over the sources present (the total configured weight is 1.0 over the real and synthetic sources together).

## Synthetic data (`orus/ocr_synth.py`)

| Store | Train lines | Share | What it is |
|---|---|---|---|
| `synth_font` | 207,216 | 15% | Bangla, English and code-mixed text (sentences, lists, prescriptions) rendered in printed and handwriting-style fonts, with paper, blur, JPEG and warp effects |
| `synth_stitch` | 150,000 | 12% | real handwritten **word** crops from the training splits of BN-HTRd, BanglaWriting, IIIT, GNHK, IMGUR5K and the prescriptions set, stitched into Bangla / English / mixed lines |
| `synth_num` | 28,694 | 4% | numbers, dates, prices, phone numbers, units, fractions and arithmetic, in Bangla and Latin digits |
| `synth_book` | 29,803 | 4% | book and newspaper lines: justified, small sizes, scan noise, bleed-through |
| `synth_receipt` | 29,100 | 3% | receipt rows: items, quantities, prices, totals, dot leaders, thermal / dot-matrix fonts |
| `synth_hdigit` | 14,732 | 2% | handwritten digit strings and arithmetic built from [MNIST](https://huggingface.co/datasets/ylecun/mnist) digits (MIT) |
| `synth_rx` | 13,745 | 1.5% | printed prescription lines |

About 8–12% of each kind is heavily degraded (low contrast, stains, blur, compression, noise).

Inputs to the generator:

- **Text:** Bangla Wikipedia ([wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), 20231101.bn) and English WikiText-103 ([Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)), both CC BY-SA; the Bangla newspaper corpus [zabir-nabil/bangla_newspaper_dataset](https://huggingface.co/datasets/zabir-nabil/bangla_newspaper_dataset) (MIT); generated numbers, receipts and prescriptions.
- **Fonts:** [Google Fonts](https://github.com/google/fonts) (OFL / Apache 2.0; one Regular, Italic and Bold file per family), Unicode Bangla fonts from [fahimscirex/bangla-fonts](https://github.com/fahimscirex/bangla-fonts), and the FreeFont / Noto Bengali system fonts. 15% of font families (by a hash of the family name) are **held out** and only used for the unseen-font test sets.

## Language model and lexicon

- `orus_lm.npz`: character 5-gram LM (absolute discounting, Indra OCR's `lm.npz` format) over 6M characters of Bangla Wikipedia, WikiText-103 and generated code-mixed / list / prescription lines (`orus/ocr_lm.py`).
- `orus_lex.txt.gz`: 246k-word lexicon with counts (Bangla + lower-case English) from Bangla Wikipedia, the Bangla newspaper corpus, WikiText-103, the system English word lists (`/usr/share/dict`) and the **training-split** labels of the real line datasets above. Words seen fewer than twice are left out (`orus/ocr_words.py`).

## Test-only data

These are used for evaluation only (see [BENCHMARKS.md](BENCHMARKS.md)):

| Set | Source | Licence |
|---|---|---|
| held-out Bangla handwriting (`rupom`) | [rupomkhondaker/Bangla_OCR_Hand_writing](https://huggingface.co/datasets/rupomkhondaker/Bangla_OCR_Hand_writing), 221 line images from writers not in any training set; never trained on | none stated |
| OCR-Bench1000-Bengali (`bench1000`) | [meharuhanzz/OCR-Bench1000-Bengali](https://huggingface.co/datasets/meharuhanzz/OCR-Bench1000-Bengali) | MIT |
| CROHME 2019 test, TextOCR test numbers, CORD test, FUNSD test, SROIE test receipts | as above, official or held-out test splits | as above |

Datasets deliberately **not** used: a Hugging Face copy of the held-out Bangla handwriting set (to keep it unseen), a third-party SROIE line-crop set (it may contain SROIE test receipts), and a re-packaged BN-HTRd set (test leakage risk).

## Other model components

- **Line detector:** Indra OCR 1.1 `det.onnx` from [datalab-to/indra-ocr](https://huggingface.co/datalab-to/indra-ocr), used unchanged and downloaded from there at run time (not redistributed).
- **Recognizer warm start:** the conv layers start from Indra OCR 1.1 `rec.onnx`, so the Orus recognizer is a derivative of it (see the licence notes).
- **Picture-region net and handwritten flag** (`orus_pic.onnx`, `orus_hand.onnx`): unchanged from Orus-OCR-0.1-alpha; trained on BN-HTRd, GNHK and BanglaWriting pages with pasted [Imagenette](https://github.com/fastai/imagenette) photos and charts (picture net), and on printed / handwritten page tiles (handwritten flag). Their training code is in the [0.1-alpha repository](https://github.com/MahirHamiAbrar/Orus-OCR-V0).
