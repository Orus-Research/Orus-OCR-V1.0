"""Orus-OCR-1.0-alpha command line.

  python run.py page.jpg [more.jpg ...]     # language, handwritten Yes/No, structured text
  python run.py page.jpg --json             # + every line with box, text, confidence, language; timings
  python run.py page.jpg --fast             # greedy + char LM only (no lexicon, no re-read): ~2x faster
  python run.py page.jpg --weights DIR      # use local weights instead of downloading them

Weights are downloaded once from Hugging Face (orus-research/Orus-OCR-V1.0) and the line detector from
datalab-to/indra-ocr; both are cached by huggingface_hub. Set ORUS_WEIGHTS=DIR to work offline.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "orus"))

import ocr  # noqa: E402

if __name__ == "__main__":
    ocr.main()
