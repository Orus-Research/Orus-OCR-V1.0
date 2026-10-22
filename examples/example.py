"""Run Orus-OCR on the example page: Rabindranath Tagore's handwritten poem "Lekhan" (Bangla + English),
public domain, from Wikimedia Commons (see examples/README.md).

  python examples/example.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "orus"))

from ocr import OCR  # noqa: E402

ocr = OCR()                                   # downloads the weights on first use (CPU only)
r = ocr(os.path.join(HERE, "tagore_lekhan.jpg"))

print("Language:   ", r["language"])
print("Handwritten:", r["handwritten"])
print(r["text"])
print()
for line in r["lines"]:                       # every line: bbox, text, conf, lang
    flag = "  <- low confidence" if line["conf"] < 0.8 else ""
    print(f'{line["conf"]:.2f}  {line["lang"]:5s}  {line["text"]}{flag}')
print(r["ms"])
