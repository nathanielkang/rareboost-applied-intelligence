# 13_AppliedIntel_RareBoost

**Paper:** Making Gradient Boosting Care About Rare Targets (RareBoost)
**Target journal:** Applied Intelligence (Springer), ISSN 0924-669X
**Inventory ID:** M2
**Authors / contact:** See `1_Manuscript/main.tex` title page (not duplicated here in the submission bundle).
**APC:** Institutional open access arrangement with Springer Nature (see Declarations in the manuscript PDF).

## Author affiliation (mandatory)

**Do not** write “IT College” (or “IT대학”) anywhere in this submission pack.

Use only:

- **Affiliation line:** School of Computer Science and Engineering, Kyungpook National University, Daegu, South Korea.

Rationale: the author’s institutional identity for this line of papers is the **department name only**; the college layer must not appear in `main.tex`, the cover letter, Editorial Manager, or proofs.

（中文備註：**不要**在投稿與校對稿中出現 IT College／IT대학；**只**寫「School of Computer Science and Engineering」+ 慶北大學 + 城市／國家。）

## Origin

Journal submission pack for *Applied Intelligence*. The manuscript and code were consolidated from an earlier **unpublished** working version used only inside the lab; no prior venue appears in the submitted materials.

## Structure

| Folder | Contents |
|---|---|
| `1_Manuscript/` | `main.tex`, `references.bib`, `figure_prompts.txt`, `sn-jnl.cls`, `sn-basic.bst`, `figures/` |
| `2_Code/` | Reference implementation: source, configs, and scripts. **No bundled experiment outputs** (run `scripts/run_experiments.py` to regenerate `results/` locally). |
| `cover_letter.tex` | One-page cover letter to the Editor-in-Chief |

## After every `main.tex` change (mandatory)

Whenever **`1_Manuscript/main.tex`** is edited (any agent or human), **immediately** run a full build from `1_Manuscript/` and leave an up-to-date **`main.pdf`**:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Do **not** end the task with “user can compile later.” If `main.pdf` is locked (viewer open), use a temporary `-jobname=…` build, then retry `main.pdf` after closing the viewer.

（中文：**每次**改完 `main.tex` 都要 **自動跑完** 上述編譯並更新 `main.pdf`；不可只改 TeX 不交 PDF。）

## Template (December 2024, official)

Use the **Springer Nature** journal article package from:  
`1_Manuscript/Download+the+journal+article+template+package+(December+2024+version).zip` (extracts to `1_Manuscript/sn-article-template/` as reference; the build uses the class and `.bst` files in `1_Manuscript/`.)

- **Class:** `sn-jnl.cls` (replaced with the one from the zip, not a third-party mirror)
- **Document options:** `pdflatex, referee, Numbered, sn-basic` (numbered natbib, double-spaced review mode)
- **Bib style:** `sn-basic.bst` in the manuscript root (and copy under `bst/`)
- **sttools / cuted:** `sn-jnl` loads `cuted.sty`; a generated `cuted.sty` + `cuted.dtx` (LPPL) live in `1_Manuscript/`. If your TeX system already has `sttools` installed, you can remove the local copy.

**Compile (from `1_Manuscript/`):**  
`pdflatex main` → `bibtex main` → `pdflatex main` → `pdflatex main`  
A successful run produces `main.pdf` (about 30+ single-column pages in referee mode).
