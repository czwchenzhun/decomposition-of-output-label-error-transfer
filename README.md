# Data acquisition and reproduction guide

> This file explains **which scripts in package can be run directly, which require external raw data, and how to obtain that data**.
> Read it together with `requirements.txt` and `01_Code/_Script_Number_Mapping.csv`.

---

## 1. What this package provides and what it does not

**Provided**: **all data extracted from the raw sources and used in the paper's analysis**---per-configuration CSVs, six-region pooled tables, training-result JSON, station base tables, **training-input npz files and per-configuration prediction samples** (`02_Results_Data/Training_Inputs/`, about 666 MB), the 16 in-use manuscript figures and model weights---together with **all scripts** that produced them, plus the manuscript source and PDFs.

**Not provided**: the raw input data themselves. They amount to about 170 GB and are all public datasets; see Section 5 for access. All **analysis scripts** in this package can run directly without them; only the **upstream-chain scripts** listed in Section 3 require them.

This package is a **self-contained directory**: apart from upstream-chain scripts that by design read external raw data, every other script reads only `02_Results_Data/`, `00_Manuscript/figures/` and `03_Figures/` inside the package, and references no data file or absolute path outside it.

---

## 2. Runtime environment

`requirements.txt` was generated from the imports actually used by the scripts and has three groups:

| Group | Dependencies | Purpose |
|---|---|---|
| Core (**needed to reproduce all numbers and figures in the paper**) | `numpy` `pandas` `scipy` `matplotlib` `PyMuPDF` | Analysis scripts (19--69) and figure regeneration |
| Upstream extraction chain | `netCDF4` `xarray` `rasterio` | Preparing regional fields from raw data (01--18) |
| Training chain | `torch` `scikit-learn` `lightgbm` | Model training in 12/13 (requires GPU + raw labels); 49/68 conventional baselines can be run directly |

Install with `pip install -r requirements.txt`.

- Python 3.10+ suffices for the analysis scripts (the original environment was conda `geoenv`: Python 3.14 / numpy 2.4 / matplotlib 3.10.8).
- **Pin the matplotlib version when regenerating figures**: a different version changes canvas size by 1--2 px and alters the MD5 of every figure. `requirements.txt` does not lock the minor version; install `matplotlib==3.10.8` if byte-identical archived figures are required.
- The training chain needs a ROCm/CUDA environment; under Windows + ROCm the backward pass of `nn.LSTM` / `nn.GRU` is unavailable, so the training scripts use hand-written `LSTMCell` / `GRUCell` loops.
- LaTeX compilation is not a pip dependency: the main text uses `xelatex -> bibtex -> xelatex x2` (external bibliography in `00_Manuscript/bibtex/`), and the Supplementary Materials still uses two `xelatex` passes. Compiled PDFs and `main.bbl` are supplied under `00_Manuscript/`, so running only `xelatex` also produces the reference list. `bibtex` and `xelatex` belong to the TeX distribution and need no pip installation.

---

## 3. Which scripts can be run directly

### 3.1 Can be run directly (**no external data required**)

The **analysis scripts** (numbers 19--48, 50--56, 59--67, 69) in
`01_Code/05_Error_Transfer_Quadratic_and_Calibration/`, `06_Paper_Numbers_and_Figures/` and `07_Cross-region_Replication_and_Robustness/`,
plus `00_Manuscript/Build_and_Check_Scripts/_regen_figs.py`.

How to run (either form works; the file header automatically adds `01_Code/00_Common_Modules` to `sys.path`):

```bash
# From the package root
python 01_Code/06_Paper_Numbers_and_Figures/33_final_numbers.py
python 01_Code/07_Cross-region_Replication_and_Robustness/52_regime_contrast.py

# Or from the script's own directory
cd 01_Code/05_Error_Transfer_Quadratic_and_Calibration && python 27_quadratic.py
```

**Runtime-copy mechanism**: on the first `import paths`, the scripts copy `02_Results_Data/truth_chain/` (about 36 MB) to `06_Reproduction_Outputs/truth_chain/`, after which all reads and writes point to that runtime copy. The archived CSVs in `02_Results_Data/` therefore **stay read-only**.

**Output locations**:

| Output | Directory |
|---|---|
| Runtime logs / txt reports (`common.report()`) | `04_Runtime_Logs/` |
| Recomputed process figures and intermediate CSVs | `06_Reproduction_Outputs/` |
| Regenerated paper figures | `00_Manuscript/figures/` (renamed and archived uniformly by `_regen_figs.py`) |

`04_Runtime_Logs/` and `06_Reproduction_Outputs/` initially contain only `README.txt`; running the scripts creates content automatically and **does not affect the archived originals in the package**.

### 3.2 Training inputs and prediction samples (included; can be run directly)

The following 4 scripts read the training-input npz files and per-configuration prediction samples directly. Those files are now all included in `02_Results_Data/Training_Inputs/` (about 666 MB); on first import, `paths.py` copies them to the runtime copy at `06_Reproduction_Outputs/_Upstream_Intermediate_Products/_bench/`, so **they can be run as soon as the package is opened**. The "requires raw data" column of `01_Code/_Script_Number_Mapping.csv` has been changed to `No (training inputs included in package)`.

| New number | Script | Included dependency |
|---|---|---|
| 49 | `49_ml_baseline.py` | `dl_R0_..._sta.npz` (default `--npz` / `--region` already set) |
| 57 | `57_pers_capacity.py` | six-region `dl_*_sta.npz` + `pred_HYBNS_staG*.npz` + `cells.npz` |
| 58 | `58_damaged_vs_pers.py` | `dl_R0_..._sta.npz` + `pred_LSTM_*.npz` + `cells.npz` |
| 68 | `68_mlbase_multiregion.py` | five-region `dl_*_sta.npz` (`--no_train` recomputes only the summaries) |

The grid index `cells.npz` is at `02_Results_Data/truth_chain/multi/<region>/samples/` and is copied to the runtime copy along with `truth_chain`. The result CSVs (`truth_chain_L2_monthly_mlbase.csv`, `pers_truthchain.csv`, `multi/mlbase_multiregion.csv`, etc.) are also supplied in full, so the paper numbers and figures do not depend on rerunning training.

### 3.3 Upstream extraction / training chain (requires raw data)

| Number | Directory | Description |
|---|---|---|
| 01--10 | `02_Multi-region_Data_Preparation/` | Prepare six-region grid fields and sample matrices from raw data |
| 11--14 | `03_DL_Training_and_Transfer/` | Extract compact npz, align PSP evapotranspiration, train models |
| 15--18 | `04_Truth_Verification_Chain/` | Station-truth comparison chain; 16/17 can now run from the package, 15/18 still need raw pairings |
| 27 `.sh` files | `08_Batch_Drivers/` | Training-period batch drivers, **kept as a process record**, requiring GPU + raw data + manual adjustment of script paths |

By design these scripts **cannot be run immediately after opening the archive**: the raw data must first be downloaded and placed, `RAW_ROOT` set, and a GPU environment prepared. They remain in the package so that reviewers can check "how the result data were extracted from the raw data", not so that they run out of the box.

---

## 4. Running order and environment variables of the upstream chain

The upstream-chain scripts read the following paths from `paths.py`; all are **relative to this package** and no machine-specific path is hard-coded:

| Variable | Default | Meaning |
|---|---|---|
| `RAW_ROOT` | `<package root>/_Raw_Data_To_Be_Downloaded` | Root directory of the raw datasets (**must be set by the user**) |
| `WORK` | `06_Reproduction_Outputs/_Upstream_Intermediate_Products` | Intermediate products (npz, sample matrices, bench) |
| `BENCH` | `WORK/_bench` | Runtime copy of training inputs / prediction samples (archived originals in `02_Results_Data/Training_Inputs/`) |
| `RAW` | as above | Read-only root of the raw datasets |
| `DATA` | `02_Results_Data` | Archived result data supplied with the package (read-only) |

Setting it (Git Bash / Linux / macOS):

```bash
export RAW_ROOT=/path/to/raw_datasets
```

Windows PowerShell:

```powershell
$env:RAW_ROOT = "D:\path\to\raw_datasets"
```

Upstream-chain order (each step requires its corresponding raw data; see the docstring at the top of each script):

```text
01_probe_multiregion.py              candidate windows and station counts
02--06_build_*_multi.py              six-region target / soil / terrain / meteorological / precipitation fields (once per region)
07_assemble_multi.py / 08_check_gate.py / 09_build_samples_multi.py
10_station_cells.py                  station-to-grid-cell matching
11_dl_prep.py / 14_prep_et.py       extract compact npz, align PSP evapotranspiration
12_dl_smoke.py / 13_dl_hybrid.py    model training (requires GPU; batch drivers in 08_Batch_Drivers/)
15--16_station_truth_chain / model_vs_truth (L1 / L2, six regions)
18_truth_chain_summary (three layers combined into one table)
19--69                               analysis chain (raw data no longer needed; training inputs included)
```

`08_Batch_Drivers/*.sh` are one-off batch scripts from the training period. Their variables are now relative to the package (`PKG_CODE` / `WORK_DIR` / `BENCH_DIR` / `DATA_DIR`), but the script invocations still follow the directory structure of that time, so reuse requires manual adjustment to the new numbers and subdirectories under `01_Code/`.

Note: netCDF4 / HDF5 cannot open non-ASCII paths, so keep the raw-data directory on an ASCII path.

---

## 5. Raw-data list and access

The table gives the **official download-page URLs of each data publisher** (full addresses, ready to paste into a browser). Most entries at the National Tibetan Plateau Data Center (TPDC) and Copernicus Data Space require free registration and a short statement of intended use.

| Data | Purpose | Official download page |
|---|---|---|
| China 732-station multilayer soil-moisture dataset (gravimetric monthly values) | Ground truth | <https://cstr.cn/31253.11.sciencedb.00539> |
| CMFD 2.0 meteorological forcing (0.1 deg / 3 h, 1951--2024) | Meteorological forcing | <https://data.tpdc.ac.cn/zh-hans/data/e60dfd96-5fd8-493f-beae-e8e5d24dece4> |
| CSDLv2 soil properties (1 km version used locally) | Static land-surface properties | <https://data.tpdc.ac.cn/zh-hans/data/46ddd893-3b2b-4bb3-b9e6-b043f3c5c3a2> |
| Copernicus GLO-30 DEM | Terrain slope | <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM> |
| PSP evapotranspiration product (ET/PET/Tr/Eb/Ei, 0.1 deg daily) | Auxiliary evapotranspiration loss | <https://data.tpdc.ac.cn/zh-hans/data/9e59e3b7-afca-41cf-98e0-3693a067e3c6> |
| SMCI1.0 merged soil-moisture product (training label) | Training label | <https://data.tpdc.ac.cn/zh-hans/data/49b22de9-5d85-44f2-a7d5-a1ccd17086d2/> |

**DOIs and alternative entry points** (DOI links resolve to the same data page; parentheses give the citation convention used in the paper's reference list):

- **732-station observations**: dataset DOI <https://doi.org/10.11922/sciencedb.00539>; authors' data-sharing page <https://nzc.iap.ac.cn/content?cid=24&aid=928>; companion paper Wang & Shi (2019), *J. Hydrometeorol.* <https://doi.org/10.1175/JHM-D-19-0035.1>.
- **CMFD 2.0**: DOI <https://doi.org/10.11888/Atmos.tpdc.302088>; English page <https://data.tpdc.ac.cn/en/data/e60dfd96-5fd8-493f-beae-e8e5d24dece4>.
- **CSDLv2**: DOI <https://doi.org/10.11888/Terre.tpdc.301235>; companion paper Shi et al. (2025), *ESSD* <https://essd.copernicus.org/articles/17/517/2025>. The first-version data page <https://data.tpdc.ac.cn/zh-hans/data/11573187-fd64-47b1-81a6-0c7c224112a0/> corresponds to Shangguan et al. (2013), *JAMES* (the methodology paper cited in the reference list).
- **Copernicus GLO-30**: registration-free AWS open mirror <https://registry.opendata.aws/copernicus-dem/>; citation DOI <https://doi.org/10.5270/ESA-c5d3d65>.
- **PSP evapotranspiration**: DOI <https://doi.org/10.11888/Terre.tpdc.303108>; DOI of its meteorological forcing HCD01 <https://doi.org/10.11888/Atmos.tpdc.302712>.
- **SMCI1.0**: DOI <https://doi.org/10.11888/Terre.tpdc.272415>; methodology paper <https://doi.org/10.5194/essd-14-5267-2022>. The product itself is publicly downloadable, but **the training-station list is not public and the overlap with the truth stations cannot be verified** (the limitation stated in the paper is unchanged); the cropped version used here was supplied with the project's raw dataset.

Raw-data volume: about 170 GB; the **extracted data** supplied with the package are about 826 MB (training inputs and prediction samples about 666 MB, model weights about 94 MB, result CSVs and figures about 66 MB). Still excluded are the large intermediate files from the training period (full-resolution npz, evapotranspiration tensors, per-region `X.dat` sample matrices).

---

## 6. Script numbering

The scripts in the package have been **numbered** `01`--`69` in pipeline order. `01_Code/_Script_Number_Mapping.csv` gives a line-by-line mapping of "number / file name / directory / whether raw data are required". The six regions are numbered R0--R5, see `02_Results_Data/_Region_ID_Mapping.csv`.
