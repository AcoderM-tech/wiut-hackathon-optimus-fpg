# Fintech track — alert eskalatsiya ehtimoli · OPTIMUS FPG

Moliyaviy monitoring alertlarining eskalatsiya qilinish ehtimolini bashorat qilish.
Metrika: ROC-AUC. EDA sayti: `site/index.html`.

## Tuzilma

```
data/                             5 ta asl fayl (train/test signals + transactions)
src/train.py                      asosiy pipeline → submission + CV + loglar
src/exp_papers.py                 adabiyotdagi retseptlar, bir xil CV
src/train_gru.py                  ketma-ketlik modeli (bi-GRU + attention)
src/site_data.py                  outputs/ dagi hamma narsa → site/data.js
notebooks/01_eda.ipynb            EDA, grafiklar site/img/ ga
notebooks/02_experiments.ipynb    yondashuvlar taqqoslash → outputs/experiments.csv
notebooks/03_final_model.ipynb    yakuniy model + submission tekshiruvi
outputs/                          submission, loglar, CV hisobotlari, feature importance
site/                             EDA sayti (index.html + data.js + img/)
```

Saytda qo'lda yozilgan raqam yo'q: barcha qiymatlar `site/data.js` dan o'qiladi,
uni esa `src/site_data.py` `data/` va `outputs/` fayllaridan hisoblaydi.

## Ishga tushirish

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tartib muhim — har bir qadam oldingisining chiqishini ishlatadi:

| # | Buyruq | Vaqt | Chiqish |
|---|--------|------|---------|
| 1 | `python src/train.py --team <ID>` | ~5 daq | `team_<ID>.csv`, `run_summary.json`, `train_log.txt`, `feature_importance.csv`, `oof_predictions.csv` |
| 2 | `python src/exp_papers.py` | ~10 daq | `experiments_papers.csv`, `exp_papers.log`, `outputs/cache/` |
| 3 | `python src/train_gru.py` | GPU ~20 daq | `gru_training.log`, `gru_best.pt` |
| 4 | `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb` | ~3 daq | `site/img/*.png` |
| 5 | `jupyter nbconvert --to notebook --execute notebooks/02_experiments.ipynb --output 02_experiments.ipynb` | ~25 daq | `experiments.csv` |
| 6 | `jupyter nbconvert --to notebook --execute notebooks/03_final_model.ipynb --output 03_final_model.ipynb` | ~5 daq | submission tekshiruvi |
| 7 | `python src/site_data.py --repo <github-url>` | ~2 daq | `site/data.js` |

Notebooklarni `notebooks/` papkasidan ishga tushiring (yo'llar nisbiy).
Har qadam ixtiyoriy: fayl bo'lmasa, saytda o'sha qiymat `—` bo'lib qoladi, xato bermaydi.

CPU'da GRU sekin — `python src/train_gru.py --epochs 30 --cpu`.

## Natija

`outputs/cv_report.txt` — 5-fold stratified CV:
LightGBM (top-60) va logistik regressiyaning rank-blend'i, ROC-AUC ≈ 0.66.