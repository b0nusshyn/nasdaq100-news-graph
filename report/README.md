# report/

One Markdown file per section in `sections/`, each owned by one person. Figures and tables
come from `results/`; reference them with relative paths (`../results/figures/<name>.png`).

| File | Owner | Issue |
|---|---|---|
| `01_introduction.md` | Everyone | #51 |
| `02_dataset.md` | Data | #46 |
| `03_news_encoding.md` | NLP | #48 |
| `04_method.md` | Modelling | #47 |
| `05_evaluation.md` | Evaluation | #50 |
| `06_results.md` | Evaluation | #50 |
| `07_interpretability.md` | NLP | #48 |
| `08_conclusion.md` | Everyone | #51 |

Build the merged draft (gitignored) in section order:

```bash
cat report/sections/*.md > report/report.md
```
