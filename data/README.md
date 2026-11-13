# data/

Everything here except this file is gitignored. The team's copy lives in the shared Google
Drive folder **`nasdaq100-news-graph/data`**, laid out exactly like this directory:

```
data/
├── raw/                      # downloads: fnspid_filtered.csv, scraped tables
├── membership.parquet
├── prices.parquet
├── labels.parquet
├── article_tickers.parquet
├── news_features.parquet
├── edges.parquet
└── predictions/<model>.parquet
```

Formats are fixed by [`SCHEMA.md`](../SCHEMA.md). Code finds the folder through the
`NDX_DATA_DIR` environment variable (default: this directory).

## Colab

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["NDX_DATA_DIR"] = "/content/drive/MyDrive/nasdaq100-news-graph/data"
```

If the folder was shared with you rather than created by you, first add a shortcut to it in
"My Drive" (right-click → Organize → Add shortcut) so it appears under `MyDrive`.

## Local

Either sync the Drive folder with Google Drive for desktop and point at it:

```bash
export NDX_DATA_DIR="/path/to/GoogleDrive/My Drive/nasdaq100-news-graph/data"   # PowerShell: $env:NDX_DATA_DIR = "..."
```

or download the files you need into this directory.

## Publishing a new artifact

1. Write it with `src.core.io.write_table`, which validates it against the contract.
2. Upload it to the Drive folder, replacing the old file.
3. Comment on the issue that produced it, then close the issue so blocked work can start.

To work before an upstream artifact exists, use the synthetic copies:
`NDX_DATA_DIR=tests/fixtures`.
