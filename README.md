# FTSE SmallCap Uncertainty Site

This project turns the MATLAB-style uncertainty pipeline into a Python-driven static website.

## What it does

- Reads FTSE SmallCap monthly data from Yahoo Finance, a CSV file, or a CSV URL
- Fits an AR model to returns and derives forecast errors
- Builds rolling and EWMA uncertainty measures
- Exports `site/data/index.json` for the website
- Exports `site/data/plot.png` for sharing or embedding
- Supports monthly rebuilds with GitHub Actions

## Files

- `src/build_smallcap_index.py`: pipeline that reads the source data and writes the web assets
- `config.json`: source and model settings
- `site/index.html`: static dashboard
- `.github/workflows/monthly-update.yml`: monthly rebuild job

## Quick start

1. Install Python 3.12 or later.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Choose a data source in `config.json`.
   - Default: Yahoo Finance ticker `^FTSC`
   - Alternative: replace the sample CSV or point to a live CSV URL
4. Run:

   ```bash
   python src/build_smallcap_index.py
   ```

5. Open `site/index.html` with a static web server, or publish the `site` folder to GitHub Pages, Netlify, or Cloudflare Pages.

## Default online source

The current default configuration uses Yahoo Finance ticker `^FTSC` with a monthly interval. The script downloads the historical index level series, converts it into monthly percentage returns, and then runs the uncertainty model.

Important: `yfinance` documents that Yahoo Finance data is intended for personal use, and Yahoo's own help says some instruments may have download limits or licensing restrictions. If you want a more durable production source, consider FTSE Russell / LSEG data.

## How monthly automation works

The GitHub Actions workflow runs on the first day of each month at 06:00 UTC. It rebuilds the JSON and chart outputs, then commits the updated files back to the repository.

For a live CSV feed instead of Yahoo, update `config.json` like this:

```json
{
  "source": {
    "type": "csv_url",
    "path": "https://your-provider.example.com/ftse-smallcap-monthly.csv",
    "date_column": "Date",
    "returns_column": "Returns",
    "price_column": null,
    "date_format": "%d/%m/%Y"
  }
}
```

If your provider gives index levels instead of returns, set `returns_column` to `null` and populate `price_column`. The script will calculate percentage returns automatically.

