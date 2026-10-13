# model_CDD — 15-day fire-susceptibility map

An interactive map of **consecutive dry days (CDD)** for South-East Asia and northern Australia
(90°E–130°E, 30°S–40°N), rebuilt twice a day from the latest
[ECMWF open-data](https://www.ecmwf.int/en/forecasts/datasets/open-data) 15-day forecast and published with
GitHub Pages.

Long runs without rain dry out vegetation, so the longest forecast dry spell is a simple early indicator of
where fire conditions may develop, and when.

## What the map shows

- **Colour layer:** the longest run of consecutive dry days (< 1 mm/day) in each land cell during the forecast.
- **Outline layers** (layer menu, top right): areas that reach ≥ 5, ≥ 10 and ≥ 15 dry days. Hover or tap an
  area for the date its first cell reached that level, its peak dry spell and its area.
- **Click anywhere** for the cell's value and the dates it reached 5, 10 and 15 dry days.
- Satellite or light basemap, optional place labels, and a collapsible legend with the model run time.
- `cdd_se_asia.nc` (published next to `index.html`) holds the same data as NetCDF.

## How it works

1. Download total precipitation (`tp`, 6-hourly, 0–360 h) from the ECMWF open-data mirrors (Azure → AWS → Google
   → ECMWF, with retries).
2. `tp` is accumulated, so a UTC calendar day's rain is the difference between two consecutive midnights.
   A 00Z run gives 15 full days, a 12Z run gives 14.
3. A day with < 1 mm is dry. The engine tracks each cell's running streak, its maximum, and the day each
   streak first reaches 5, 10 and 15.
4. Ocean cells are masked. Each cell is sampled at 4×4 points and kept if at least 15 % of it is land, so
   small islands survive.
5. Threshold areas are traced exactly from the grid as polygons with holes, rather than from contour lines.

### Limitations

- It uses **rainfall only**. Operational fire danger indices (e.g. Fire Weather Index) also use temperature,
  humidity, wind and fuel moisture.
- Dry spells are counted **from the first forecast day**. Days before the forecast are unknown, so a spell
  already under way is under-counted.
- It is a single deterministic forecast, not an ensemble. Skill falls off steeply beyond about a week.
- The 5/10/15-day levels are display choices, not official fire-danger thresholds.

## Run it locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest                      # unit tests, no network needed
python main.py              # writes public/index.html and public/cdd_se_asia.nc
```

Useful options: `--grib FILE` (skip the download), `--sources aws,google`, `--no-land-mask`,
`--max-age-hours N` (the run fails if the forecast is older than N hours; default 36).

## Deployment

`.github/workflows/pipeline.yml` runs the tests, builds the map and deploys it to GitHub Pages at 08:45 and
20:45 UTC, shortly after the 00Z and 12Z runs become available, and on every push to `main` that touches the
code. Pull requests run the tests only. Enable it under *Settings → Pages → Source: GitHub Actions*.

GitHub switches off scheduled workflows after 60 days without repository activity. The deploy job re-enables the
workflow on each successful run to reset that timer (best-effort).

## Attribution

Forecast data: © ECMWF, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Satellite imagery and labels:
© Esri. Light basemap: © OpenStreetMap contributors © CARTO. Check each provider's terms before reusing the map.
