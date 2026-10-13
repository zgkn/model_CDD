"""Fire-susceptibility map from the ECMWF 15-day forecast.

Pipeline:
  1. download total precipitation (HRES, 6-hourly, 0-360 h) from the ECMWF open data service
  2. turn the accumulated rainfall into UTC-calendar-day totals
  3. count consecutive dry days (CDD, < 1 mm/day) per grid cell
  4. publish a NetCDF file and an interactive Leaflet map (folium)

Run ``python main.py --help`` for options.
"""

import argparse
import base64
import io
import json
import shutil
import tempfile
import time
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.image import imsave
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy import ndimage

# Bounding box
LAT_TOP, LAT_BOTTOM = 40, -30
LON_LEFT, LON_RIGHT = 90, 130

# A day with less rain than this is a "dry day" (standard CDD definition).
DRY_DAY_MM = 1.0
# Thresholds (days) that get their own toggleable outline layer.
LEVELS = {
    5: {"color": "#0000FF", "opacity": 0.15},
    10: {"color": "#FF00FF", "opacity": 0.20},
    15: {"color": "#000000", "opacity": 0.25},
}
RASTER_VMAX = 15  # colour scale saturates here ("15+")
RASTER_ANCHORS = [(0.0, "#FFFFCC"), (0.333, "#FECC5C"), (0.666, "#F03B20"), (1.0, "#BD0026")]

# ECMWF open-data mirrors, tried in order.
DEFAULT_SOURCES = ("azure", "aws", "google", "ecmwf")
FORECAST_STEPS = list(range(0, 361, 6))

# Land-mask sampling: each grid cell is sampled on SUBSAMPLE x SUBSAMPLE points and kept
# when at least LAND_FRACTION of them are land (so small islands are not lost).
SUBSAMPLE = 4
LAND_FRACTION = 0.15

NODATA = 255  # sentinel in the grids embedded in the map

ATTRIBUTION = (
    "Forecast: <a href='https://www.ecmwf.int/en/forecasts/datasets/open-data'>ECMWF</a> "
    "open data (CC BY 4.0)"
)


# --------------------------------------------------------------------------- #
# 1. Data retrieval
# --------------------------------------------------------------------------- #
def fetch_forecast(target, sources=DEFAULT_SOURCES, passes=2):
    """Download the latest 15-day precipitation forecast, trying each mirror in turn."""
    from ecmwf.opendata import Client

    last_err = None
    for attempt in range(passes):
        for source in sources:
            try:
                print(f"   downloading from '{source}' (pass {attempt + 1}/{passes})...")
                Client(source=source).retrieve(
                    type="fc", levtype="sfc", param=["tp"], step=FORECAST_STEPS, target=str(target)
                )
                return
            except Exception as err:  # network errors, 404 while a run is still being published...
                last_err = err
                print(f"   source '{source}' failed: {err}")
        if attempt < passes - 1:
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"All ECMWF sources failed; last error: {last_err}")


def load_precip(grib_path):
    """Open the GRIB file and return cumulative tp (step, lat, lon), latitude descending, cropped."""
    # indexpath="" stops cfgrib from leaving *.idx files next to the data
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    tp = ds["tp"].sortby("latitude", ascending=False)
    tp = tp.sel(latitude=slice(LAT_TOP, LAT_BOTTOM), longitude=slice(LON_LEFT, LON_RIGHT))
    tp = tp.load()
    ds.close()
    return tp


# --------------------------------------------------------------------------- #
# 2. Daily rainfall (UTC calendar days)
# --------------------------------------------------------------------------- #
def daily_rainfall_mm(tp, min_days=5):
    """Daily rainfall totals in mm for full UTC calendar days.

    ``tp`` is accumulated from the start of the run, so a day's rain is the difference
    between two consecutive midnights. A 00Z run yields 15 days; a 12Z run yields 14
    (the first 12 h do not make a full day and are dropped).

    Returns a DataArray (day, latitude, longitude) with a ``date`` coordinate.
    """
    valid = pd.DatetimeIndex(tp["valid_time"].values)
    is_midnight = (valid.hour == 0) & (valid.minute == 0)
    idx = np.flatnonzero(is_midnight)
    if len(idx) - 1 < min_days:
        raise ValueError(
            f"Only {max(len(idx) - 1, 0)} full forecast days available (need {min_days}); "
            "the model run is probably not a long-range 00/12 UTC run."
        )
    cumulative = tp.isel(step=idx).values.astype("float64")
    daily = np.diff(cumulative, axis=0) * 1000.0  # m -> mm
    daily = np.clip(daily, 0.0, None)  # GRIB packing can produce tiny negatives
    dates = valid[idx][:-1]
    return xr.DataArray(
        daily,
        dims=("day", "latitude", "longitude"),
        coords={
            "day": np.arange(len(dates)),
            "date": ("day", dates.values),
            "latitude": tp["latitude"].values,
            "longitude": tp["longitude"].values,
        },
        name="daily_rain_mm",
    )


# --------------------------------------------------------------------------- #
# 3. Consecutive-dry-day engine
# --------------------------------------------------------------------------- #
def consecutive_dry_days(dry):
    """Longest dry spell per cell, and the first day each spell reaches a threshold.

    ``dry`` is a boolean array (day, lat, lon). Streaks are counted from the first forecast
    day (dry days before the forecast are unknown).

    Returns ``(max_cdd, first_day)`` where ``max_cdd`` is an int array (lat, lon) and
    ``first_day[level]`` is the index of the day on which a streak first reached ``level``
    days (-1 where it never did).
    """
    dry = np.asarray(dry, dtype=bool)
    streak = np.zeros(dry.shape[1:], dtype=np.int16)
    max_cdd = np.zeros_like(streak)
    first_day = {level: np.full(streak.shape, -1, dtype=np.int16) for level in LEVELS}

    for t in range(dry.shape[0]):
        streak = np.where(dry[t], streak + 1, 0).astype(np.int16)
        max_cdd = np.maximum(max_cdd, streak)
        for level, first in first_day.items():
            first[(streak == level) & (first == -1)] = t
    return max_cdd, first_day


# --------------------------------------------------------------------------- #
# Land mask
# --------------------------------------------------------------------------- #
def land_mask(lats, lons, subsample=SUBSAMPLE, min_fraction=LAND_FRACTION):
    """Boolean (lat, lon) mask: True where at least ``min_fraction`` of the cell is land."""
    from global_land_mask import globe

    dlat = abs(lats[1] - lats[0])
    dlon = abs(lons[1] - lons[0])
    offsets = (np.arange(subsample) + 0.5) / subsample - 0.5
    sub_lat = (lats[:, None] + offsets[None, :] * dlat).ravel()
    sub_lon = (lons[:, None] + offsets[None, :] * dlon).ravel()
    sub_lon = ((sub_lon + 180) % 360) - 180
    land = globe.is_land(sub_lat[:, None], sub_lon[None, :])
    land = land.reshape(len(lats), subsample, len(lons), subsample)
    return land.mean(axis=(1, 3)) >= min_fraction


# --------------------------------------------------------------------------- #
# 4a. Threshold regions as proper polygons
# --------------------------------------------------------------------------- #
def cell_area_km2(lats, lons):
    """Area of each grid cell in km2, shape (lat, lon)."""
    dlat = abs(lats[1] - lats[0])
    dlon = abs(lons[1] - lons[0])
    km_per_deg = 111.32
    row = (dlat * km_per_deg) * (dlon * km_per_deg * np.cos(np.radians(lats)))
    return np.repeat(row[:, None], len(lons), axis=1)


def threshold_regions(max_cdd, first_day, level, lats, lons, dates):
    """GeoJSON features for each connected area whose longest dry spell is >= ``level`` days.

    ``max_cdd`` may contain NaN (masked cells, never part of a region). Each feature carries
    the earliest date at which any cell of that area reached ``level`` dry days, its peak CDD
    and its area. Holes are preserved; the outline follows the grid cells exactly.
    """
    mask = np.nan_to_num(max_cdd, nan=0.0) >= level
    labels, n = ndimage.label(mask)  # 4-connectivity, same as rasterio.shapes
    if n == 0:
        return []

    dlat = abs(lats[1] - lats[0])
    dlon = abs(lons[1] - lons[0])
    transform = from_origin(lons.min() - dlon / 2, lats.max() + dlat / 2, dlon, dlat)
    areas = cell_area_km2(lats, lons)
    index = np.arange(1, n + 1)
    peak = ndimage.maximum(np.nan_to_num(max_cdd, nan=0.0), labels, index)
    area = ndimage.sum(areas, labels, index)
    reached = np.where(first_day[level] >= 0, first_day[level], np.iinfo(np.int16).max)
    earliest = ndimage.minimum(reached, labels, index)

    features = []
    for geom, value in shapes(labels.astype("int32"), mask=labels > 0, transform=transform):
        i = int(value) - 1
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "earliest": pd.Timestamp(dates[int(earliest[i])]).strftime("%d %b %Y"),
                    "peak": f"{int(peak[i])} days",
                    "area": f"{int(round(area[i], -2)):,} km²",
                },
            }
        )
    return features


# --------------------------------------------------------------------------- #
# 4b. Map
# --------------------------------------------------------------------------- #
def mercator_rows(lats, n_rows):
    """For an image of ``n_rows`` equal Web-Mercator steps, the source row of each image row."""
    dlat = abs(lats[1] - lats[0])
    top, bottom = lats.max() + dlat / 2, lats.min() - dlat / 2

    def lat_to_y(lat):
        return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

    def y_to_lat(y):
        return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)

    y = np.linspace(lat_to_y(top), lat_to_y(bottom), n_rows + 1)
    centres = y_to_lat((y[:-1] + y[1:]) / 2)
    return np.abs(lats[None, :] - centres[:, None]).argmin(axis=1)  # lats descending -> row 0 = north


def raster_data_uri(max_cdd, lats):
    """RGBA PNG (Web-Mercator rows, transparent where dry-spell is 0 or masked) as a data URI."""
    cmap = LinearSegmentedColormap.from_list("cdd", RASTER_ANCHORS)
    rows = mercator_rows(lats, len(lats) * 2)
    resampled = max_cdd[rows, :]
    rgba = cmap(Normalize(vmin=0, vmax=RASTER_VMAX)(resampled))
    rgba[~np.isfinite(resampled) | (resampled == 0)] = [0, 0, 0, 0]
    buf = io.BytesIO()
    imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def encode_grid(arr):
    """uint8 grid (NODATA where NaN or negative) as base64, for the click read-out."""
    out = np.full(arr.shape, NODATA, dtype=np.uint8)
    valid = np.isfinite(arr) & (arr >= 0)
    out[valid] = np.clip(arr[valid], 0, NODATA - 1).astype(np.uint8)
    return base64.b64encode(out.tobytes()).decode()


CSS = """
<style>
  .leaflet-image-layer { image-rendering: pixelated; }
  .cdd-legend { position: fixed; bottom: 40px; left: 12px; width: 205px; max-width: calc(100vw - 30px);
    background: rgba(255,255,255,0.95); border: 2px solid #555; z-index: 9999; font: 11px sans-serif;
    padding: 6px 8px 8px; border-radius: 6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.4); }
  .cdd-legend summary { cursor: pointer; font-weight: bold; font-size: 12px; }
  .cdd-legend .row { display: flex; align-items: center; gap: 5px; margin-top: 2px; }
  .cdd-legend .swatch { width: 12px; height: 12px; box-sizing: border-box; flex: none; }
  .cdd-legend small { color: #555; display: block; margin-top: 4px; line-height: 1.3; }
  @media (max-width: 600px) { .cdd-legend { bottom: 70px; width: 175px; font-size: 10px; } }
</style>
"""

LEGEND = """
<details class="cdd-legend" id="cdd-legend" open>
  <summary>Fire Susceptibility</summary>
  Longest run of consecutive dry days (&lt; {dry_mm:g} mm/day)
  <div style="background: linear-gradient(to right, #FFFFCC 0%, #FECC5C 33.3%, #F03B20 66.6%, #BD0026 100%);
              height: 12px; border: 1px solid #aaa; margin: 6px 0 2px;"></div>
  <div style="display: flex; justify-content: space-between; font-size: 10px; font-weight: bold; margin-bottom: 6px;">
    <span>0</span><span>5</span><span>10</span><span>15+</span>
  </div>
  <b>Outlines</b> (layers menu, top right)
  <div class="row"><i class="swatch" style="background: rgba(0,0,255,0.2); border: 2px solid #0000FF;"></i> &ge; 5 days</div>
  <div class="row"><i class="swatch" style="background: rgba(255,0,255,0.2); border: 2px solid #FF00FF;"></i> &ge; 10 days</div>
  <div class="row"><i class="swatch" style="background: rgba(0,0,0,0.2); border: 2px solid #000000;"></i> &ge; 15 days</div>
  <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ccc;">
  <b>Model run:</b> {run}<br>
  <b>Forecast days:</b> {first} &rarr; {last}<br>
  <i>Click the map for the value at a point.</i>
  <small>Spells are counted from the first forecast day. {attribution}</small>
</details>
<script>
  if (window.innerWidth <= 600) document.getElementById('cdd-legend').removeAttribute('open');
</script>
"""

# Click read-out. Runs on window load because folium renders the map script after this element.
CLICK_JS = """
<script>
window.addEventListener('load', function () {{
  var map = window['{map_name}'];
  var meta = {meta};
  function decode(b64) {{
    var s = atob(b64), a = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
    return a;
  }}
  var grids = {{cdd: decode(meta.cdd), d5: decode(meta.d5), d10: decode(meta.d10), d15: decode(meta.d15)}};
  function when(v) {{ return v === {nodata} ? 'not reached' : meta.dates[v]; }}
  map.on('click', function (e) {{
    var r = Math.floor((meta.north - e.latlng.lat) / meta.dlat);
    var c = Math.floor((e.latlng.lng - meta.west) / meta.dlon);
    if (r < 0 || c < 0 || r >= meta.rows || c >= meta.cols) return;
    var k = r * meta.cols + c, v = grids.cdd[k];
    var html = '<b>' + e.latlng.lat.toFixed(2) + '&deg;, ' + e.latlng.lng.toFixed(2) + '&deg;</b><br>';
    if (v === {nodata}) {{ html += 'Ocean (not evaluated)'; }}
    else {{
      html += 'Longest dry spell: <b>' + v + (v === 1 ? ' day' : ' days') + '</b>' +
        '<br>&ge; 5 days: ' + when(grids.d5[k]) + '<br>&ge; 10 days: ' + when(grids.d10[k]) +
        '<br>&ge; 15 days: ' + when(grids.d15[k]);
    }}
    L.popup({{maxWidth: 240}}).setLatLng(e.latlng).setContent(html).openOn(map);
  }});
}});
</script>
"""


def build_map(max_cdd, first_day, dates, lats, lons, base_time, out_html, land_masked):
    dlat = abs(lats[1] - lats[0])
    dlon = abs(lons[1] - lons[0])
    north, south = lats.max() + dlat / 2, lats.min() - dlat / 2
    west, east = lons.min() - dlon / 2, lons.max() + dlon / 2
    bounds = [[south, west], [north, east]]

    m = folium.Map(location=[(north + south) / 2, (west + east) / 2], zoom_start=4, tiles=None, zoom_snap=0.25, zoom_delta=0.5)
    m.fit_bounds(bounds)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Imagery &copy; Esri, Maxar, Earthstar Geographics",
        name="Satellite",
        overlay=False,
        show=True,
    ).add_to(m)
    # Only one base layer may be added to the map at load; the other is switched in via the control.
    folium.TileLayer(tiles="CartoDB positron", name="Light map", overlay=False, show=False).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=raster_data_uri(max_cdd, lats),
        bounds=bounds,
        opacity=0.7,
        name="Consecutive dry days",
        interactive=False,
    ).add_to(m)

    # Outline layers
    for level, style in LEVELS.items():
        features = threshold_regions(max_cdd, first_day, level, lats, lons, dates)
        group = folium.FeatureGroup(
            name=f"<span style='color: {style['color']}; font-weight: bold;'>Area: &ge; {level} days</span>",
            show=False,
        )
        if features:
            fields = ["earliest", "peak", "area"]
            aliases = [f"First reached {level} dry days:", "Peak dry spell:", "Area:"]
            folium.GeoJson(
                {"type": "FeatureCollection", "features": features},
                style_function=lambda _f, s=style: {
                    "color": s["color"],
                    "weight": 2,
                    "fillColor": s["color"],
                    "fillOpacity": s["opacity"],
                },
                tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
                popup=folium.GeoJsonPopup(fields=fields, aliases=aliases),
            ).add_to(group)
        group.add_to(m)

    # Place names and borders on top of the satellite imagery
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Labels &copy; Esri",
        name="Labels",
        overlay=True,
    ).add_to(m)

    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    # Legend + click read-out
    date_labels = [pd.Timestamp(d).strftime("%d %b %Y") for d in dates]
    legend = LEGEND.format(
        dry_mm=DRY_DAY_MM,
        run=base_time.strftime("%Y-%m-%d %H:%M UTC"),
        first=date_labels[0],
        last=date_labels[-1],
        attribution=ATTRIBUTION + ("; ocean masked." if land_masked else "."),
    )
    meta = {
        "north": float(north),
        "west": float(west),
        "dlat": float(dlat),
        "dlon": float(dlon),
        "rows": int(len(lats)),
        "cols": int(len(lons)),
        "dates": date_labels,
        "cdd": encode_grid(max_cdd),
        "d5": encode_grid(first_day[5]),
        "d10": encode_grid(first_day[10]),
        "d15": encode_grid(first_day[15]),
    }
    root = m.get_root()
    root.header.add_child(folium.Element(CSS))
    root.html.add_child(folium.Element(legend))
    root.html.add_child(
        folium.Element(CLICK_JS.format(map_name=m.get_name(), meta=json.dumps(meta), nodata=NODATA))
    )
    m.save(str(out_html))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compute(tp, apply_land_mask=True):
    """Everything between the raw GRIB data and the map. Returns a dict of results."""
    daily = daily_rainfall_mm(tp)
    lats = daily["latitude"].values
    lons = daily["longitude"].values
    dates = daily["date"].values

    dry = daily.values < DRY_DAY_MM
    max_cdd, first_day = consecutive_dry_days(dry)

    max_cdd = max_cdd.astype("float32")
    if apply_land_mask:
        land = land_mask(lats, lons)
        max_cdd[~land] = np.nan
        for first in first_day.values():
            first[~land] = -1

    if not np.isfinite(max_cdd).any() or np.nanmax(max_cdd) <= 0:
        raise RuntimeError("Sanity check failed: no valid dry-spell values in the output grid.")
    return {"max_cdd": max_cdd, "first_day": first_day, "dates": dates, "lats": lats, "lons": lons}


def write_netcdf(result, path, base_time):
    dims = ("latitude", "longitude")
    data_vars = {
        "max_cdd": (dims, result["max_cdd"], {"long_name": "Longest consecutive dry spell", "units": "days"})
    }
    for level, first in result["first_day"].items():
        data_vars[f"first_day_ge_{level}"] = (
            dims,
            first.astype("int16"),
            {"long_name": f"Forecast day index on which a {level}-day dry spell was first reached (-1: never)"},
        )
    dates = result["dates"]
    ds = xr.Dataset(
        data_vars,
        coords={"latitude": result["lats"], "longitude": result["lons"]},
        attrs={
            "title": "ECMWF HRES forecast: consecutive dry days",
            "dry_day_threshold_mm": DRY_DAY_MM,
            "model_run": base_time.strftime("%Y-%m-%dT%H:%MZ"),
            "first_forecast_day": pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
            "last_forecast_day": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
            "source": "ECMWF open data (CC BY 4.0)",
        },
    )
    ds.to_netcdf(path, engine="h5netcdf")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="public", help="output directory (default: %(default)s)")
    p.add_argument("--grib", help="use this existing GRIB file instead of downloading (for testing)")
    p.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="comma-separated ECMWF mirrors to try, in order")
    p.add_argument("--no-land-mask", action="store_true", help="also show cells over the ocean")
    p.add_argument(
        "--max-age-hours",
        type=float,
        default=36.0,
        help="fail if the model run is older than this (0 disables; default: %(default)s)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = None if args.grib else Path(tempfile.mkdtemp(prefix="cdd_"))
    grib = Path(args.grib) if args.grib else workdir / "latest_15d_6h.grib"

    try:
        print("🚀 1/4 Retrieving ECMWF HRES precipitation...")
        if workdir:
            fetch_forecast(grib, sources=tuple(s for s in args.sources.split(",") if s))

        print("📂 2/4 Loading and cropping the region...")
        tp = load_precip(grib)
        base_time = pd.Timestamp(tp["time"].values)
        age_h = (pd.Timestamp.now(tz="UTC").tz_localize(None) - base_time) / pd.Timedelta(hours=1)
        if args.max_age_hours and age_h > args.max_age_hours:
            raise RuntimeError(f"Model run {base_time} is {age_h:.0f} h old (limit {args.max_age_hours:g} h).")

        print("🧠 3/4 Computing consecutive dry days...")
        result = compute(tp, apply_land_mask=not args.no_land_mask)
        print(f"   peak dry spell: {np.nanmax(result['max_cdd']):.0f} days")

        print("🗺️ 4/4 Writing NetCDF and map...")
        write_netcdf(result, out_dir / "cdd_se_asia.nc", base_time)
        build_map(
            result["max_cdd"],
            result["first_day"],
            result["dates"],
            result["lats"],
            result["lons"],
            base_time,
            out_dir / "index.html",
            land_masked=not args.no_land_mask,
        )
        dates = result["dates"]
        print(f"🎉 Done. Run {base_time:%Y-%m-%d %H:%M} UTC, days {pd.Timestamp(dates[0]):%Y-%m-%d} → {pd.Timestamp(dates[-1]):%Y-%m-%d}")
    finally:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
