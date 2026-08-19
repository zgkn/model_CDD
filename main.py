import os
import numpy as np
import pandas as pd
import xarray as xr
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import measure
from scipy.interpolate import interp1d
from ecmwf.opendata import Client

# File Targets
TARGET_GRIB = "latest_15d_6h.grib"
TARGET_NC = "cdd_se_asia.nc"
TARGET_HTML = "fire_susceptibility_map.html"

# Bounding Box
LAT_TOP, LAT_BOTTOM = 40, -30
LON_LEFT, LON_RIGHT = 90, 130

try:
    # SECTION 1: Dynamic Data Fetching
    print("🚀 Section 1/5: Initializing dynamic ECMWF HRES retrieval...")
    client = Client(source="azure")
    steps = list(range(0, 361, 6))

    client.retrieve(
        type="fc",
        levtype="sfc",
        param=["tp"],
        step=steps,
        target=TARGET_GRIB
    )
    print("✅ Section 1/5 Complete: Raw 15-day 6-hourly GRIB downloaded.")

    # SECTION 2: Spatial & Temporal Realignment
    print("📂 Section 2/5: Loading dataset and filtering UTC midnight boundaries...")
    ds = xr.open_dataset(TARGET_GRIB, engine="cfgrib")
    ds_region = ds.sel(latitude=slice(LAT_TOP, LAT_BOTTOM), longitude=slice(LON_LEFT, LON_RIGHT))

    raw_time = ds_region.time.values
    base_time = pd.to_datetime(raw_time if raw_time.ndim == 0 else raw_time[0])
    base_hour = base_time.hour

    if base_hour == 0:
        target_steps = list(range(0, 361, 24))
        ds_filtered = ds_region.sel(step=np.array(target_steps, dtype="timedelta64[h]"))
        daily_rain = ds_filtered["tp"].diff(dim="step")
        window_start = base_time
        window_end = base_time + pd.Timedelta(hours=360)
    elif base_hour == 12:
        target_steps = list(range(12, 349, 24))
        ds_filtered = ds_region.sel(step=np.array(target_steps, dtype="timedelta64[h]"))
        daily_rain = ds_filtered["tp"].diff(dim="step")
        window_start = base_time + pd.Timedelta(hours=12)
        window_end = base_time + pd.Timedelta(hours=348)
    else:
        raise ValueError(f"Unexpected model run hour: {base_hour}. Expected 00 or 12 UTC.")

    print("✅ Section 2/5 Complete: Chronological boundaries anchored to midnight UTC.")

    # SECTION 3: Accumulation & Threshold Masking
    print("🧮 Section 3/5: Scaling units and applying WMO fire risk thresholds...")
    daily_rain_mm = daily_rain * 1000.0
    dry_days_mask = xr.where(daily_rain_mm < 1.0, 1, 0)
    print("✅ Section 3/5 Complete: Rainfall matrices converted to binary drying maps.")

    # SECTION 4: CDD Core Engine Analysis
    print("🧠 Section 4/5: Running spatial matrix streak analyzer...")
    current_streak = xr.zeros_like(dry_days_mask.isel(step=0))
    max_cdd = xr.zeros_like(dry_days_mask.isel(step=0))

    step_reached_5 = xr.full_like(dry_days_mask.isel(step=0), -1, dtype=float)
    step_reached_10 = xr.full_like(dry_days_mask.isel(step=0), -1, dtype=float)
    step_reached_15 = xr.full_like(dry_days_mask.isel(step=0), -1, dtype=float)

    for t in range(len(dry_days_mask.step)):
        day_mask = dry_days_mask.isel(step=t)
        current_streak = xr.where(day_mask == 1, current_streak + 1, 0)
        max_cdd = xr.where(current_streak > max_cdd, current_streak, max_cdd)

        step_reached_5 = xr.where((current_streak == 5) & (step_reached_5 == -1), t, step_reached_5)
        step_reached_10 = xr.where((current_streak == 10) & (step_reached_10 == -1), t, step_reached_10)
        step_reached_15 = xr.where((current_streak == 15) & (step_reached_15 == -1), t, step_reached_15)

    max_cdd.name = "max_cdd"
    max_cdd.to_netcdf(TARGET_NC)

    if os.path.exists(TARGET_GRIB):
        os.remove(TARGET_GRIB)
    print(f"✅ Section 4/5 Complete: CDD analysis finished. Peak dry spell found: {float(max_cdd.max()):.0f} days.")

    # SECTION 5: Mobile-Optimized Folium Map Generation
    print("🗺️ Section 5/5: Compiling Web-Mercator aligned HTML map with vector polygons...")
    cdd_data = max_cdd.values
    lats = max_cdd.latitude.values
    lons = max_cdd.longitude.values

    # 5A. WEB MERCATOR PROJECTION CORRECTION
    dlat = abs(lats[1] - lats[0])
    dlon = abs(lons[1] - lons[0])
    lat_edge_top = np.max(lats) + (dlat / 2)
    lat_edge_bot = np.min(lats) - (dlat / 2)
    lon_edge_left = np.min(lons) - (dlon / 2)
    lon_edge_right = np.max(lons) + (dlon / 2)
    img_bounds = [[lat_edge_bot, lon_edge_left], [lat_edge_top, lon_edge_right]]

    def lat_to_y(lat): return np.log(np.tan(np.pi/4 + np.radians(lat)/2))
    def y_to_lat(y): return np.degrees(2 * np.arctan(np.exp(y)) - np.pi/2)

    y_max = lat_to_y(lat_edge_top)
    y_min = lat_to_y(lat_edge_bot)
    y_grid = np.linspace(y_max, y_min, len(lats) * 2)
    lat_grid_merc = y_to_lat(y_grid)

    sort_idx = np.argsort(lats)
    lats_sorted = lats[sort_idx]
    cdd_sorted = cdd_data[sort_idx, :]

    interpolator = interp1d(lats_sorted, cdd_sorted, axis=0, kind='nearest', bounds_error=False, fill_value=np.nan)
    cdd_merc = interpolator(lat_grid_merc)

    # 5B. RASTER LAYER
    color_anchors = [
        (0.0, '#FFFFCC'),
        (0.333, '#FECC5C'),
        (0.666, '#F03B20'),
        (1.0, '#BD0026')
    ]
    cmap = LinearSegmentedColormap.from_list('custom_cdd', color_anchors)
    norm = plt.Normalize(vmin=0, vmax=15)
    rgba_img = cmap(norm(cdd_merc))

    rgba_img[cdd_merc == 0] = [0, 0, 0, 0]
    rgba_img[np.isnan(cdd_merc)] = [0, 0, 0, 0]

    temp_img_path = "temp_layer.png"
    plt.imsave(temp_img_path, rgba_img)

    m = folium.Map(
        location=[(LAT_TOP + LAT_BOTTOM) / 2, (LON_LEFT + LON_RIGHT) / 2],
        zoom_start=4,
        tiles=None
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr=" ",
        name="Satellite Basemap",
        overlay=False,
        control=True
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=temp_img_path,
        bounds=img_bounds,
        opacity=0.65,
        name="Base Raster: Consecutive Dry Days",
        interactive=True
    ).add_to(m)

    # 5C. VECTOR POLYGON LAYERS
    thresholds = {
        5:  {'name': 'Area: >= 5 Days',  'color': '#0000FF', 'op': 0.15, 'matrix': step_reached_5.values},
        10: {'name': 'Area: >= 10 Days', 'color': '#FF00FF', 'op': 0.20, 'matrix': step_reached_10.values},
        15: {'name': 'Area: >= 15 Days', 'color': '#000000', 'op': 0.25, 'matrix': step_reached_15.values}
    }

    cdd_filled = np.nan_to_num(cdd_data, nan=0.0)

    for level, props in thresholds.items():
        fg = folium.FeatureGroup(name=f"<span style='color: {props['color']}; font-weight: bold;'>{props['name']}</span>", show=False)
        contours = measure.find_contours(cdd_filled, level)

        for contour in contours:
            lat_vals = np.interp(contour[:, 0], np.arange(len(lats)), lats)
            lon_vals = np.interp(contour[:, 1], np.arange(len(lons)), lons)
            poly_coords = list(zip(lat_vals, lon_vals))

            r_min, r_max = int(np.floor(contour[:, 0].min())), int(np.ceil(contour[:, 0].max()))
            c_min, c_max = int(np.floor(contour[:, 1].min())), int(np.ceil(contour[:, 1].max()))

            matrix_reached = props['matrix']
            block = matrix_reached[max(0, r_min):r_max+1, max(0, c_min):c_max+1]
            valid_steps = block[block != -1]

            if len(valid_steps) > 0:
                earliest_step = int(np.min(valid_steps))
                reached_date = window_start + pd.Timedelta(days=earliest_step + 1)
                date_str = reached_date.strftime('%Y-%m-%d')
                label_html = f"<b>{props['name']}</b><br>📅 Earliest Date Reached: <b>{date_str}</b>"
            else:
                label_html = f"<b>{props['name']}</b>"

            folium.Polygon(
                locations=poly_coords,
                color=props['color'],
                weight=2,
                fill=True,
                fill_color=props['color'],
                fill_opacity=props['op'],
                tooltip=folium.Tooltip(label_html, sticky=True),
                popup=folium.Popup(label_html, max_width=300)
            ).add_to(fg)

        fg.add_to(m)

    # 5D. CONTROLS & LEGEND
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    run_str = base_time.strftime('%Y-%m-%d %H:%M UTC')
    start_str = window_start.strftime('%Y-%m-%d')
    end_str = window_end.strftime('%Y-%m-%d')

    legend_html = f"""
     <div style="position: fixed; bottom: 25px; left: 25px; width: 190px; height: auto; min-height: 200px;
                 background-color: rgba(255, 255, 255, 0.95); border:2px solid #555; z-index:9999; font-size:11px;
                 padding: 8px 8px 12px 8px; border-radius: 6px; font-family: sans-serif; box-shadow: 2px 2px 6px rgba(0,0,0,0.4);">
     <b>Fire Susceptibility</b><br>
     Max CDD (&lt; 1mm)<br>
     <div style="background: linear-gradient(to right, #FFFFCC 0%, #FECC5C 33.3%, #F03B20 66.6%, #BD0026 100%);
                 height: 12px; width: 100%; border: 1px solid #aaa; margin-top: 8px; margin-bottom: 2px;"></div>
     <div style="display: flex; justify-content: space-between; font-size: 10px; font-weight: bold; margin-bottom: 8px;">
         <span>0</span>
         <span>5</span>
         <span>10</span>
         <span>15+</span>
     </div>
     <b>Toggle Areas:</b><br>
     <i style="background:rgba(0,0,255,0.2); width:12px; height:12px; float:left; margin-right:5px; border:2px solid #0000FF;"></i> &ge; 5 Days<br>
     <i style="background:rgba(255,0,255,0.2); width:12px; height:12px; float:left; margin-right:5px; border:2px solid #FF00FF;"></i> &ge; 10 Days<br>
     <i style="background:rgba(0,0,0,0.2); width:12px; height:12px; float:left; margin-right:5px; border:2px solid #000000;"></i> &ge; 15 Days<br>
     <hr style="margin:6px 0; border:0; border-top:1px solid #ccc;">
     <b>Model Run:</b> {run_str}<br>
     <b>Start:</b> {start_str}<br>
     <b>End:</b> {end_str}
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(TARGET_HTML)

    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    print(f"🎉 Pipeline Success!")
    print(f"• Model Run: {run_str}")
    print(f"• Window Start: {start_str}")
    print(f"• Window End: {end_str}")
    print(f"• Domain: 90E-130E, 30S-40N")

except Exception as err:
    print(f"🚨 Pipeline Crash! Error trace: {str(err)}")
    raise err

