import numpy as np
import pandas as pd
import pytest
import xarray as xr

import main


def make_tp(run_hour, daily_mm, lats=(1.0, 0.0), lons=(100.0, 101.0)):
    """Cumulative 6-hourly tp (metres) from per-day rainfall in mm, shape (n_days, lat, lon).

    All of a day's rain falls in the 6 h before 06 UTC, which makes day boundaries easy to check.
    """
    daily_mm = np.asarray(daily_mm, dtype="float64")
    base = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=run_hour)
    steps = np.arange(0, 361, 6)
    valid = base + pd.to_timedelta(steps, unit="h")
    rain = np.zeros((len(steps),) + daily_mm.shape[1:])
    for d in range(daily_mm.shape[0]):
        day_start = pd.Timestamp("2026-01-01") + pd.Timedelta(days=d + (1 if run_hour == 12 else 0))
        # the 6 h window ending at 06 UTC of that calendar day
        k = np.flatnonzero(valid == day_start + pd.Timedelta(hours=6))
        if k.size:
            rain[k[0]] = daily_mm[d] / 1000.0
    cumulative = np.cumsum(rain, axis=0)
    return xr.DataArray(
        cumulative,
        dims=("step", "latitude", "longitude"),
        coords={
            "step": pd.to_timedelta(steps, unit="h"),
            "valid_time": ("step", valid),
            "time": base,
            "latitude": list(lats),
            "longitude": list(lons),
        },
    )


# ---- consecutive_dry_days ------------------------------------------------- #
def test_streaks_reset_and_first_reached():
    #            day: 0  1  2  3  4  5  6  7  8
    cell = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)  # 5 dry, wet, 3 dry
    dry = cell.reshape(-1, 1, 1)
    max_cdd, first = main.consecutive_dry_days(dry)
    assert max_cdd[0, 0] == 5
    assert first[5][0, 0] == 4  # fifth dry day is index 4
    assert first[10][0, 0] == -1 and first[15][0, 0] == -1


def test_longest_spell_not_last_spell():
    dry = np.array([0, 1, 1, 1, 0, 1], dtype=bool).reshape(-1, 1, 1)
    assert main.consecutive_dry_days(dry)[0][0, 0] == 3


def test_all_dry_reaches_every_level():
    dry = np.ones((15, 2, 3), dtype=bool)
    max_cdd, first = main.consecutive_dry_days(dry)
    assert (max_cdd == 15).all()
    assert (first[5] == 4).all() and (first[10] == 9).all() and (first[15] == 14).all()


# ---- daily_rainfall_mm ---------------------------------------------------- #
@pytest.mark.parametrize("run_hour,n_days", [(0, 15), (12, 14)])
def test_daily_rainfall_is_calendar_days(run_hour, n_days):
    rain = np.zeros((15, 2, 2))
    rain[3] = 7.5  # all 7.5 mm on the 4th usable calendar day
    if run_hour == 12:
        rain = rain[:14]
    tp = make_tp(run_hour, rain)
    daily = main.daily_rainfall_mm(tp)
    assert daily.sizes["day"] == n_days
    assert daily.values[3].max() == pytest.approx(7.5, abs=1e-6)
    assert np.delete(daily.values, 3, axis=0).max() == 0
    first = pd.Timestamp(daily["date"].values[0])
    expected = pd.Timestamp("2026-01-01") + (pd.Timedelta(days=1) if run_hour == 12 else pd.Timedelta(0))
    assert first == expected and first.hour == 0


def test_short_run_is_rejected():
    tp = make_tp(0, np.zeros((15, 2, 2))).isel(step=slice(0, 9))  # only 48 h
    with pytest.raises(ValueError):
        main.daily_rainfall_mm(tp)


def test_negative_packing_noise_is_clipped():
    tp = make_tp(0, np.zeros((15, 2, 2)))
    tp = tp.copy(data=tp.values - 1e-9 * np.arange(len(tp.step))[:, None, None])
    assert main.daily_rainfall_mm(tp).values.min() >= 0


# ---- threshold regions ---------------------------------------------------- #
def grid(n=8, step=0.25):
    lats = 10 - step * np.arange(n)
    lons = 100 + step * np.arange(n)
    return lats, lons


def test_region_includes_cells_equal_to_level():
    lats, lons = grid()
    cdd = np.zeros((8, 8), dtype="float32")
    cdd[2:4, 2:4] = 5  # a 2x2 block of exactly 5
    first = {5: np.full((8, 8), -1, np.int16)}
    first[5][2:4, 2:4] = 4
    dates = pd.date_range("2026-01-01", periods=15).values
    feats = main.threshold_regions(cdd, first, 5, lats, lons, dates)
    assert len(feats) == 1
    xs, ys = zip(*feats[0]["geometry"]["coordinates"][0])
    # the outline hugs the cell edges: 2 cells x 0.25 deg wide, not a degenerate line
    assert max(xs) - min(xs) == pytest.approx(0.5)
    assert max(ys) - min(ys) == pytest.approx(0.5)
    # first reached on day index 4 -> 5 Jan, not the following day
    assert feats[0]["properties"]["earliest"] == "05 Jan 2026"


def test_regions_are_separate_with_own_dates_and_keep_holes():
    lats, lons = grid(n=12)
    cdd = np.zeros((12, 12), dtype="float32")
    cdd[0:7, 0:7] = 6
    cdd[3, 3] = 0  # a hole
    cdd[9:11, 9:11] = 8  # separate island
    first = {5: np.full((12, 12), -1, np.int16)}
    first[5][0:7, 0:7] = 4
    first[5][3, 3] = -1
    first[5][9:11, 9:11] = 7
    dates = pd.date_range("2026-01-01", periods=15).values
    feats = main.threshold_regions(cdd, first, 5, lats, lons, dates)
    assert len(feats) == 2
    by_date = {f["properties"]["earliest"]: f for f in feats}
    assert set(by_date) == {"05 Jan 2026", "08 Jan 2026"}
    assert len(by_date["05 Jan 2026"]["geometry"]["coordinates"]) == 2  # exterior + hole
    assert by_date["08 Jan 2026"]["properties"]["peak"] == "8 days"


def test_nan_cells_are_never_in_a_region():
    lats, lons = grid()
    cdd = np.full((8, 8), np.nan, dtype="float32")
    first = {5: np.full((8, 8), -1, np.int16)}
    dates = pd.date_range("2026-01-01", periods=15).values
    assert main.threshold_regions(cdd, first, 5, lats, lons, dates) == []


# ---- land mask, projection, end to end ------------------------------------ #
def test_land_mask_keeps_land_and_drops_sea():
    lats = np.array([-6.25, -4.75])  # Java, Java Sea
    lons = np.array([107.0, 110.0])
    land = main.land_mask(lats, lons)
    assert land[0, 0]  # near Bandung
    assert not land[1, 1]  # open sea
    # a ~25 km island cell is not lost to point sampling
    assert main.land_mask(np.array([-8.75, -8.5]), np.array([115.0, 115.25])).any()  # Bali


def test_mercator_rows_are_monotonic_and_in_range():
    lats = np.arange(40, -30.01, -0.25)
    rows = main.mercator_rows(lats, len(lats) * 2)
    assert rows.min() == 0 and rows.max() == len(lats) - 1
    assert (np.diff(rows) >= 0).all()


def test_compute_and_build_map(tmp_path):
    rain = np.full((15, 2, 2), 5.0)
    rain[:, 0, 0] = 0.0  # always dry -> 15 days
    tp = make_tp(0, rain, lats=(40.0, 39.75), lons=(100.0, 100.25))  # land in Mongolia/China
    result = main.compute(tp, apply_land_mask=False)
    assert result["max_cdd"][0, 0] == 15 and result["max_cdd"][1, 1] == 0
    base = pd.Timestamp("2026-01-01")
    main.write_netcdf(result, tmp_path / "x.nc", base)
    main.build_map(
        result["max_cdd"], result["first_day"], result["dates"], result["lats"], result["lons"],
        base, tmp_path / "index.html", land_masked=False,
    )
    html = (tmp_path / "index.html").read_text()
    assert "Fire Susceptibility" in html and "ECMWF" in html and "Esri" in html
    assert xr.open_dataset(tmp_path / "x.nc")["max_cdd"].max() == 15


def test_all_masked_output_fails_sanity_check():
    tp = make_tp(0, np.full((15, 2, 2), 5.0), lats=(-10.0, -10.25), lons=(80.0, 80.25))  # Indian Ocean
    with pytest.raises(RuntimeError):
        main.compute(tp, apply_land_mask=True)
