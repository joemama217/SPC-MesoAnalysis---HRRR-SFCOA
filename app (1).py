import io
import os
import re
import glob
import time
import traceback
import gc
import threading
import base64
import requests
from datetime import datetime, timedelta, timezone

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import cartopy.io.shapereader as shpreader
from shapely.geometry import box as shapely_box
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree

# --- PAGE CONFIG ---
st.set_page_config(page_title="SPC Mesoanalysis", page_icon="🌪️", layout="wide")
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🌪️ SPC Mesoanalysis Dashboard")

# --- COLOR GRADIENTS ---
_lr_grad = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list(
    'lr_grad', ['orange', 'darkred'])(np.linspace(0, 1, 10))]
_lcl_green_grad = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list(
    'lcl_green', ['darkgreen', 'lightgreen'])(np.linspace(0, 1, 4))]
_ship_grad = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list(
    'ship_grad', ['red', '#b19cd9'])(np.linspace(0, 1, 9))]
_lhp_grad = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list(
    'lhp_grad', ['#ffdd00', '#ff0000'])(np.linspace(0, 1, 8))]
_sars_base_grad = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list(
    'sars_base', ['#7b68ee', '#ff00ff'])(np.linspace(0, 1, 6))]
_ehi_grad = [mcolors.to_hex(c) for c in mcolors.LinearSegmentedColormap.from_list(
    'ehi_grad', ['#00cc00', '#995511', '#e60000'])(np.linspace(0, 1, 10))]

PROJECTION = ccrs.LambertConformal(
    central_longitude=-95.0,
    central_latitude=25.0,
    standard_parallels=(25.0, 25.0)
)


# Sharper maps on high-DPI (Retina / 4K) screens; the browser scales them down to fit.
# Lower to 115 if pages load slowly on poor connections.
RENDER_DPI = 150

# Map line styling (SPC look). Tweak here to change every map at once.
STATE_LINE_COLOR = '#a4a2a4'   # SPC's state-line gray
STATE_LINE_WIDTH = 0.9
CWA_LINE_COLOR = '#fc0204'     # SPC's red CWA outlines
CWA_LINE_WIDTH = 0.45
ISOBAR_WIDTH = 1.4             # MSLP isobars on every map (was 1.8-2.0)
HEIGHT_LINE_WIDTH = 1.2        # upper-air height lines (was 1.5)

# Layering, top to bottom: isobars/heights > everything plotted > red CWA outlines > gray state lines.
# (Swap the two base-map numbers to put state lines over the CWA outlines instead.)
ISOBAR_ZORDER = 5.0
CWA_ZORDER = 1.3
STATE_ZORDER = 1.2
COUNTY_ZORDER = 1.1
COUNTY_LINE_COLOR = '#cccacc'  # a shade lighter than the state gray
COUNTY_LINE_WIDTH = 0.25       # thinner than the CWA outlines
OCEAN_COLOR = 'white'          # SPC leaves water white


# --- NOMADS: ONE SHARED RUN LIST + ONE-TIME FILE DOWNLOADS ---
# SPEEDUP: previously every slider position scraped NOMADS separately (its own 5-minute cache per
# offset) and then downloaded its file while you waited. Now one cached directory listing serves
# every offset, each file is downloaded exactly once, and older hours are prefetched in the
# background so the slider is ready before you touch it.
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod"
MAX_PAST_HOURS = 3


def _nomads_session():
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Python/MesoViewer-Web',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache'
    })
    return s


@st.cache_resource(show_spinner=False)
def _shared_state():
    """Process-wide state that survives Streamlit reruns (module globals do not)."""
    return {'lock': threading.Lock(), 'file_locks': {}, 'prefetched': set(),
            'compute_lock': threading.Lock(),
            'render_slots': threading.BoundedSemaphore(MAX_CONCURRENT_RENDERS)}


@st.cache_data(ttl=300, show_spinner=False)
def list_available_runs():
    """Newest-first list of (date_str, filename, cycle_hour, ana_type) for the last 2 UTC days.
    Prefers 'final' over earlier analyses for the same hour."""
    now = datetime.now(timezone.utc)
    checked_epoch_ms = int(now.timestamp() * 1000)
    checked_utc_str = now.strftime("%H:%M:%S UTC")
    session = _nomads_session()
    runs = []
    for day_offset in range(2):
        date_str = (now - timedelta(days=day_offset)).strftime("%Y%m%d")
        try:
            resp = session.get(f"{NOMADS_BASE}/spc_post.{date_str}/sfcoa-hrrr/?cb={time.time_ns()}", timeout=10)
            if resp.status_code != 200:
                continue
            # 'firstguess' per the SPC_POST v3 notice; 'prel' kept in case it is still used
            matches = re.findall(
                r'href="(spc_post\.t(\d{2})z\.sfcoa_hrrr\.(final|prel|firstguess)\.nc)"', resp.text)
            by_hour = {}
            for fname, cyc, atype in matches:
                prio = 1 if atype == 'final' else 0
                if cyc not in by_hour or prio > by_hour[cyc][0]:
                    by_hour[cyc] = (prio, fname, atype)
            for cyc, (prio, fname, atype) in by_hour.items():
                runs.append(((int(date_str), int(cyc)), date_str, fname, cyc, atype))
        except Exception:
            continue
    if not runs:
        raise RuntimeError("No SFCOA-HRRR NetCDF files found on NOMADS.")
    runs.sort(key=lambda r: r[0], reverse=True)
    return [r[1:] for r in runs], checked_epoch_ms, checked_utc_str


def fetch_run_file(date_str, fname):
    """Download one run exactly once. Writes to a .part file and renames when complete, so a
    half-downloaded file is never opened (important now that downloads also happen in the background)."""
    local_path = f"{date_str}_{fname}"
    state = _shared_state()
    with state['lock']:
        flock = state['file_locks'].setdefault(local_path, threading.Lock())
    with flock:
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        tmp_path = local_path + ".part"
        with _nomads_session().get(f"{NOMADS_BASE}/spc_post.{date_str}/sfcoa-hrrr/{fname}", stream=True, timeout=60) as dl:
            dl.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in dl.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, local_path)

        # Keep the newest 5 runs on disk (latest + 3 past hours + 1 spare) to save disk space
        cached = sorted(glob.glob("*_spc_post.t*z.sfcoa_hrrr.*.nc"), key=os.path.getmtime, reverse=True)
        for old_f in cached[5:]:
            try:
                os.remove(old_f)
            except OSError:
                pass
    return local_path


def download_sfcoa(hour_offset=0):
    """Same return values as before: (local_path, note, checked_epoch_ms, checked_utc_str).
    hour_offset 0 = latest, -1..-6 = previous hourly analyses."""
    runs, checked_epoch_ms, checked_utc_str = list_available_runs()
    note = None
    idx = min(abs(min(hour_offset, 0)), len(runs) - 1)
    if abs(hour_offset) > idx:
        note = f"Requested {hour_offset:+d}h; showing the oldest cycle available on NOMADS."
    date_str, fname, _, _ = runs[idx]
    return fetch_run_file(date_str, fname), note, checked_epoch_ms, checked_utc_str


def start_background_prefetch(parameter, wfo):
    """After the current map is on screen: download the previous 6 hours (once per file) and
    pre-render the current parameter/sector for each, so moving the slider is instant."""
    try:
        runs, _, _ = list_available_runs()
    except Exception:
        return
    job_key = (runs[0][1], parameter, wfo)  # newest file + what the user is looking at
    state = _shared_state()
    with state['lock']:
        if job_key in state['prefetched']:
            return
        state['prefetched'].add(job_key)

    def _worker():
        for date_str, fname, _, _ in runs[1:MAX_PAST_HOURS + 1]:
            try:
                path = fetch_run_file(date_str, fname)
                # MEMORY: pre-draw only sector views; a Full CONUS map is drawn when someone asks for it
                if wfo != 'Full CONUS':
                    render_map_png(path, parameter, wfo)  # warms the shared map cache
            except Exception:
                continue

    threading.Thread(target=_worker, daemon=True).start()


# --- CACHING GEOMETRIES & PRE-PROJECTED COORDINATES ---
@st.cache_resource
def load_geometries():
    domains = {'Full CONUS': [-126.0, -66.0, 23.0, 50.0]}
    cwa_geoms = None
    try:
        cwa_shapefile = "https://www.weather.gov/source/gis/Shapefiles/WSOM/w_18mr25.zip"
        reader = Reader(cwa_shapefile)
        records = list(reader.records())
        cwa_geoms = [rec.geometry for rec in records]
        parsed = {}
        for rec in records:
            wfo_id = rec.attributes.get('WFO', '').strip()
            if wfo_id:
                bounds = rec.geometry.bounds
                if len(bounds) == 4:
                    parsed[f"{wfo_id} ({rec.attributes.get('NAME', 'WFO')})"] = [
                        bounds[0] - 0.5, bounds[2] + 0.5,
                        bounds[1] - 0.5, bounds[3] + 0.5
                    ]
        domains.update(dict(sorted(parsed.items())))
    except Exception:
        domains.update({
            'SLC (Salt Lake City, UT)': [-116.5, -108.5, 36.5, 42.5],
            'GJT (Grand Junction, CO)': [-111.0, -105.0, 36.0, 41.5],
            'BOI (Boise, ID)': [-120.5, -114.0, 41.5, 46.0],
            'PIH (Pocatello, ID)': [-116.5, -110.5, 41.5, 45.5],
            'LKN (Elko, NV)': [-120.0, -113.5, 38.0, 42.5],
            'VEG (Las Vegas, NV)': [-119.0, -112.5, 34.0, 39.0]
        })
    return domains, cwa_geoms


WFO_DOMAINS, _CWA_GEOMETRY_CACHE = load_geometries()

# --- Extra base-map layers: U.S. counties (Census cartographic boundaries) and two lakes ---
_NON_CONUS_STATEFP = {'02', '15', '60', '66', '69', '72', '78'}   # AK, HI, territories


@st.cache_resource(show_spinner=False)
def load_county_geometries(scale):
    """Census county outlines. scale '20m' (light, for Full CONUS) or '5m' (more detail, for CWA views)."""
    for year in (2024, 2023):
        url = f"https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_county_{scale}.zip"
        try:
            return [rec.geometry for rec in Reader(url).records()
                    if str(rec.attributes.get('STATEFP', '')) not in _NON_CONUS_STATEFP]
        except Exception:
            continue
    return []


@st.cache_resource(show_spinner=False)
def load_extra_lakes():
    """Great Salt Lake and Lake Okeechobee outlines (Natural Earth 10m lakes; only these two are kept)."""
    try:
        path = shpreader.natural_earth(resolution='10m', category='physical', name='lakes')
        keep = []
        for rec in shpreader.Reader(path).records():
            name = str(rec.attributes.get('name') or rec.attributes.get('NAME') or '').lower()
            if 'great salt' in name or 'okeechobee' in name:
                keep.append(rec.geometry)
        return keep
    except Exception:
        return []


def _near(geoms, lon_min, lon_max, lat_min, lat_max, pad=1.0):
    w, e, s, n = lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad
    return [g for g in geoms if g.bounds[0] < e and g.bounds[2] > w and g.bounds[1] < n and g.bounds[3] > s]



@st.cache_resource(show_spinner=False)
def _projected_grid_store():
    return {}


# SPEEDUP: room for the latest run + 6 past hours (was 4, so moving the slider evicted files and
# re-opened them), and the HRRR grid is projected once and reused for every file instead of per file.
@st.cache_resource(max_entries=4, show_spinner=False)
def get_cached_dataset_and_grid(file_path):
    """Opens the NetCDF once and pre-projects (lon, lat) -> native Lambert Conformal (x, y) meters."""
    # MEMORY: cache=False stops xarray from keeping every full-CONUS variable it has read in RAM
    ds = xr.open_dataset(file_path, engine='h5netcdf', cache=False).squeeze()
    lons = ds.coords['longitude'].values
    lats = ds.coords['latitude'].values
    grid_key = (lons.shape, float(lons[0, 0]), float(lats[0, 0]), float(lons[-1, -1]), float(lats[-1, -1]))
    store = _projected_grid_store()
    if grid_key not in store:
        pts = PROJECTION.transform_points(ccrs.PlateCarree(), lons, lats)
        store[grid_key] = (pts[..., 0], pts[..., 1])
    x_proj, y_proj = store[grid_key]
    return ds, lons, lats, x_proj, y_proj


# SPEEDUP: crop the dataset itself to the sector BEFORE any calculation.
# Previously every derived field (metpy, divergence, smoothing, and the 3D DCAPE / 0-3km MLCAPE /
# P-type / SNSQ / DGZ fallbacks) ran on all ~1.9M CONUS points and was only cropped afterward
# for plotting. The crop is lazy, so only the sector's slice of each variable is read from disk.
DOMAIN_PAD_DEG = 3.0  # > the 1.5 deg plot pad + room for sigma=15 smoothing, so no edge artifacts in view

@st.cache_resource(max_entries=16, show_spinner=False)
def get_domain_dataset_and_grid(file_path, wfo):
    ds, lons, lats, x_proj, y_proj = get_cached_dataset_and_grid(file_path)
    if wfo == 'Full CONUS' or wfo not in WFO_DOMAINS:
        return ds, lons, lats, x_proj, y_proj

    lon_min, lon_max, lat_min, lat_max = WFO_DOMAINS[wfo]
    p = DOMAIN_PAD_DEG
    mask = (lons >= lon_min - p) & (lons <= lon_max + p) & (lats >= lat_min - p) & (lats <= lat_max + p)
    if not np.any(mask):
        return ds, lons, lats, x_proj, y_proj
    rows, cols = np.where(mask)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    y_dim, x_dim = ds['tmpk_2m'].dims[-2:]
    ds_c = ds.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)})
    return (ds_c, lons[r0:r1, c0:c1], lats[r0:r1, c0:c1],
            x_proj[r0:r1, c0:c1], y_proj[r0:r1, c0:c1])


# --- PARAMETERS DICTIONARY ---
PARAMETERS = {
    'Mean Sea Level Pressure (hPa)': {
        'var_name': 'pres_msl',
        'scale': 0.01,
        'color': 'gray',
        'contour_range': (950, 1060, 2),
        'title': 'Mean Sea Level Pressure (hPa)',
        'plot_type': 'contour'
    },
    'Surface Temp, Dewpoint, and PMSL': {
        'title': 'Surface Temp (°F), Dewpoint (°F), PMSL (hPa) & Wind',
        'plot_type': 'sfc_composite',
        # --- Matched to SPC's "Surface temp, dewpoint, and pmsl" map ---
        'ocean_color': 'none',                       # SPC leaves water white; keeps the hatching readable
        'mslp_levels': np.arange(900, 1080, 4),      # black solid, every 4 hPa
        'temp_levels': np.arange(-40, 125, 5),       # solid, every 5 F
        'temp_cold_max': 30, 'temp_mild_max': 55,    # <=30 purple | 35-55 goldenrod | >=60 red
        'temp_colors': ('#8c6acc', '#cc8604', '#fc0204'),
        'freeze_color': '#942eec',                   # bold 32 F line
        'dwpt_levels': np.arange(-40, 92, 4),        # blue dashed, every 4 F
        'dwpt_color': '#1c92fc',
        'dwpt_fill_levels': [56, 60, 64, 68, 72, 200],                         # hatched shading >= 56 F
        'dwpt_fill_colors': ['#04eeec', '#048a04', '#04ce04', '#7cfe04', '#fcd604'],
        'barb_color': '#8c4624',
        'smooth_sigma_km': 12.0,                     # light smoothing of 3-km HRRR noise for readability
    },
    'Mixing Ratio / Theta': {
        'title': 'Theta (K, Red) & Mixing Ratio (g/kg, Filled)',
        'plot_type': 'composite',
        'mr_levels': [7, 9, 11, 13, 15, 17, 20, 22],
        'mr_colors': ['#FCECE7', 'cyan', 'forestgreen', 'limegreen', 'greenyellow', 'gold', 'lightpink']
    },
    '10m Wind Speed (kt)': {
        'scale': 1.94384,
        'cmap': 'Purples',
        'contour_range': (0, 60, 5),
        'title': '10m Wind Speed (kt) & Barbs',
        'plot_type': 'filled'
    },
    '850mb Wind Speed (kt)': {
        'level': 850,
        'scale': 1.94384,
        'cmap': 'YlOrRd',
        'contour_range': (25, 80, 5),
        'title': '850mb Wind Speed (kt), Barbs, Temp (°C) & Heights (dam)',
        'plot_type': 'upper_air',
        'height_levels': np.arange(120, 165, 3),
        'temp_levels': np.arange(-20, 35, 2),
        'temp_dynamic_color': True
    },
    '700mb Wind Speed (kt)': {
        'level': 700,
        'scale': 1.94384,
        'cmap': 'Greens',
        'contour_range': (70, 101, 10),
        'title': '700mb Wind Barbs, Temp (°C), Heights (dam) & 700-500mb Mean RH >= 70%',
        'plot_type': 'upper_air',
        'height_levels': np.arange(270, 330, 3),
        'temp_levels': np.arange(-30, 20, 2),
        'temp_dynamic_color': True,
        'plot_rh': True
    },
    '500mb Wind Speed (kt)': {
        'level': 500,
        'scale': 1.94384,
        'cmap': 'YlOrRd',
        'contour_range': (40, 120, 5),
        'title': '500mb Wind Speed (kt), Barbs, Temp (°C) & Heights (dam)',
        'plot_type': 'upper_air',
        'height_levels': np.arange(500, 610, 6),
        'temp_levels': np.arange(-40, 10, 2),
        'temp_dynamic_color': False
    },
    '300mb Wind Speed (kt)': {
        'level': 300,
        'scale': 1.94384,
        'cmap': 'cool',
        'contour_range': (60, 150, 5),
        'title': '300mb Wind Speed (kt), Barbs, Heights (dam) & Divergence (1e-5/s)',
        'plot_type': 'upper_air',
        'height_levels': np.arange(804, 1150, 12),
        'plot_divergence': True,
        'div_levels': np.arange(2, 12, 2)
    },
    'MSLP, Theta-E, & Wind': {
        'title': 'MSLP (hPa), Theta-E (K) & 10m Wind',
        'plot_type': 'theta_e_composite',
        # --- Matched to SPC's "MSLP, Theta-E, Sfc Wind" map ---
        'ocean_color': 'none',
        'mslp_levels': np.arange(900, 1080, 2),      # black solid, every 2 hPa
        'te_levels': np.arange(230, 392, 2),         # dashed, every 2 K (old range stopped at 340 K)
        # color by value: <=322 blue | 324-334 green | 336-346 orange | >=348 red
        'te_color_breaks': [(322, '#1c92fc'), (334, '#04ce04'), (346, '#fc7e04'), (9999, '#cc0204')],
        'barb_color': '#cc8604',
        'smooth_sigma_km': 12.0,
    },
    'Moisture Convergence & Mixing Ratio': {
        'title': 'Moisture Convergence, Mixing Ratio (g/kg) & 10m Wind',
        'plot_type': 'moisture_convergence',
        'mr_levels': [7, 9, 11, 13, 15, 17, 20, 22],
        'mr_colors': ['#FCECE7', 'cyan', 'forestgreen', 'limegreen', 'greenyellow', 'gold', 'lightpink'],
        'mconv_levels': np.arange(5, 35, 5)
    },
    'Surface-Based CAPE & CIN': {
        'title': 'Surface-Based CAPE (J/kg) & CIN (J/kg)',
        'plot_type': 'cape_cin',
        'var_name_cape': 'sb_cape',
        'var_name_cin': 'sb_cinh',
        'cape_levels': [100] + list(np.arange(250, 7250, 250)),
        'cape_colors': 'red',
        'cin_levels': [25, 100, 5000],
        'cin_colors': ['lightblue', 'darkblue'],
        'smoothing': 1.5
    },
    '100mb Mixed-Layer CAPE & CIN': {
        'title': '100mb Mixed-Layer CAPE (J/kg) & CIN (J/kg)',
        'plot_type': 'cape_cin',
        'var_name_cape': 'ml_cape',
        'var_name_cin': 'ml_cinh',
        'cape_levels': [100] + list(np.arange(250, 7250, 250)),
        'cape_colors': 'red',
        'cin_levels': [25, 100, 5000],
        'cin_colors': ['lightblue', 'darkblue'],
        'smoothing': 1.5
    },
    'Downdraft CAPE & DCIN': {
        'title': 'Downdraft CAPE (J/kg) & DCIN (J/kg)',
        'plot_type': 'cape_cin',
        'var_name_cape': 'dcape',
        'var_name_cin': 'dcinh',
        'cape_levels': list(np.arange(200, 4000, 200)),
        'cape_dynamic_colors': True,
        'cin_levels': [25, 100, 5000],
        'cin_colors': ['lightblue', 'darkblue'],
        'smoothing': 1.5
    },
    'Mid-Level Lapse Rate (700-500mb)': {
        'title': '700-500mb Lapse Rate (°C/km)',
        'plot_type': 'custom_contour',
        'var_name': 'lapse_rate_700_500hPa',
        'scale': 1.0,
        'levels': list(np.arange(5.5, 12.0, 0.5)),
        'colors': ['blue', 'green', 'green'] + _lr_grad,
        'linestyles': ['dashed'] + ['solid'] * 12,
        'label_fmt': '%g',
        'smoothing': 2.0
    },
    'Low-Level Lapse Rate (0-3km)': {
        'title': '0-3km Lapse Rate (°C/km)',
        'plot_type': 'custom_contour',
        'var_name': 'lapse_rate_0_3km',
        'scale': 1.0,
        'levels': list(np.arange(5.5, 12.0, 0.5)),
        'colors': ['blue', 'green', 'green'] + _lr_grad,
        'linestyles': ['dashed'] + ['solid'] * 12,
        'label_fmt': '%g',
        'smoothing': 2.0
    },
    '100mb Mixed-Layer LCL Height (m)': {
        'title': '100mb Mixed-Layer LCL Height (m AGL)',
        'plot_type': 'custom_contour',
        'var_name': 'ml_lcl_hght_agl',
        'scale': 1.0,
        'levels': [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000],
        'colors': _lcl_green_grad + ['black'] * 7,
        'linestyles': 'solid',
        'label_fmt': '%d',
        'smoothing': 2.0
    },
    'Effective Bulk Shear (kt)': {
        'title': 'Effective Bulk Shear (kt) & Vectors',
        'plot_type': 'shear',
        'var_u': 'effective_shear_u',
        'var_v': 'effective_shear_v',
        'scale': 1.94384,
        'levels': list(np.arange(20, 105, 5)),
        'colors': ['#004c99'] * 17,
        'linestyles': ['dotted'] + ['solid'] * 16,
        'linewidths': [0.8] + [0.9] * 3 + [1.8] * 13,
        'smoothing': 1.0
    },
    'Surface - 1km Shear (kt)': {
        'title': '0-1km Bulk Shear (kt) & Vectors',
        'plot_type': 'shear',
        'var_u': 'shear_0_1km_u',
        'var_v': 'shear_0_1km_v',
        'scale': 1.94384,
        'levels': list(np.arange(10, 55, 5)),
        'colors': ['#0099ff'] * 2 + ['#004488'] * 7,
        'linestyles': 'solid',
        'linewidths': [0.9] * 2 + [1.8] * 7,
        'smoothing': 2.0
    },
    'Surface - 3km Shear (kt)': {
        'title': '0-3km Bulk Shear (kt) & Vectors',
        'plot_type': 'shear',
        'var_u': 'shear_0_3km_u',
        'var_v': 'shear_0_3km_v',
        'scale': 1.94384,
        'levels': list(np.arange(20, 75, 5)),
        'colors': ['#0099ff'] * 2 + ['#004488'] * 9,
        'linestyles': 'solid',
        'linewidths': [0.9] * 2 + [1.8] * 9,
        'smoothing': 2.0
    },
    'Surface - 6km Shear (kt)': {
        'title': '0-6km Bulk Shear (kt) & Vectors',
        'plot_type': 'shear',
        'var_u': 'shear_0_6km_u',
        'var_v': 'shear_0_6km_v',
        'scale': 1.94384,
        'levels': [25, 30, 40, 50, 60, 70, 80, 90],
        'colors': ['#0099ff'] + ['#0077ee'] * 2 + ['#004488'] * 5,
        'linestyles': ['dashed'] + ['solid'] * 7,
        'linewidths': [0.9] + [1.2] * 2 + [1.8] * 5,
        'smoothing': 2.0
    },
    'Surface - 8km Shear (kt)': {
        'title': '0-8km Bulk Shear (kt) & Vectors',
        'plot_type': 'shear',
        'var_u': 'shear_0_8km_u',
        'var_v': 'shear_0_8km_v',
        'scale': 1.94384,
        'levels': list(np.arange(40, 130, 10)),
        'colors': ['#0088ff'] * 2 + ['#004488'] * 7,
        'linestyles': 'solid',
        'linewidths': [1.0] * 2 + [1.8] * 7,
        'smoothing': 2.0
    },
    'Effective SRH & Inflow Base': {
        'title': 'Eff. Inflow Base (fill, m AGL), ESRH (m²/s²) & Storm Motion (kt)',
        'plot_type': 'esrh_composite',
        'var_srh': 'srh_eff',
        'var_eil_base': 'eil_bottom_hght_agl',
        'var_u': 'supercell_right_mover_u',
        'var_v': 'supercell_right_mover_v',
        'srh_levels': [25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'srh_colors': ['#0088ff', '#0088ff', '#0077ee'] + ['#004488'] * 9,
        'srh_linestyles': ['dotted'] + ['solid'] * 11,
        'srh_linewidths': [0.9, 0.9, 1.1] + [1.9] * 9,
        'eil_levels': [10, 250, 1000, 6000],
        'eil_colors': ['#f9a8a8', '#e84848', '#cc0000'],
        'smoothing': 1.5
    },
    'Sfc-500m SRH & Inflow Base': {
        'title': 'Eff. Inflow Base (fill, m AGL), 0-500m SRH (m²/s²) & Storm Motion (kt)',
        'plot_type': 'esrh_composite',
        'plot_eil': True,
        'var_srh': ['srh_0_500m', 'esrh_0_500m'],
        'var_eil_base': 'eil_bottom_hght_agl',
        'var_u': 'supercell_right_mover_u',
        'var_v': 'supercell_right_mover_v',
        'srh_levels': [25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'srh_colors': ['#0088ff', '#0088ff'] + ['#004488'] * 10,
        'srh_linestyles': ['dotted'] + ['solid'] * 11,
        'srh_linewidths': [0.9, 0.9] + [1.9] * 10,
        'eil_levels': [10, 25, 100, 6000],
        'eil_colors': ['#f9a8a8', '#e84848', '#cc0000'],
        'eil_ticks': [25, 100],
        'eil_ticklabels': ['25', '100'],
        'smoothing': 1.5
    },
    'Sfc-1km SRH & Inflow Base': {
        'title': 'Eff. Inflow Base (fill, m AGL), 0-1km SRH (m²/s²) & Storm Motion (kt)',
        'plot_type': 'esrh_composite',
        'plot_eil': True,
        'var_srh': 'srh_0_1km',
        'var_eil_base': 'eil_bottom_hght_agl',
        'var_u': 'supercell_right_mover_u',
        'var_v': 'supercell_right_mover_v',
        'srh_levels': [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'srh_colors': ['#0088ff', '#0088ff', '#0066dd'] + ['#0055cc'] * 9,
        'srh_linestyles': 'solid',
        'srh_linewidths': [0.9, 0.9, 1.4] + [1.9] * 9,
        'eil_levels': [10, 250, 1000, 6000],
        'eil_colors': ['#f9a8a8', '#e84848', '#cc0000'],
        'smoothing': 1.5
    },
    'Sfc-3km SRH': {
        'title': '0-3km SRH (m²/s²) & Bunkers Storm Motion (kt)',
        'plot_type': 'esrh_composite',
        'plot_eil': False,
        'var_srh': 'srh_0_3km',
        'var_u': 'supercell_right_mover_u',
        'var_v': 'supercell_right_mover_v',
        'srh_levels': list(np.arange(100, 1250, 50)),
        'srh_colors': ['#0088ff', '#0088ff', '#0066dd', '#0066dd'] + ['#004488'] * 19,
        'srh_linestyles': 'solid',
        'srh_linewidths': [0.9, 0.9, 1.4, 1.4] + [1.9] * 19,
        'smoothing': 1.5
    },
    'Surface - 2km SR Winds': {
        'title': '0-2km Storm-Relative Winds (kt)',
        'plot_type': 'shear',
        'var_u': ['srw_0_2km_u', 'sr_wind_0_2km_u', 'storm_rel_wind_0_2km_u', 'srw_0_2_u'],
        'var_v': ['srw_0_2km_v', 'sr_wind_0_2km_v', 'storm_rel_wind_0_2km_v', 'srw_0_2_v'],
        'srw_layer_m': (0, 2000),
        'scale': 1.94384,
        'levels': list(np.arange(15, 105, 5)),
        'colors': ['#0099ff'] + ['#004488'] * 17,
        'linestyles': 'solid',
        'linewidths': [0.9] + [1.7] * 17,
        'smoothing': 2.0
    },
    '4 - 6km SR Winds': {
        'title': '4-6km (Mid-Level) Storm-Relative Winds (kt)',
        'plot_type': 'shear',
        'var_u': ['srw_4_6km_u', 'sr_wind_4_6km_u', 'storm_rel_wind_4_6km_u', 'srw_4_6_u'],
        'var_v': ['srw_4_6km_v', 'sr_wind_4_6km_v', 'storm_rel_wind_4_6km_v', 'srw_4_6_v'],
        'srw_layer_m': (4000, 6000),
        'scale': 1.94384,
        'levels': list(np.arange(15, 105, 5)),
        'colors': ['#0099ff'] + ['#004488'] * 17,
        'linestyles': 'solid',
        'linewidths': [0.9] + [1.7] * 17,
        'smoothing': 2.0
    },
    '9 - 11km SR Winds': {
        'title': '9-11km (Upper-Level) Storm-Relative Winds (kt)',
        'plot_type': 'shear',
        'var_u': ['srw_9_11km_u', 'sr_wind_9_11km_u', 'storm_rel_wind_9_11km_u', 'srw_9_11_u'],
        'var_v': ['srw_9_11km_v', 'sr_wind_9_11km_v', 'storm_rel_wind_9_11km_v', 'srw_9_11_v'],
        'srw_layer_m': (9000, 11000),
        'scale': 1.94384,
        'levels': [15, 20] + list(np.arange(30, 150, 10)),
        'colors': ['#0099ff'] + ['#0077ee'] * 3 + ['#004488'] * 10,
        'linestyles': 'solid',
        'linewidths': [0.9] + [1.3] * 3 + [1.7] * 10,
        'smoothing': 2.0
    },
    'Supercell Composite (Right-Moving)': {
        'title': 'Supercell Composite Parameter (Eff. Layer w/ muCIN) & Bunkers Storm Motion (kt)',
        'plot_type': 'composite_index',
        'var_index': ['scp_eff', 'scp', 'supercell_composite_eff', 'supercell_composite'],
        'fallback_type': 'scp_right',
        'levels': [0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50],
        'colors': ['#0099ff', '#0088ff'] + ['#0066dd'] * 3 + ['#004488'] * 13,
        'linestyles': ['dashed'] + ['solid'] * 17,
        'linewidths': [0.9, 1.0] + [1.4] * 3 + [1.9] * 13,
        'cin_var': ['mu_cinh', 'mucin'],
        'cin_levels': [40, 80, 120, 160, 5000],
        'cin_colors': ['#f9c6c6', '#f08080', '#d63030', '#800000'],
        'cin_ticks': [40, 80, 120, 160],
        'cin_ticklabels': ['40', '80', '120', '160'],
        'cin_label': 'MUCIN (J/kg)',
        'wind_u': 'supercell_right_mover_u',
        'wind_v': 'supercell_right_mover_v',
        'smoothing': 1.5
    },
    'Supercell Composite (Left-Moving)': {
        'title': 'Left-Moving Supercell Composite (Eff. Layer w/ muCIN) & Bunkers Left Motion (kt)',
        'plot_type': 'composite_index',
        'var_index': ['scp_eff_left', 'scp_left', 'supercell_composite_left', 'left_scp'],
        'fallback_type': 'scp_left',
        'force_negative': True,
        'levels': [-50, -45, -40, -35, -30, -25, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1, -0.5],
        'colors': ['#004488'] * 13 + ['#0066dd'] * 3 + ['#0088ff', '#0099ff'],
        'linestyles': ['solid'] * 17 + ['dashed'],
        'linewidths': [1.9] * 13 + [1.4] * 3 + [1.0, 0.9],
        'cin_var': ['mu_cinh', 'mucin'],
        'cin_levels': [40, 80, 120, 160, 5000],
        'cin_colors': ['#f9c6c6', '#f08080', '#d63030', '#800000'],
        'cin_ticks': [40, 80, 120, 160],
        'cin_ticklabels': ['40', '80', '120', '160'],
        'cin_label': 'MUCIN (J/kg)',
        'wind_u': 'supercell_left_mover_u',
        'wind_v': 'supercell_left_mover_v',
        'smoothing': 1.5
    },
    'Significant Tornado Parameter (Fixed Layer)': {
        'title': 'Significant Tornado Parameter (Fixed Layer) & SBCIN (J/kg)',
        'plot_type': 'composite_index',
        'var_index': ['stp_fixed', 'stp_fix', 'sig_tor_fixed', 'stp'],
        'fallback_type': 'stp_fixed',
        'levels': [0.5] + list(range(1, 17)) + list(range(18, 32, 2)),
        'colors': ['#ff8c00', '#ff8c00'] + ['#e60000'] * 3 + ['#990033'] * 19,
        'linestyles': ['dashed'] + ['solid'] * 23,
        'linewidths': [1.1, 1.4] + [1.7] * 3 + [2.0] * 19,
        'cin_var': ['sb_cinh', 'sbcin'],
        'cin_levels': [25, 100, 5000],
        'cin_colors': ['#00ffff', '#00aaff'],
        'cin_ticks': [25, 100],
        'cin_ticklabels': ['25', '100'],
        'cin_label': 'SBCIN (J/kg)',
        'smoothing': 1.5
    },
    'Significant Tornado Parameter (Effective Layer)': {
        'title': 'Significant Tornado Parameter (Effective Layer) & MLCIN (J/kg)',
        'plot_type': 'composite_index',
        'var_index': ['stp_eff', 'sig_tor_eff', 'stp_effective'],
        'fallback_type': 'stp_eff',
        'levels': [0.5] + list(range(1, 17)) + list(range(18, 32, 2)),
        'colors': ['#ff8c00', '#ff8c00'] + ['#e60000'] * 3 + ['#990033'] * 19,
        'linestyles': ['dashed'] + ['solid'] * 23,
        'linewidths': [1.1, 1.4] + [1.7] * 3 + [2.0] * 19,
        'cin_var': ['ml_cinh', 'mlcin'],
        'cin_levels': [25, 100, 5000],
        'cin_colors': ['#00ffff', '#00aaff'],
        'cin_ticks': [25, 100],
        'cin_ticklabels': ['25', '100'],
        'cin_label': 'MLCIN (J/kg)',
        'smoothing': 1.5
    },
    'Significant Tornado Parameter (0-500m SRH)': {
        'title': 'Significant Tornado Parameter (0-500m SRH, Eff. Layer) & MLCIN (J/kg)',
        'plot_type': 'composite_index',
        'var_index': ['stp_0_500m', 'stp_500m', 'stp_eff_500m', 'stp_0_500'],
        'fallback_type': 'stp_500m',
        'levels': [0.25, 0.5] + list(range(1, 13)),
        'colors': ['#ff8c00'] * 3 + ['#e60000'] * 3 + ['#990033'] * 8,
        'linestyles': ['dashed', 'dashed'] + ['solid'] * 12,
        'linewidths': [1.1, 1.1, 1.4] + [1.7] * 3 + [2.0] * 8,
        'cin_var': ['ml_cinh', 'mlcin'],
        'cin_levels': [25, 100, 5000],
        'cin_colors': ['#00ffff', '#00aaff'],
        'cin_ticks': [25, 100],
        'cin_ticklabels': ['25', '100'],
        'cin_label': 'MLCIN (J/kg)',
        'smoothing': 1.5
    },
    'Conditional SigTor Prob. (Eqn. 1)': {
        'title': 'Conditional Probability of Significant Tornado (Eqn 1 - mlcape;shr6;srh1;mlcin) (%)',
        'plot_type': 'composite_index',
        'var_index': ['cond_prob_sigtor_1', 'cprob_sigtor_1', 'sigtor_prob_1', 'prob_sigtor_1'],
        'fallback_type': 'cprob_1',
        'is_percent': True,
        'levels': [10, 20, 30, 40, 50, 60, 70, 80, 90],
        'colors': ['black'] * 9,
        'linestyles': 'solid',
        'linewidths': [0.9, 1.3, 1.9, 1.9, 1.9, 1.0, 1.0, 1.0, 1.0],
        'smoothing': 2.0
    },
    'Conditional SigTor Prob. (Eqn. 2)': {
        'title': 'Conditional Probability of Significant Tornado (Eqn 2 - mlcape;shr6;srh1;mlcin) (%)',
        'plot_type': 'composite_index',
        'var_index': ['cond_prob_sigtor_2', 'cprob_sigtor_2', 'sigtor_prob_2', 'prob_sigtor_2'],
        'fallback_type': 'cprob_2',
        'is_percent': True,
        'levels': [10, 20, 30, 40, 50, 60, 70, 80, 90],
        'colors': ['black'] * 9,
        'linestyles': 'solid',
        'linewidths': [0.9, 1.3, 1.9, 1.9, 1.9, 1.0, 1.0, 1.0, 1.0],
        'smoothing': 2.0
    },
    'Non-Supercell Tornado Parameter': {
        'title': 'Non-Supercell Tornado Parameter',
        'plot_type': 'composite_index',
        'var_index': ['nstp', 'nst', 'nonsupercell_tor', 'non_supercell_tor'],
        'fallback_type': 'nstp',
        'levels': [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10],
        'colors': ['#ff8c00'] * 3 + ['#e60000'] * 7,
        'linestyles': 'solid',
        'linewidths': [1.2] * 3 + [1.6] * 7,
        'smoothing': 1.5
    },
    'Violent Tornado Parameter': {
        'title': 'Violent Tornado Parameter & MLCIN (J/kg)',
        'plot_type': 'composite_index',
        'var_index': ['vtp', 'violent_tor', 'violent_tornado'],
        'fallback_type': 'vtp',
        'levels': [0.25, 0.5, 1, 2, 3] + list(range(4, 16)),
        'colors': ['#ff8c00'] * 3 + ['#e60000'] * 3 + ['#990033'] * 11,
        'linestyles': ['dashed', 'dashed'] + ['solid'] * 15,
        'linewidths': [1.1, 1.1, 1.4] + [1.7] * 3 + [2.0] * 11,
        'cin_var': ['ml_cinh', 'mlcin'],
        'cin_levels': [25, 100, 5000],
        'cin_colors': ['#00ffff', '#00aaff'],
        'cin_ticks': [25, 100],
        'cin_ticklabels': ['25', '100'],
        'cin_label': 'MLCIN (J/kg)',
        'smoothing': 1.5
    },
    'Significant Hail Parameter': {
        'title': 'Significant Hail Parameter (SHIP)',
        'plot_type': 'composite_index',
        'var_index': ['ship', 'sig_hail', 'significant_hail'],
        'fallback_type': 'ship',
        'levels': [0.5, 1, 1.5] + list(range(2, 11)),
        'colors': ['#8b4513', '#e68a00', '#e69900'] + _ship_grad,
        'linestyles': ['dashed'] + ['solid'] * 11,
        'linewidths': [0.9, 1.1, 1.2] + [1.6] * 9,
        'smoothing': 1.5
    },
    'SARS Hail Size (in)': {
        'title': 'SARS Hail Size (in)',
        'plot_type': 'composite_index',
        'var_index': ['sars_hail_size', 'sars_size', 'hail_size_sars'],
        'fallback_type': 'sars_size',
        'no_contours': True,
        'fill_self': True,
        'cin_levels': [0.75, 1.00, 1.75, 2.00, 2.50, 2.75, 3.00, 4.00, 10.0],
        'cin_colors': ['#ffffff00', '#ffff00', '#ff0000', '#0000e6', '#1e90ff', '#00bfff', '#00ffff', '#fce4ec'],
        'cin_alpha': 0.85,
        'cin_ticks': [0.75, 1.00, 1.75, 2.00, 2.50, 2.75, 3.00, 4.00],
        'cin_ticklabels': ['0.75', '1.00', '1.75', '2.00', '2.50', '2.75', '3.00', '4.00'],
        'cin_label': 'SARS Hail Size (in)',
        'smoothing': 1.0
    },
    'SARS Sig. Hail Percentage': {
        'title': 'SARS Base Hail % (Contours) & Sig. Hail % (Fill)',
        'plot_type': 'composite_index',
        'var_index': ['sars_hail_prob', 'sars_prob_hail', 'sars_base_hail'],
        'fallback_type': 'sars_base_pct',
        'is_percent': True,
        'levels': [5, 20, 55, 65, 75, 85, 95, 100],
        'colors': ['#800000', '#fa8072'] + _sars_base_grad,
        'linestyles': 'solid',
        'linewidths': [1.2, 1.3] + [1.7] * 6,
        'cin_var': ['sars_sig_hail_prob', 'sars_prob_sig_hail', 'sars_sig_hail'],
        'fallback_cin': 'sars_sig_pct',
        'cin_is_percent': True,
        'cin_levels': [50, 60, 70, 80, 90, 105],
        'cin_colors': ['#008000', '#66ff00', '#ccff00', '#ffcc00', '#cc0000'],
        'cin_alpha': 0.85,
        'cin_ticks': [50, 60, 70, 80, 90, 100],
        'cin_ticklabels': ['50', '60', '70', '80', '90', '100'],
        'cin_label': 'SARS Sig. Hail Probability (%)',
        'smoothing': 1.5
    },
    'Large Hail Parameter': {
        'title': 'Large Hail Parameter (LHP)',
        'plot_type': 'composite_index',
        'var_index': ['lhp', 'large_hail', 'large_hail_param'],
        'fallback_type': 'lhp',
        'levels': list(range(2, 28, 2)),
        'colors': ['#b8860b', '#cc7a00', '#e69900', '#ffcc00', '#ffe600'] + _lhp_grad,
        'linestyles': ['dashed'] + ['solid'] * 12,
        'linewidths': [0.9, 1.1, 1.2, 1.3, 1.4] + [1.7] * 8,
        'smoothing': 1.5
    },
    'Derecho Composite': {
        'title': 'Derecho Composite Parameter (DCP)',
        'plot_type': 'composite_index',
        'var_index': ['dcp', 'derecho_composite', 'derecho'],
        'fallback_type': 'dcp',
        'levels': list(range(1, 21)),
        'colors': ['#ff8c00'] + ['#e63900'] * 19,
        'linestyles': 'solid',
        'linewidths': [1.0] + [1.8] * 19,
        'smoothing': 1.5
    },
    'Craven Brooks Significant Severe': {
        'title': 'Craven-Brooks Significant Severe (x1000 m³/s³)',
        'plot_type': 'composite_index',
        'var_index': ['craven_sigsig', 'cbsigsvr', 'craven_brooks', 'sigsig'],
        'fallback_type': 'craven_sigsig',
        'scale_thousands': True,
        'levels': [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
        'colors': ['#0033ff'] * 3 + ['#ff00ff'] * 2 + ['#ff0000'] * 8,
        'linestyles': 'solid',
        'linewidths': [0.9] * 3 + [1.1] * 2 + [1.8] * 8,
        'fill_self': True,
        'cin_levels': [30, 1000],
        'cin_colors': ['#d2b48c'],
        'cin_alpha': 0.32,
        'cin_ticks': [30],
        'cin_ticklabels': ['> 30'],
        'cin_label': 'Craven-Brooks SigSvr (> 30k m³/s³)',
        'smoothing': 1.5
    },
    'Bulk Richardson Number': {
        'title': 'Bulk Richardson Number (BRN, based on MLCAPE)',
        'plot_type': 'composite_index',
        'var_index': ['brn', 'bulk_rich', 'bulk_richardson'],
        'fallback_type': 'brn',
        'levels': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140],
        'colors': ['#b8860b', '#ff0000'] + ['#ff00ff'] * 5 + ['#ff0000'] * 3 + ['#b8860b'] * 3,
        'linestyles': ['dashed'] + ['solid'] * 9 + ['dashed'] * 3,
        'linewidths': [1.1, 1.4] + [1.2] * 5 + [1.4] * 3 + [1.1] * 3,
        'fill_self': True,
        'cin_levels': [10, 50],
        'cin_colors': ['#ff99cc'],
        'cin_alpha': 0.45,
        'cin_ticks': [10, 50],
        'cin_ticklabels': ['10', '50'],
        'cin_label': 'BRN Supercell Regime (10 - 50)',
        'smoothing': 1.5
    },
    'MCS Maintenance': {
        'title': 'MCS Maintenance Parameter',
        'plot_type': 'composite_index',
        'var_index': ['mcsm', 'mcs_maint', 'mcs_maintenance'],
        'fallback_type': 'mcsm',
        'levels': list(range(10, 110, 10)),
        'colors': ['#cc8400'] * 4 + ['#ff0000'] * 6,
        'linestyles': 'solid',
        'linewidths': [1.0] * 4 + [1.7] * 6,
        'smoothing': 2.0
    },
    'Microburst Composite': {
        'title': 'Microburst Composite Parameter',
        'plot_type': 'composite_index',
        'var_index': ['mbcp', 'microburst', 'microburst_composite'],
        'fallback_type': 'mbcp',
        'levels': list(range(1, 11)),
        'colors': ['#8b5a2b'] + ['#e68a00'] * 3 + ['#e60000'] * 6,
        'linestyles': ['dashed'] + ['solid'] * 9,
        'linewidths': [1.0] + [1.1] * 3 + [1.3] * 3 + [2.0] * 3,
        'smoothing': 1.5
    },
    'Enhanced Stretching Potential': {
        'title': 'Enhanced Stretching Potential (ESP)',
        'plot_type': 'composite_index',
        'var_index': ['esp', 'enhanced_stretching'],
        'fallback_type': 'esp',
        'levels': [0.5, 1, 1.5] + list(range(2, 16)),
        'colors': ['#ff8c00', '#ff8c00', '#ff6600'] + ['#e60000'] * 14,
        'linestyles': ['dashed'] + ['solid'] * 16,
        'linewidths': [1.0, 1.1, 1.2] + [1.8] * 14,
        'smoothing': 1.5
    },
    'Sfc - 1km EHI': {
        'title': '0-1km Energy Helicity Index (EHI)',
        'plot_type': 'composite_index',
        'var_index': ['ehi_0_1km', 'ehi1', 'ehi_1km'],
        'fallback_type': 'ehi1',
        'levels': [0.5] + list(range(1, 11)),
        'colors': ['#008000'] + _ehi_grad,
        'linestyles': ['dashed'] + ['solid'] * 10,
        'linewidths': [1.1] + [1.5] * 10,
        'smoothing': 1.5
    },
    'Sfc - 3km EHI': {
        'title': '0-3km Energy Helicity Index (EHI)',
        'plot_type': 'composite_index',
        'var_index': ['ehi_0_3km', 'ehi3', 'ehi_3km'],
        'fallback_type': 'ehi3',
        'levels': [0.5] + list(range(1, 11)),
        'colors': ['#008000'] + _ehi_grad,
        'linestyles': ['dashed'] + ['solid'] * 10,
        'linewidths': [1.1] + [1.5] * 10,
        'smoothing': 1.5
    },
    'Sfc - 3km VGP': {
        'title': '0-3km Vorticity Generation Parameter (VGP) & 0-3km Shear Vectors (kt)',
        'plot_type': 'composite_index',
        'var_index': ['vgp_0_3km', 'vgp3', 'vgp', 'vgp_3km'],
        'fallback_type': 'vgp3',
        'levels': [round(x, 1) for x in np.arange(0.1, 1.1, 0.1)],
        'colors': ['#0077ee'] * 10,
        'linestyles': 'solid',
        'linewidths': [1.4] * 10,
        'wind_u': 'shear_0_3km_u',
        'wind_v': 'shear_0_3km_v',
        'smoothing': 2.0
    },
    'Sfc Vorticity / Sfc - 3km MLCAPE': {
        'title': '0-3km MLCAPE (J/kg, Red), Surface Vorticity (1e-5/s, Blue) & 10m Wind (kt)',
        'plot_type': 'vort_mlcape3',
        'vort_levels': list(range(1, 26)),
        'cape3_levels': list(np.arange(25, 310, 15)),
        'cape3_colors': ['#ff6666'] + ['#e60000'] * 18,
        'cape3_linewidths': [1.0] + [1.6] * 18,
        'vort_smoothing': 4.0,
        'cape_smoothing': 1.5
    },
    'Precipitable Water (in)': {
        'title': 'Precipitable Water (in)',
        'plot_type': 'moisture_field',
        'var_field': ['pwat', 'pwat_400mb', 'precip_water', 'pw'],
        'fallback_type': 'pwat',
        'levels': list(np.round(np.arange(0.1, 3.1, 0.1), 1)),
        'contour_color': 'black',
        'contour_width': 0.95,
        'fill_levels': [1.0, 1.5, 2.0, 5.0],
        'fill_colors': ['#7cfc00', '#00cc00', '#008000'],
        'fill_alpha': 0.55,
        'fill_ticks': [1.0, 1.5, 2.0],
        'fill_ticklabels': ['1.0', '1.5', '2.0'],
        'fill_label': 'Precipitable Water (in)',
        'smoothing': 1.5
    },
    'Precipitable Water & 850mb Moisture Transport': {
        'title': 'Precipitable Water (in) & 850mb Moisture Transport Vectors',
        'plot_type': 'moisture_field',
        'var_field': ['pwat', 'pwat_400mb', 'precip_water', 'pw'],
        'fallback_type': 'pwat',
        'levels': list(np.round(np.arange(0.1, 3.1, 0.1), 1)),
        'contour_color': 'black',
        'contour_width': 0.95,
        'fill_levels': [1.0, 1.5, 2.0, 5.0],
        'fill_colors': ['#7cfc00', '#00cc00', '#008000'],
        'fill_alpha': 0.45,
        'fill_ticks': [1.0, 1.5, 2.0],
        'fill_ticklabels': ['1.0', '1.5', '2.0'],
        'fill_label': 'Precipitable Water (in)',
        'vector_mode': 'quiver_850_mtrans',
        'smoothing': 1.5
    },
    '100mb Mean Mixing Ratio': {
        'title': '100mb Mean Mixing Ratio (g/kg) & Surface Streamlines',
        'plot_type': 'moisture_field',
        'var_field': ['mixr_100mb', 'ml_mixr', 'mean_mixr_100mb', 'mixr_mean_100mb'],
        'fallback_type': 'ml_mixr',
        'levels': [6, 8, 10, 12, 14, 16, 18, 20],
        'contour_color': 'black',
        'contour_width': 1.35,
        'fill_levels': [10, 12, 14, 16, 18, 30],
        'fill_colors': ['#7cfc00', '#00cc00', '#008000', '#ffff00', '#ffcc00'],
        'fill_alpha': 0.65,
        'fill_ticks': [10, 12, 14, 16, 18],
        'fill_ticklabels': ['10', '12', '14', '16', '18'],
        'fill_label': '100mb Mean Mixing Ratio (g/kg)',
        'vector_mode': 'streamlines_sfc',
        'smoothing': 1.5
    },
    'Precipitation Type': {
        'title': 'Precipitation Type (0=RA 1=ZL-/ZR 2=ZR 3=ZR/IP 4=IP- 5=IP 6=IP/SN 7=SN- 8=SN 9=SN+ 10=Uk)',
        'plot_type': 'winter_ptype',
    },
    'Lake Effect Snow 1': {
        'title': 'Boundary-Layer RH (fill, %), Sfc-850mb Temp Diff (°C, Orange) & SBCAPE (J/kg, Red)',
        'plot_type': 'les1',
        'rh_levels': [70, 75, 80, 85, 90, 95],
        'rh_fill_levels': [70, 80, 90, 105],
        'rh_fill_colors': ['#7cfc00', '#00cc00', '#008000'],
        'dt_levels': [10, 12, 14, 16, 18, 20, 22, 24],
        'cape_levels': list(range(50, 1050, 50)),
        'smoothing': 1.5
    },
    'Lake Effect Snow 2': {
        'title': 'Boundary-Layer Wind (kt), Streamlines (Black) & Convergence (1e-5/s, Red)',
        'plot_type': 'les2',
        'conv_levels': list(range(-20, 0, 2)),
        'smoothing': 2.5
    },
    'Snow Squall Parameter': {
        'title': 'Snow Squall Parameter (SNSQ)',
        'plot_type': 'composite_index',
        'var_index': ['snsq', 'snow_squall', 'snow_squall_param'],
        'fallback_type': 'snsq',
        'no_contours': True,
        'fill_self': True,
        'show_max_h': True,
        'cin_levels': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 20.0],
        'cin_colors': [
            '#084594', '#0077ee', '#00aaff', '#00eeee',
            '#7b68ee', '#9932cc', '#8b008b', '#ff00ff', '#f4a6c0'
        ],
        'cin_alpha': 0.90,
        'cin_ticks': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        'cin_ticklabels': ['0.1', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '4.0', '5.0'],
        'cin_label': 'Snow Squall Parameter',
        'smoothing': 1.5
    },
    'Dendritic Growth Layer Depth & RH': {
        'title': 'Dendritic Layer (-12 to -17°C) Depth (m, Purple) & Mean RH (%, Green/Orange)',
        'plot_type': 'dgz_composite',
        'depth_levels': [600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000],
        'depth_colors': ['#8a2be2'] * 5 + ['#6a0dad'] * 5,
        'depth_linewidths': [0.9, 1.2, 1.2, 1.2, 1.2, 2.1, 2.1, 2.1, 2.1, 2.1],
        'rh_levels': [50, 60, 70, 80, 90],
        'rh_colors': ['#cc7a00', '#cc7a00', '#228b22', '#009900', '#008000'],
        'rh_linewidths': [1.0, 1.0, 1.2, 1.6, 1.8],
        'smoothing': 1.5
    }
}

# =====================================================================================
# SPC STYLE MATCHING (batch 1). Levels and colors read from SPC's own mesoanalysis images
# (archive, 2024-05-21 20Z). Applied on top of PARAMETERS so the originals stay visible above.
# =====================================================================================
SPC_BARB = '#cc8604'   # SPC draws most wind/shear/storm-motion vectors in this goldenrod


def _by_level(levels, breaks):
    """Color for each level from [(upper_bound, color), ...] (first bound the level fits under)."""
    out = []
    for v in levels:
        for upper, col in breaks:
            if v <= upper:
                out.append(col)
                break
    return out


def _spc_set(name, **kw):
    if name in PARAMETERS:
        PARAMETERS[name].update(kw)


# --- CAPE / CIN ---
_cape_lv = [100, 250, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
for _n in ('Surface-Based CAPE & CIN', '100mb Mixed-Layer CAPE & CIN'):
    _spc_set(_n,
             cape_levels=_cape_lv,
             cape_colors=_by_level(_cape_lv, [(250, '#fc6a6c'), (3000, '#ec2e2c'), (5000, '#cc0204'), (1e9, '#8c0204')]),
             cape_linewidths=[0.9 if v < 1000 else 1.7 for v in _cape_lv],
             cin_levels=[25, 100, 5000], cin_colors=['#04eeec', '#04b2ec'], cin_hatch=True)

_dcape_lv = list(range(200, 3100, 100))
_spc_set('Downdraft CAPE & DCIN',
         cape_levels=_dcape_lv,
         cape_colors=_by_level(_dcape_lv, [(700, '#cc8604'), (900, '#ec4204'), (1e9, '#8c0204')]),
         cape_linewidths=[0.9 if v <= 700 else (1.3 if v <= 900 else 1.7) for v in _dcape_lv],
         cape_dynamic_colors=False,
         cin_levels=[25, 100, 5000], cin_colors=['#04eeec', '#04b2ec'], cin_hatch=True)

# --- Lapse rates: blue dashed <=5.5, green 6-6.5, orange 7-7.5, red >=8 with tan hatching ---
_lr_lv = [float(v) for v in np.arange(5.0, 12.5, 0.5)]
for _n, _lw in (('Mid-Level Lapse Rate (700-500mb)', 1.5), ('Low-Level Lapse Rate (0-3km)', 1.0)):
    _spc_set(_n,
             levels=_lr_lv,
             colors=_by_level(_lr_lv, [(5.5, '#04b2ec'), (6.5, '#048a04'), (7.5, '#fc7e04'), (1e9, '#ec2e2c')]),
             linestyles=['dashed' if v <= 5.5 else 'solid' for v in _lr_lv],
             linewidths=_lw,
             hatch_levels=[8.0, 99.0], hatch_colors=['#fca64c'])

# --- LCL: green <=1250 m, black above; hatched 2000/3000/4000 m ---
_lcl_lv = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000]
_spc_set('100mb Mixed-Layer LCL Height (m)',
         levels=_lcl_lv,
         colors=_by_level(_lcl_lv, [(750, '#048a04'), (1250, '#04ce04'), (1e9, 'black')]),
         linestyles='solid',
         linewidths=[1.7 if v <= 1250 else 0.8 for v in _lcl_lv],
         hatch_levels=[2000, 3000, 4000, 99999], hatch_colors=['#fcd604', '#cc8604', '#8c4624'],
         hatch_legend='LCL Height (m AGL)')

# --- Bulk shear (kt): light blue -> blue -> navy, goldenrod shear vectors ---
def _shear(name, levels, breaks):
    _spc_set(name, levels=levels, colors=_by_level(levels, breaks), linestyles='solid',
             linewidths=[0.9 if c != '#144e8c' else 1.8 for c in _by_level(levels, breaks)], barb_color=SPC_BARB)

_shear('Effective Bulk Shear (kt)', list(range(20, 110, 10)), [(30, '#04b2ec'), (40, '#1c92fc'), (1e9, '#144e8c')])
_shear('Surface - 1km Shear (kt)', list(range(10, 90, 10)), [(10, '#1c92fc'), (1e9, '#144e8c')])
_shear('Surface - 3km Shear (kt)', list(range(20, 110, 10)), [(30, '#04b2ec'), (40, '#1c92fc'), (1e9, '#144e8c')])
_shear('Surface - 6km Shear (kt)', list(range(30, 130, 10)), [(30, '#04b2ec'), (40, '#1c92fc'), (1e9, '#144e8c')])
_shear('Surface - 8km Shear (kt)', list(range(30, 130, 10)), [(30, '#04b2ec'), (40, '#1c92fc'), (1e9, '#144e8c')])

# --- Storm-relative winds (kt) ---
_shear('Surface - 2km SR Winds', list(range(15, 105, 5)), [(15, '#1c92fc'), (1e9, '#144e8c')])
_shear('4 - 6km SR Winds', list(range(15, 105, 5)), [(15, '#1c92fc'), (1e9, '#144e8c')])
_shear('9 - 11km SR Winds', [15] + list(range(20, 160, 10)), [(15, '#04b2ec'), (40, '#1c92fc'), (1e9, '#144e8c')])

# --- SRH ---
_esrh_lv = [50, 100] + list(range(200, 1100, 100))
for _n in ('Effective SRH & Inflow Base', 'Sfc-500m SRH & Inflow Base'):
    _spc_set(_n,
             srh_levels=_esrh_lv, srh_colors=['#144e8c'] * len(_esrh_lv), srh_linestyles='solid',
             srh_linewidths=[1.0] + [1.7] * (len(_esrh_lv) - 1),
             eil_levels=[10, 250, 1000, 6000], eil_colors=['#fcaebc', '#fc6a6c', '#ec2e2c'],
             eil_ticks=[10, 250, 1000], eil_ticklabels=['10', '250', '1000'],
             eil_hatch=True, barb_color='black')
_srh1_lv = [50, 100, 150, 200] + list(range(300, 1100, 100))
_spc_set('Sfc-1km SRH & Inflow Base',
         plot_eil=False,
         srh_levels=_srh1_lv,
         srh_colors=_by_level(_srh1_lv, [(100, '#04b2ec'), (200, '#1c92fc'), (1e9, '#144e8c')]),
         srh_linestyles='solid',
         srh_linewidths=[0.9 if v <= 100 else (1.2 if v <= 200 else 1.8) for v in _srh1_lv],
         barb_color=SPC_BARB)
_srh3_lv = list(range(100, 1300, 100))
_spc_set('Sfc-3km SRH',
         srh_levels=_srh3_lv,
         srh_colors=_by_level(_srh3_lv, [(100, '#04b2ec'), (200, '#1c92fc'), (1e9, '#144e8c')]),
         srh_linestyles='solid',
         srh_linewidths=[0.9 if v <= 100 else (1.2 if v <= 200 else 1.8) for v in _srh3_lv],
         barb_color=SPC_BARB)

# --- Supercell Composite: blue family; muCIN hatched in reds ---
_scp_lv = [1, 2, 4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50]
_scp_cols = _by_level(_scp_lv, [(1, '#049acc'), (2, '#04b2ec'), (4, '#1c92fc'), (1e9, '#144e8c')])
_scp_lw = [0.8 if v <= 2 else (1.2 if v <= 4 else 1.9) for v in _scp_lv]
_scp_cin = dict(cin_levels=[40, 80, 120, 160, 5000], cin_colors=['#fcaebc', '#fc6a6c', '#ec2e2c', '#8c0204'],
                cin_hatch=True, barb_color=SPC_BARB)
_spc_set('Supercell Composite (Right-Moving)', levels=_scp_lv, colors=_scp_cols, linestyles='solid',
         linewidths=_scp_lw, **_scp_cin)
_spc_set('Supercell Composite (Left-Moving)', levels=[-v for v in reversed(_scp_lv)],
         colors=list(reversed(_scp_cols)), linestyles='solid', linewidths=list(reversed(_scp_lw)), **_scp_cin)

# --- STP (fixed, effective, 0-500m): orange dashed <1, orange 1, red 2-4, dark red 5-9, maroon 10+ ---
_stp_lv = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
for _n in ('Significant Tornado Parameter (Fixed Layer)', 'Significant Tornado Parameter (Effective Layer)',
           'Significant Tornado Parameter (0-500m SRH)'):
    _spc_set(_n, levels=_stp_lv,
             colors=_by_level(_stp_lv, [(1, '#fc7e04'), (4, '#fc0204'), (9, '#cc0204'), (1e9, '#8c0204')]),
             linestyles=['dashed' if v < 1 else 'solid' for v in _stp_lv],
             linewidths=[1.0 if v < 1 else (1.4 if v == 1 else 1.9) for v in _stp_lv],
             cin_levels=[25, 100, 5000], cin_colors=['#04eeec', '#04b2ec'], cin_alpha=0.85, barb_color=SPC_BARB)


# --- SPC STYLE MATCHING (batch 2): remaining composite indices ---
# NSTP: orange dashed 0.5, orange 1, red 2+
_nstp_lv = [0.5, 1, 2, 3, 4, 5, 6, 8, 10]
_spc_set('Non-Supercell Tornado Parameter', levels=_nstp_lv,
         colors=_by_level(_nstp_lv, [(1, '#fc7e04'), (1e9, '#fc0204')]),
         linestyles=['dashed' if v < 1 else 'solid' for v in _nstp_lv],
         linewidths=[1.0 if v < 1 else (1.3 if v == 1 else 1.8) for v in _nstp_lv])

# VTP: STP-style colors; CIN hatched like SPC's VTP map
_vtp_lv = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
_spc_set('Violent Tornado Parameter', levels=_vtp_lv,
         colors=_by_level(_vtp_lv, [(1, '#fc7e04'), (4, '#fc0204'), (9, '#cc0204'), (1e9, '#8c0204')]),
         linestyles=['dashed' if v < 1 else 'solid' for v in _vtp_lv],
         linewidths=[1.0 if v < 1 else (1.4 if v == 1 else 1.9) for v in _vtp_lv],
         cin_levels=[25, 100, 5000], cin_colors=['#04eeec', '#04b2ec'], cin_hatch=True)

# SHIP: brown dashed 0.5, goldenrod 1-1.5, yellow 2, red 3-4, magenta 5+
_ship_lv = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]
_spc_set('Significant Hail Parameter', levels=_ship_lv,
         colors=_by_level(_ship_lv, [(0.5, '#8c4624'), (1.5, '#cc8604'), (2, '#fcd604'), (4, '#fc0204'), (1e9, '#fc02fc')]),
         linestyles=['dashed' if v < 1 else 'solid' for v in _ship_lv],
         linewidths=[0.9 if v < 2 else 1.7 for v in _ship_lv])

# LHP: goldenrod dashed 4, yellow 8-12, red 16-20, magenta 24+
_lhp_lv = list(range(4, 44, 4))
_spc_set('Large Hail Parameter', levels=_lhp_lv,
         colors=_by_level(_lhp_lv, [(4, '#cc8604'), (12, '#fcd604'), (20, '#fc0204'), (1e9, '#fc02fc')]),
         linestyles=['dashed' if v == 4 else 'solid' for v in _lhp_lv],
         linewidths=[0.9 if v <= 4 else (1.3 if v <= 12 else 1.8) for v in _lhp_lv])

# DCP: thin orange 1, orange-red 2-4, red 6-10, maroon 12+
_dcp_lv = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
_spc_set('Derecho Composite', levels=_dcp_lv,
         colors=_by_level(_dcp_lv, [(1, '#fc7e04'), (4, '#ec4204'), (10, '#cc0204'), (1e9, '#8c0204')]),
         linestyles='solid', linewidths=[0.9 if v == 1 else 1.6 for v in _dcp_lv])

# Craven-Brooks: blue 5-15, magenta 20-25, red 30+ (SPC shows no shading)
_cb = PARAMETERS.get('Craven Brooks Significant Severe', {}).get('levels', [])
_spc_set('Craven Brooks Significant Severe',
         colors=_by_level(_cb, [(15, '#0402fc'), (25, '#fc02fc'), (1e9, '#fc0204')]), fill_self=False)

# BRN: same layout as before, SPC's exact colors and pink shading
_spc_set('Bulk Richardson Number',
         colors=[{'#b8860b': '#cc8604', '#ff0000': '#fc0204', '#ff00ff': '#fc02fc'}.get(c, c)
                 for c in PARAMETERS.get('Bulk Richardson Number', {}).get('colors', [])],
         cin_colors=['#fcaebc'])

# MCS maintenance: goldenrod 10-40, red 50+
_mcs = PARAMETERS.get('MCS Maintenance', {}).get('levels', [])
_spc_set('MCS Maintenance', colors=_by_level(_mcs, [(40, '#cc8604'), (1e9, '#fc0204')]))

# Microburst composite: brown dashed 1, orange 2-4, red 5+
_mb = PARAMETERS.get('Microburst Composite', {}).get('levels', [])
_spc_set('Microburst Composite', colors=_by_level(_mb, [(1, '#8c4624'), (4, '#fc7e04'), (1e9, '#fc0204')]))

# ESP: orange (0.5 dashed) up to 1.5, red 2-4, magenta 5+
_esp = PARAMETERS.get('Enhanced Stretching Potential', {}).get('levels', [])
_spc_set('Enhanced Stretching Potential', colors=_by_level(_esp, [(1.5, '#fc7e04'), (4, '#fc0204'), (1e9, '#fc02fc')]))

# EHI (0-1 and 0-3 km): green dashed 0.5, dark green 1, bright green 2, goldenrod 3+
_ehi_lv = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for _n in ('Sfc - 1km EHI', 'Sfc - 3km EHI'):
    _spc_set(_n, levels=_ehi_lv,
             colors=_by_level(_ehi_lv, [(0.5, '#04ce04'), (1, '#048a04'), (2, '#04ce04'), (1e9, '#cc8604')]),
             linestyles=['dashed' if v < 1 else 'solid' for v in _ehi_lv],
             linewidths=[0.9 if v <= 1 else 1.8 for v in _ehi_lv])

# VGP: blue contours every 0.1, goldenrod shear vectors
_spc_set('Sfc - 3km VGP', colors=['#1c92fc'] * len(PARAMETERS.get('Sfc - 3km VGP', {}).get('levels', [])),
         barb_color=SPC_BARB)

# 0-3 km MLCAPE (red, every 25 J/kg) + surface vorticity (light blue) + goldenrod winds
_c3 = list(range(25, 325, 25))
_spc_set('Sfc Vorticity / Sfc - 3km MLCAPE', cape3_levels=_c3, cape3_colors=['#fc0204'] * len(_c3),
         cape3_linewidths=[1.6] * len(_c3), vort_color='#04b2ec', barb_color=SPC_BARB)

# SARS hail size: same bands, SPC's exact colors
_spc_set('SARS Hail Size (in)',
         cin_colors=['#ffffff00', '#fcfe04', '#fc0204', '#0402fc', '#1c92fc', '#04b2ec', '#04eeec', '#fce6dc'])

# SARS sig-hail %: fill in SPC greens/yellow/red; base-% contours dark red / salmon / purple / cyan
_sh = PARAMETERS.get('SARS Sig. Hail Percentage', {}).get('levels', [])
_spc_set('SARS Sig. Hail Percentage',
         colors=_by_level(_sh, [(5, '#8c0204'), (20, '#fc6a6c'), (75, '#8c6acc'), (1e9, '#04eeec')]),
         cin_colors=['#048a04', '#7cfe04', '#ecee04', '#fcd604', '#cc0204'])


# --- SPC STYLE MATCHING (batch 3): surface, upper air, moisture ---
# MSLP: black isobars every 2 hPa with goldenrod 10 m wind barbs
_spc_set('Mean Sea Level Pressure (hPa)', color='black', linewidth=ISOBAR_WIDTH, contour_range=(940, 1072, 2),
         title='MSL Pressure (hPa) & 10m Wind (kt)', sfc_barbs=True, barb_color=SPC_BARB)

# Mixing ratio / theta: hatched MR bands (SPC legend 9/11/13/15/17/20), blue dashed MR lines,
# red theta every 2 K, black surface barbs
_spc_set('Mixing Ratio / Theta',
         mr_levels=[9, 11, 13, 15, 17, 20, 40],
         mr_colors=['#04eeec', '#048a04', '#04ce04', '#7cfe04', '#fcd604', '#fcaebc'],
         mr_hatch=True, mr_contour_levels=list(range(1, 31)), mr_contour_color='#1c92fc',
         theta_color='#fc0204', sfc_barbs=True, barb_color='black',
         title='Theta (K, Red), Mixing Ratio (g/kg) & 10m Wind (kt)')

# Moisture convergence: no fill; black MR every 2 g/kg, red convergence, brown barbs
_spc_set('Moisture Convergence & Mixing Ratio', mr_fill=False, mr_contour_levels=list(range(2, 32, 2)),
         mconv_color='#fc0204', barb_color='#8c4624')

# 850mb: heights black (3 dam), temp dashed red / blue at or below 0, dewpoint green; no isotach shading
_spc_set('850mb Wind Speed (kt)', temp_dynamic_color=True,
         temp_cold_color='#0402fc', temp_warm_color='#fc0204', dwpt_levels=None,
         isotach_levels=[25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 400],
         isotach_colors=['#04eeec', '#1c92fc', '#1c92fc', '#04b2ec', '#04b2ec',
                         '#8c6acc', '#8c6acc', '#942eec', '#942eec', '#8c028c'],
         isotach_outline=None, barb_color=SPC_BARB,
         title='850mb Heights (dam), Temp (°C) & Wind (kt, hatched >= 25 kt)')

# 700mb: same temps; 700-500mb mean RH >= 70% hatched dark green with bright green 70/80/90 lines
_spc_set('700mb Wind Speed (kt)', temp_dynamic_color=True, temp_cold_color='#0402fc', temp_warm_color='#fc0204',
         rh_hatch=True)

# 500mb: temps all red dashed; hatched isotachs 40/60/80/100/120/140 kt
_spc_set('500mb Wind Speed (kt)', temp_dynamic_color=False, temp_warm_color='#fc0204',
         isotach_levels=[40, 60, 80, 100, 120, 140, 400],
         isotach_colors=['#1c92fc', '#04b2ec', '#04eeec', '#8c6acc', '#942eec', '#8c028c'],
         title='500mb Heights (dam), Temp (°C), Isotachs (kt, hatched) & Wind (kt)')

# 300mb: hatched isotachs 60/80/100/120/140/160 kt; magenta divergence
_spc_set('300mb Wind Speed (kt)',
         isotach_levels=[60, 80, 100, 120, 140, 160, 400],
         isotach_colors=['#1c92fc', '#04b2ec', '#04eeec', '#8c6acc', '#942eec', '#8c028c'],
         div_color='#fc02fc',
         title='300mb Heights (dam), Isotachs (kt, hatched), Divergence (1e-5/s) & Wind (kt)')

# Precipitable water: hatched 1.0/1.5/2.0 in. greens; black lines every 0.1 in.
for _n in ('Precipitable Water (in)', 'Precipitable Water & 850mb Moisture Transport'):
    _spc_set(_n, fill_colors=['#7cfe04', '#04ce04', '#048a04'], fill_hatch=True, vector_color='#8c0204')

# 100mb mean mixing ratio: hatched 10-18 g/kg; brown surface streamlines
_spc_set('100mb Mean Mixing Ratio', fill_colors=['#7cfe04', '#04ce04', '#048a04', '#ecee04', '#fcd604'],
         fill_hatch=True, vector_color='#8c4624')


# --- SPC STYLE MATCHING (batch 4): winter (read from SPC archive 2024-01-13 18Z; SNSQ from live) ---
_spc_set('Precipitation Type', ptype_fontsize=8.5, ptype_colors={
    0: '#a4a2a4', 1: '#8c0204', 2: '#ec2e2c', 3: '#ec2e2c', 4: '#942eec', 5: '#942eec',
    6: '#8c6acc', 7: '#04eeec', 8: '#04b2ec', 9: '#04b2ec', 10: '#04b2ec'})
_spc_set('Lake Effect Snow 1', rh_fill_colors=['#7cfe04', '#04ce04', '#048a04'],
         dt_color='#fc7e04', cape_color='#fc0204')
_spc_set('Lake Effect Snow 2', conv_color='#fc0204', barb_color=SPC_BARB)
_spc_set('Snow Squall Parameter', cin_colors=['#144e8c', '#1c92fc', '#04b2ec', '#04eeec',
                                              '#8c6acc', '#942eec', '#8c028c', '#fc02fc', '#fcaebc'])
_dgz_d = PARAMETERS.get('Dendritic Growth Layer Depth & RH', {}).get('depth_levels', [])
_spc_set('Dendritic Growth Layer Depth & RH',
         depth_colors=['#942eec'] * len(_dgz_d), depth_linewidths=[1.1] * len(_dgz_d),
         rh_colors=['#cc8604', '#cc8604', '#04ce04', '#04ce04', '#048a04'],
         rh_fill_hatch=True, rh_fill_color='#048a04')

# --- Readability round 2 ---
_spc_set('Surface Temp, Dewpoint, and PMSL', temp_linewidth=0.55, freeze_linewidth=1.5, dwpt_linewidth=0.9,
         smooth_sigma_km_conus=36.0)
_spc_set('MSLP, Theta-E, & Wind', smooth_sigma_km_conus=36.0)
for _n in ('850mb Wind Speed (kt)', '500mb Wind Speed (kt)', '300mb Wind Speed (kt)'):
    _spc_set(_n, isotach_outline='#144e8c', isotach_tint=0.35)

PARAMETERS.pop('10m Wind Speed (kt)', None)   # removed from the app

SPC_CATEGORIES = {
    'Surface': [
        'Surface Temp, Dewpoint, and PMSL',
        'Mean Sea Level Pressure (hPa)',
        'MSLP, Theta-E, & Wind',
        'Moisture Convergence & Mixing Ratio',
        'Mixing Ratio / Theta',
    ],
    'Upper Air': [
        '850mb Wind Speed (kt)',
        '700mb Wind Speed (kt)',
        '500mb Wind Speed (kt)',
        '300mb Wind Speed (kt)',
    ],
    'Thermodynamics': [
        'Surface-Based CAPE & CIN',
        '100mb Mixed-Layer CAPE & CIN',
        'Downdraft CAPE & DCIN',
        'Mid-Level Lapse Rate (700-500mb)',
        'Low-Level Lapse Rate (0-3km)',
        '100mb Mixed-Layer LCL Height (m)',
        '100mb Mean Mixing Ratio',
    ],
    'Wind Shear': [
        'Effective Bulk Shear (kt)',
        'Surface - 1km Shear (kt)',
        'Surface - 3km Shear (kt)',
        'Surface - 6km Shear (kt)',
        'Surface - 8km Shear (kt)',
        'Effective SRH & Inflow Base',
        'Sfc-500m SRH & Inflow Base',
        'Sfc-1km SRH & Inflow Base',
        'Sfc-3km SRH',
        'Surface - 2km SR Winds',
        '4 - 6km SR Winds',
        '9 - 11km SR Winds',
    ],
    'Composite Indices': [
        'Supercell Composite (Right-Moving)',
        'Supercell Composite (Left-Moving)',
        'Significant Tornado Parameter (Fixed Layer)',
        'Significant Tornado Parameter (Effective Layer)',
        'Significant Tornado Parameter (0-500m SRH)',
        'Conditional SigTor Prob. (Eqn. 1)',
        'Conditional SigTor Prob. (Eqn. 2)',
        'Non-Supercell Tornado Parameter',
        'Violent Tornado Parameter',
        'Significant Hail Parameter',
        'SARS Hail Size (in)',
        'SARS Sig. Hail Percentage',
        'Large Hail Parameter',
        'Derecho Composite',
        'Craven Brooks Significant Severe',
        'Bulk Richardson Number',
        'MCS Maintenance',
        'Microburst Composite',
        'Enhanced Stretching Potential',
        'Sfc - 1km EHI',
        'Sfc - 3km EHI',
        'Sfc - 3km VGP',
    ],
    'Multi-Parameter Fields': [
        'Sfc Vorticity / Sfc - 3km MLCAPE',
    ],
    'Heavy Rainfall': [
        'Precipitable Water (in)',
        'Precipitable Water & 850mb Moisture Transport',
    ],
    'Winter Weather': [
        'Precipitation Type',
        'Lake Effect Snow 1',
        'Lake Effect Snow 2',
        'Snow Squall Parameter',
        'Dendritic Growth Layer Depth & RH',
    ]
}


# --- 3D THERMODYNAMIC & WINTER SOLVER HELPERS ---
# --- MEMORY: RUN 3D CALCULATIONS IN STRIPS OF ROWS ---
# On Full CONUS the 3D fallbacks (DCAPE, 0-3km MLCAPE, P-type, SNSQ, DGZ, lake-effect) used to hold
# the whole 3D column (~37 levels x 1.9M points, float64) plus temporaries at once: several GB.
# Each works column-by-column, so running them on strips of rows gives identical results with a
# peak of a few hundred MB. Only one heavy calculation runs at a time.
STRIP_MAX_POINTS = 60_000


def _mixr_is_g_per_kg(ds_main, z_dim):
    """Decide mixing-ratio units ONCE for the whole domain. The per-function check looks at the
    median of whatever it is handed, which could flip between strips over dry or cold areas."""
    units_attr = str(ds_main['mixr'].attrs.get('units', '')).lower().replace(' ', '')
    if units_attr in ('g/kg', 'gkg-1', 'gkg**-1'):
        return True
    if units_attr in ('kg/kg', 'kgkg-1', 'kgkg**-1'):
        return False
    bottom = int(np.argmax(ds_main[z_dim].values))  # highest pressure = lowest level
    return float(np.nanmedian(ds_main['mixr'].isel({z_dim: bottom}).values)) > 0.1


def _in_strips(func, ds_main, z_dim, max_points=STRIP_MAX_POINTS):
    y_dim, x_dim = ds_main['tmpk_2m'].dims[-2:]
    ny, nx = ds_main.sizes[y_dim], ds_main.sizes[x_dim]
    g_per_kg = _mixr_is_g_per_kg(ds_main, z_dim) if 'mixr' in ds_main.data_vars else False

    def _prep(ds_part):
        # Every strip gets kg/kg, so all strips are handled the same way
        return ds_part.assign(mixr=ds_part['mixr'] * 0.001) if g_per_kg else ds_part

    with _shared_state()['compute_lock']:
        if ny * nx <= max_points:
            return func(_prep(ds_main), z_dim)
        step = max(1, max_points // nx)
        parts = [func(_prep(ds_main.isel({y_dim: slice(r, r + step)})), z_dim) for r in range(0, ny, step)]
    if isinstance(parts[0], tuple):
        return tuple(np.concatenate([p[k] for p in parts], axis=0) for k in range(len(parts[0])))
    return np.concatenate(parts, axis=0)


def compute_les1_fields(ds_main, z_dim):
    """Lake Effect Snow 1 fields: BL RH, Sfc-850mb temp difference, SBCAPE (column-by-column)."""
    p_1d = ds_main[z_dim].values.astype(np.float64)
    if p_1d[0] > 2000:
        p_1d = p_1d / 100.0
    spatial_dims = [d for d in ds_main['tmpk'].dims if d != z_dim]
    T_3d = ds_main['tmpk'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    q_3d = ds_main['mixr'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    if np.nanmedian(q_3d) > 0.1:
        q_3d = q_3d * 0.001

    p_sfc = ds_main['pres_sfc'].values.astype(np.float64) / 100.0
    T_sfc = ds_main['tmpk_2m'].values.astype(np.float64)
    Td_sfc = ds_main['dwpk_2m'].values.astype(np.float64)

    es_sfc = 6.112 * np.exp(17.67 * (T_sfc - 273.15) / np.maximum(T_sfc - 29.65, 1.0))
    e_sfc = 6.112 * np.exp(17.67 * (Td_sfc - 273.15) / np.maximum(Td_sfc - 29.65, 1.0))
    rh_sfc = np.clip((e_sfc / np.maximum(es_sfc, 1e-5)) * 100.0, 0.0, 100.0)

    p_3d = p_1d[:, None, None]
    es_3d = 6.112 * np.exp(17.67 * (T_3d - 273.15) / np.maximum(T_3d - 29.65, 1.0))
    e_3d = (p_3d * q_3d) / (0.62198 + q_3d)
    rh_3d = np.clip((e_3d / np.maximum(es_3d, 1e-5)) * 100.0, 0.0, 100.0)

    in_bl = (p_3d <= p_sfc[None, :, :]) & (p_3d >= (p_sfc[None, :, :] - 100.0))
    cnt = np.sum(in_bl, axis=0)
    bl_rh = (rh_sfc + np.sum(np.where(in_bl, rh_3d, 0.0), axis=0)) / (cnt + 1.0)

    target_850 = 850 * 100 if ds_main[z_dim].values[0] > 2000 else 850
    T_850 = ds_main.sel({z_dim: target_850}, method='nearest')['tmpk'].values.astype(np.float64)
    dt_sfc_850 = T_sfc - T_850
    sbcape = np.maximum(ds_main['sb_cape'].values.astype(np.float64), 0.0)
    return bl_rh, dt_sfc_850, sbcape


def compute_bl_wind(ds_main, z_dim):
    """Lowest-100 hPa mean wind (column-by-column)."""
    p_1d = ds_main[z_dim].values.astype(np.float64)
    if p_1d[0] > 2000:
        p_1d = p_1d / 100.0
    spatial_dims = [d for d in ds_main['uwin'].dims if d != z_dim]
    u_3d = ds_main['uwin'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    v_3d = ds_main['vwin'].transpose(z_dim, *spatial_dims).values.astype(np.float64)

    p_sfc = ds_main['pres_sfc'].values.astype(np.float64) / 100.0
    u_10m = ds_main['uwin_10m'].values.astype(np.float64)
    v_10m = ds_main['vwin_10m'].values.astype(np.float64)

    p_3d = p_1d[:, None, None]
    in_bl = (p_3d <= p_sfc[None, :, :]) & (p_3d >= (p_sfc[None, :, :] - 100.0))
    cnt = np.sum(in_bl, axis=0)
    u_bl = (u_10m + np.sum(np.where(in_bl, u_3d, 0.0), axis=0)) / (cnt + 1.0)
    v_bl = (v_10m + np.sum(np.where(in_bl, v_3d, 0.0), axis=0)) / (cnt + 1.0)
    return u_bl, v_bl


def compute_dcape_3d(ds_main, z_dim):
    Rd, cpd, Lv, eps = 287.058, 1005.7, 2.501e6, 0.62198
    p_1d = ds_main[z_dim].values.astype(np.float64)
    if p_1d[0] > 2000:
        p_1d = p_1d / 100.0
    spatial_dims = [d for d in ds_main['tmpk'].dims if d != z_dim]
    T_3d = ds_main['tmpk'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    q_3d = ds_main['mixr'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    if np.nanmedian(q_3d) > 0.1:
        q_3d = q_3d * 0.001
    if p_1d[0] < p_1d[-1]:
        p_1d = p_1d[::-1]
        T_3d, q_3d = T_3d[::-1], q_3d[::-1]

    p_sfc = ds_main['pres_sfc'].values.astype(np.float64) / 100.0
    T_sfc = ds_main['tmpk_2m'].values.astype(np.float64)
    Td_sfc = ds_main['dwpk_2m'].values.astype(np.float64)
    p_3d = p_1d[:, None, None]

    e_3d = np.maximum((p_3d * q_3d) / (eps + q_3d), 1e-5)
    ln_e = np.log(e_3d / 6.112)
    Td_3d = np.minimum((243.5 * ln_e) / (17.67 - ln_e) + 273.15, T_3d)
    TL = 1.0 / (1.0 / np.maximum(Td_3d - 56.0, 1.0) + np.log(np.maximum(T_3d / Td_3d, 1.0)) / 800.0) + 56.0
    theta_e = T_3d * (1000.0 / p_3d) ** (0.2854 * (1.0 - 0.28 * q_3d)) * np.exp(
        q_3d * (1.0 + 0.81 * q_3d) * (3376.0 / TL - 2.54)
    )

    valid_layer = (p_3d <= (p_sfc[None, :, :] - 50.0)) & (p_3d >= (p_sfc[None, :, :] - 450.0))
    min_idx = np.argmin(np.where(valid_layer, theta_e, np.inf), axis=0)
    jj, ii = np.meshgrid(np.arange(T_3d.shape[1]), np.arange(T_3d.shape[2]), indexing='ij')
    p_start = p_1d[min_idx]
    T_start = T_3d[min_idx, jj, ii]
    q_start = q_3d[min_idx, jj, ii]

    Tw = T_start.copy()
    for _ in range(4):
        es_w = 6.112 * np.exp(17.67 * (Tw - 273.15) / (Tw - 29.65))
        qs_w = eps * es_w / np.maximum(p_start - es_w, 1.0)
        des_dT = es_w * (17.67 * 243.5) / ((Tw - 29.65) ** 2)
        dqs_dT = (eps * p_start / (np.maximum(p_start - es_w, 1.0) ** 2)) * des_dT
        Tw = Tw - (cpd * (Tw - T_start) + Lv * (qs_w - q_start)) / (cpd + Lv * dqs_dT)

    e_sfc = 6.112 * np.exp(17.67 * (Td_sfc - 273.15) / (Td_sfc - 29.65))
    q_sfc = eps * e_sfc / np.maximum(p_sfc - e_sfc, 1.0)
    Tv_sfc = T_sfc * (1.0 + 0.608 * q_sfc)
    Tv_3d = T_3d * (1.0 + 0.608 * q_3d)

    n_steps = 16
    ln_p_start = np.log(np.minimum(p_start, p_sfc))
    dlnp = (np.log(p_sfc) - ln_p_start) / n_steps
    Tp = Tw.copy()
    dcape = np.zeros_like(p_sfc)

    for step in range(n_steps):
        p_mid = np.exp(ln_p_start + (step + 0.5) * dlnp)
        es_p = 6.112 * np.exp(17.67 * (Tp - 273.15) / (Tp - 29.65))
        qs_p = eps * es_p / np.maximum(p_mid - es_p, 1.0)
        gamma_m = (Rd * Tp + Lv * qs_p) / (cpd + (Lv ** 2 * qs_p * eps) / (Rd * Tp ** 2))
        Tp_mid = Tp + 0.5 * gamma_m * dlnp
        Tp = Tp + gamma_m * dlnp

        es_m = 6.112 * np.exp(17.67 * (Tp_mid - 273.15) / (Tp_mid - 29.65))
        qs_m = eps * es_m / np.maximum(p_mid - es_m, 1.0)
        Tv_parcel = Tp_mid * (1.0 + 0.608 * qs_m)

        idx_hi = np.clip(np.searchsorted(-p_1d, -p_mid), 1, len(p_1d) - 1)
        idx_lo = idx_hi - 1
        p_lo, p_hi = p_1d[idx_lo], p_1d[idx_hi]
        Tv_lo, Tv_hi = Tv_3d[idx_lo, jj, ii], Tv_3d[idx_hi, jj, ii]

        case1 = p_mid > p_lo
        case2 = (~case1) & (p_lo >= p_sfc)
        p_bot = np.where(case1 | case2, p_sfc, p_lo)
        Tv_bot = np.where(case1 | case2, Tv_sfc, Tv_lo)
        p_top = np.where(case1, p_lo, p_hi)
        Tv_top = np.where(case1, Tv_lo, Tv_hi)

        w = np.clip((np.log(p_mid) - np.log(p_bot)) / np.minimum(np.log(p_top) - np.log(p_bot), -1e-6), 0.0, 1.0)
        Tv_env = Tv_bot + w * (Tv_top - Tv_bot)
        dcape += Rd * np.maximum(Tv_env - Tv_parcel, 0.0) * dlnp

    return dcape


def compute_mlcape_0_3km(ds_main, z_dim):
    Rd, cpd, Lv, eps = 287.058, 1005.7, 2.501e6, 0.62198
    kappa = Rd / cpd
    p_1d = ds_main[z_dim].values.astype(np.float64)
    if p_1d[0] > 2000:
        p_1d = p_1d / 100.0
    spatial_dims = [d for d in ds_main['tmpk'].dims if d != z_dim]
    T_3d = ds_main['tmpk'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    q_3d = ds_main['mixr'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    z_3d = ds_main['hght'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    if np.nanmedian(q_3d) > 0.1:
        q_3d = q_3d * 0.001
    if p_1d[0] < p_1d[-1]:
        p_1d = p_1d[::-1]
        T_3d, q_3d, z_3d = T_3d[::-1], q_3d[::-1], z_3d[::-1]

    p_sfc = ds_main['pres_sfc'].values.astype(np.float64) / 100.0
    T_sfc = ds_main['tmpk_2m'].values.astype(np.float64)
    Td_sfc = ds_main['dwpk_2m'].values.astype(np.float64)
    sfc_var = next((v for v in ['hght_sfc', 'orog', 'sfc_hght'] if v in ds_main.data_vars), None)
    z_sfc = ds_main[sfc_var].values.astype(np.float64) if sfc_var else np.nanmin(z_3d, axis=0)

    p_3d = p_1d[:, None, None]
    theta_3d = T_3d * (1000.0 / p_3d) ** kappa
    e_sfc = 6.112 * np.exp(17.67 * (Td_sfc - 273.15) / (Td_sfc - 29.65))
    q_sfc = eps * e_sfc / np.maximum(p_sfc - e_sfc, 1.0)
    theta_sfc = T_sfc * (1000.0 / p_sfc) ** kappa

    in_ml = (p_3d <= p_sfc[None, :, :]) & (p_3d >= (p_sfc[None, :, :] - 100.0))
    cnt = np.sum(in_ml, axis=0)
    theta_ml = (theta_sfc + np.sum(np.where(in_ml, theta_3d, 0.0), axis=0)) / (cnt + 1.0)
    q_ml = (q_sfc + np.sum(np.where(in_ml, q_3d, 0.0), axis=0)) / (cnt + 1.0)

    p_start = p_sfc.copy()
    T_start = theta_ml * (p_start / 1000.0) ** kappa
    e_ml = np.maximum((p_start * q_ml) / (eps + q_ml), 1e-5)
    ln_e = np.log(e_ml / 6.112)
    Td_start = np.minimum((243.5 * ln_e) / (17.67 - ln_e) + 273.15, T_start)
    T_lcl = 1.0 / (1.0 / np.maximum(Td_start - 56.0, 1.0) + np.log(np.maximum(T_start / Td_start, 1.0)) / 800.0) + 56.0
    p_lcl = p_start * (T_lcl / T_start) ** (1.0 / kappa)

    z_target = z_sfc + 3000.0
    p_3km = np.full_like(p_sfc, 700.0)
    for k in range(len(p_1d) - 1):
        z_lo, z_hi = z_3d[k], z_3d[k + 1]
        in_bracket = (z_target >= z_lo) & (z_target <= z_hi)
        if np.any(in_bracket):
            frac = (z_target - z_lo) / np.maximum(z_hi - z_lo, 1.0)
            lnp_interp = np.log(p_1d[k]) + frac * (np.log(p_1d[k + 1]) - np.log(p_1d[k]))
            p_3km = np.where(in_bracket, np.exp(lnp_interp), p_3km)
    p_3km = np.minimum(p_3km, p_start - 10.0)

    Tv_sfc = T_sfc * (1.0 + 0.608 * q_sfc)
    Tv_3d = T_3d * (1.0 + 0.608 * q_3d)
    jj, ii = np.meshgrid(np.arange(T_3d.shape[1]), np.arange(T_3d.shape[2]), indexing='ij')

    n_steps = 20
    ln_p_bot = np.log(p_start)
    dlnp = (np.log(p_3km) - ln_p_bot) / n_steps
    Tp = T_start.copy()
    cape_3km = np.zeros_like(p_sfc)

    for step in range(n_steps):
        p_mid = np.exp(ln_p_bot + (step + 0.5) * dlnp)
        saturated = p_mid < p_lcl
        Tp_dry_mid = theta_ml * (p_mid / 1000.0) ** kappa
        Tv_dry = Tp_dry_mid * (1.0 + 0.608 * q_ml)

        Tp_moist_in = np.where(Tp > T_lcl, T_lcl, Tp)
        es_p = 6.112 * np.exp(17.67 * (Tp_moist_in - 273.15) / (Tp_moist_in - 29.65))
        qs_p = eps * es_p / np.maximum(p_mid - es_p, 1.0)
        gamma_m = (Rd * Tp_moist_in + Lv * qs_p) / (cpd + (Lv ** 2 * qs_p * eps) / (Rd * Tp_moist_in ** 2))
        Tp_moist_mid = Tp_moist_in + 0.5 * gamma_m * dlnp
        Tp_moist_next = Tp_moist_in + gamma_m * dlnp

        es_m = 6.112 * np.exp(17.67 * (Tp_moist_mid - 273.15) / (Tp_moist_mid - 29.65))
        qs_m = eps * es_m / np.maximum(p_mid - es_m, 1.0)
        Tv_moist = Tp_moist_mid * (1.0 + 0.608 * qs_m)

        Tp = np.where(saturated, Tp_moist_next, Tp_dry_mid)
        Tv_parcel = np.where(saturated, Tv_moist, Tv_dry)

        idx_hi = np.clip(np.searchsorted(-p_1d, -p_mid), 1, len(p_1d) - 1)
        idx_lo = idx_hi - 1
        p_lo, p_hi = p_1d[idx_lo], p_1d[idx_hi]
        Tv_lo, Tv_hi = Tv_3d[idx_lo, jj, ii], Tv_3d[idx_hi, jj, ii]

        case1 = p_mid > p_lo
        case2 = (~case1) & (p_lo >= p_sfc)
        p_b = np.where(case1 | case2, p_sfc, p_lo)
        Tv_b = np.where(case1 | case2, Tv_sfc, Tv_lo)
        p_t = np.where(case1, p_lo, p_hi)
        Tv_t = np.where(case1, Tv_lo, Tv_hi)

        w = np.clip((np.log(p_mid) - np.log(p_b)) / np.minimum(np.log(p_t) - np.log(p_b), -1e-6), 0.0, 1.0)
        Tv_env = Tv_b + w * (Tv_t - Tv_b)
        cape_3km += Rd * np.maximum(Tv_parcel - Tv_env, 0.0) * (-dlnp)

    return cape_3km


def compute_dgz_depth_and_rh(ds_main, z_dim):
    eps = 0.62198
    p_1d = ds_main[z_dim].values.astype(np.float64)
    if p_1d[0] > 2000:
        p_1d = p_1d / 100.0
    spatial_dims = [d for d in ds_main['tmpk'].dims if d != z_dim]
    T_3d = ds_main['tmpk'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    q_3d = ds_main['mixr'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    z_3d = ds_main['hght'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    if np.nanmedian(q_3d) > 0.1:
        q_3d = q_3d * 0.001
    if p_1d[0] < p_1d[-1]:
        p_1d = p_1d[::-1]
        T_3d, q_3d, z_3d = T_3d[::-1], q_3d[::-1], z_3d[::-1]

    p_sfc = ds_main['pres_sfc'].values.astype(np.float64) / 100.0
    sfc_var = next((v for v in ['hght_sfc', 'orog', 'sfc_hght'] if v in ds_main.data_vars), None)
    z_sfc = ds_main[sfc_var].values.astype(np.float64) if sfc_var else np.nanmin(z_3d, axis=0)

    p_3d = p_1d[:, None, None]
    T_c = T_3d - 273.15
    es = 6.112 * np.exp(17.67 * T_c / np.maximum(T_3d - 29.65, 1.0))
    e = (p_3d * q_3d) / (eps + q_3d)
    rh_3d = np.clip((e / np.maximum(es, 1e-5)) * 100.0, 0.0, 100.0)

    z_m12 = np.full_like(p_sfc, np.nan)
    z_m17 = np.full_like(p_sfc, np.nan)
    for k in range(len(p_1d) - 1):
        above_sfc = p_1d[k + 1] <= p_sfc
        t0_k, t1_k = T_c[k], T_c[k + 1]
        z0_k, z1_k = z_3d[k], z_3d[k + 1]

        cross12 = above_sfc & np.isnan(z_m12) & (t0_k >= -12.0) & (t1_k <= -12.0)
        if np.any(cross12):
            denom = np.where(np.abs(t1_k - t0_k) > 1e-5, t1_k - t0_k, -1e-5)
            frac = np.clip((-12.0 - t0_k) / denom, 0.0, 1.0)
            z_m12 = np.where(cross12, z0_k + frac * (z1_k - z0_k), z_m12)

        cross17 = above_sfc & np.isnan(z_m17) & (t0_k >= -17.0) & (t1_k <= -17.0)
        if np.any(cross17):
            denom = np.where(np.abs(t1_k - t0_k) > 1e-5, t1_k - t0_k, -1e-5)
            frac = np.clip((-17.0 - t0_k) / denom, 0.0, 1.0)
            z_cand = z0_k + frac * (z1_k - z0_k)
            valid17 = cross17 & (np.isnan(z_m12) | (z_cand >= z_m12))
            z_m17 = np.where(valid17, z_cand, z_m17)

    sfc_below_m12 = np.isnan(z_m12) & (~np.isnan(z_m17)) & (z_m17 > z_sfc)
    z_m12 = np.where(sfc_below_m12, z_sfc, z_m12)
    dgz_depth = np.maximum(z_m17 - z_m12, 0.0)

    in_dgz = (p_3d <= p_sfc[None, :, :]) & (T_c <= -12.0) & (T_c >= -17.0)
    cnt = np.sum(in_dgz, axis=0)
    dgz_rh = np.where(cnt > 0, np.sum(np.where(in_dgz, rh_3d, 0.0), axis=0) / np.maximum(cnt, 1), np.nan)

    if np.any((cnt == 0) & (dgz_depth > 0)):
        z_mid = 0.5 * (z_m12 + z_m17)
        nearest_k = np.argmin(np.abs(z_3d - z_mid[None, :, :]), axis=0)
        jj, ii = np.meshgrid(np.arange(T_3d.shape[1]), np.arange(T_3d.shape[2]), indexing='ij')
        dgz_rh = np.where((cnt == 0) & (dgz_depth > 0), rh_3d[nearest_k, jj, ii], dgz_rh)

    return dgz_depth, dgz_rh


def compute_snsq_3d(ds_main, z_dim):
    Rd, cpd, Lv, eps = 287.058, 1005.7, 2.501e6, 0.62198
    kappa = Rd / cpd
    p_1d = ds_main[z_dim].values.astype(np.float64)
    if p_1d[0] > 2000:
        p_1d = p_1d / 100.0
    spatial_dims = [d for d in ds_main['tmpk'].dims if d != z_dim]
    T_3d = ds_main['tmpk'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    q_3d = ds_main['mixr'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    z_3d = ds_main['hght'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    u_3d = ds_main['uwin'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    v_3d = ds_main['vwin'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    if np.nanmedian(q_3d) > 0.1:
        q_3d = q_3d * 0.001
    if p_1d[0] < p_1d[-1]:
        p_1d = p_1d[::-1]
        T_3d, q_3d, z_3d, u_3d, v_3d = T_3d[::-1], q_3d[::-1], z_3d[::-1], u_3d[::-1], v_3d[::-1]

    p_sfc = ds_main['pres_sfc'].values.astype(np.float64) / 100.0
    T_sfc = ds_main['tmpk_2m'].values.astype(np.float64)
    Td_sfc = ds_main['dwpk_2m'].values.astype(np.float64)
    u_10m = ds_main['uwin_10m'].values.astype(np.float64)
    v_10m = ds_main['vwin_10m'].values.astype(np.float64)
    sfc_var = next((v for v in ['hght_sfc', 'orog', 'sfc_hght'] if v in ds_main.data_vars), None)
    z_sfc = ds_main[sfc_var].values.astype(np.float64) if sfc_var else np.nanmin(z_3d, axis=0)
    z_agl = z_3d - z_sfc[None, :, :]
    p_3d = p_1d[:, None, None]

    es_3d = 6.112 * np.exp(17.67 * (T_3d - 273.15) / np.maximum(T_3d - 29.65, 1.0))
    e_3d = (p_3d * q_3d) / (eps + q_3d)
    rh_3d = np.clip((e_3d / np.maximum(es_3d, 1e-5)) * 100.0, 0.0, 100.0)

    es_sfc = 6.112 * np.exp(17.67 * (T_sfc - 273.15) / np.maximum(T_sfc - 29.65, 1.0))
    e_sfc = 6.112 * np.exp(17.67 * (Td_sfc - 273.15) / np.maximum(Td_sfc - 29.65, 1.0))
    rh_sfc = np.clip((e_sfc / np.maximum(es_sfc, 1e-5)) * 100.0, 0.0, 100.0)
    q_sfc = eps * e_sfc / np.maximum(p_sfc - e_sfc, 1.0)
    theta_e_sfc = T_sfc * (1000.0 / p_sfc) ** kappa * np.exp((Lv * q_sfc) / (cpd * T_sfc))

    in_2km = (z_agl > 0.0) & (z_agl <= 2000.0)
    cnt = np.sum(in_2km, axis=0)
    rh_mean_2km = (rh_sfc + np.sum(np.where(in_2km, rh_3d, 0.0), axis=0)) / (cnt + 1.0)
    spd_mean_2km = (np.hypot(u_10m, v_10m) + np.sum(np.where(in_2km, np.hypot(u_3d, v_3d), 0.0), axis=0)) / (cnt + 1.0)

    z_target = z_sfc + 2000.0
    T_2km = np.full_like(p_sfc, np.nan)
    p_2km = np.full_like(p_sfc, np.nan)
    q_2km = np.full_like(p_sfc, np.nan)
    for k in range(len(p_1d) - 1):
        z0, z1 = z_3d[k], z_3d[k + 1]
        in_br = (z_target >= z0) & (z_target <= z1)
        if np.any(in_br):
            f = (z_target - z0) / np.maximum(z1 - z0, 1.0)
            T_2km = np.where(in_br, T_3d[k] + f * (T_3d[k + 1] - T_3d[k]), T_2km)
            q_2km = np.where(in_br, q_3d[k] + f * (q_3d[k + 1] - q_3d[k]), q_2km)
            p_2km = np.where(in_br, np.exp(np.log(p_1d[k]) + f * (np.log(p_1d[k + 1]) - np.log(p_1d[k]))), p_2km)

    theta_e_2km = T_2km * (1000.0 / np.maximum(p_2km, 100.0)) ** kappa * np.exp((Lv * q_2km) / (cpd * np.maximum(T_2km, 150.0)))
    delta_te = theta_e_2km - theta_e_sfc
    snsq = np.maximum((rh_mean_2km - 60.0) / 15.0, 0.0) * np.maximum((4.0 - delta_te) / 4.0, 0.0) * (spd_mean_2km / 9.0)
    return np.where((T_sfc - 273.15) <= 3.0, snsq, 0.0)


def compute_ptype_spc(ds_main, z_dim):
    direct_var = next((v for v in ['ptype', 'wx_type', 'precip_type', 'spc_ptype'] if v in ds_main.data_vars), None)
    if direct_var is not None:
        return np.rint(ds_main[direct_var].values).astype(int)

    mask_keys = {
        'rain': next((v for v in ['crain', 'rain_mask', 'cat_rain'] if v in ds_main.data_vars), None),
        'frzr': next((v for v in ['cfrzr', 'frzr_mask', 'cat_frzr'] if v in ds_main.data_vars), None),
        'icep': next((v for v in ['cicep', 'icep_mask', 'cat_icep'] if v in ds_main.data_vars), None),
        'snow': next((v for v in ['csnow', 'snow_mask', 'cat_snow'] if v in ds_main.data_vars), None),
    }
    if any(mask_keys.values()):
        ny, nx = ds_main['tmpk_2m'].shape
        ptype = np.full((ny, nx), -1, dtype=int)
        r = (ds_main[mask_keys['rain']].values > 0.5) if mask_keys['rain'] else np.zeros((ny, nx), dtype=bool)
        z = (ds_main[mask_keys['frzr']].values > 0.5) if mask_keys['frzr'] else np.zeros((ny, nx), dtype=bool)
        i = (ds_main[mask_keys['icep']].values > 0.5) if mask_keys['icep'] else np.zeros((ny, nx), dtype=bool)
        s = (ds_main[mask_keys['snow']].values > 0.5) if mask_keys['snow'] else np.zeros((ny, nx), dtype=bool)
        ptype = np.where(r, 0, ptype)
        ptype = np.where(s, 7, ptype)
        ptype = np.where(i, 5, ptype)
        ptype = np.where(z, 2, ptype)
        ptype = np.where(z & i, 3, ptype)
        ptype = np.where(i & s, 6, ptype)
        return ptype

    p_1d = ds_main[z_dim].values.astype(np.float64)
    if p_1d[0] > 2000:
        p_1d = p_1d / 100.0
    spatial_dims = [d for d in ds_main['tmpk'].dims if d != z_dim]
    T_3d = ds_main['tmpk'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    q_3d = ds_main['mixr'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
    if np.nanmedian(q_3d) > 0.1:
        q_3d = q_3d * 0.001

    p_sfc = ds_main['pres_sfc'].values.astype(np.float64) / 100.0
    T_sfc = ds_main['tmpk_2m'].values.astype(np.float64)
    Td_sfc = ds_main['dwpk_2m'].values.astype(np.float64)

    es_sfc = 6.112 * np.exp(17.67 * (T_sfc - 273.15) / np.maximum(T_sfc - 29.65, 1.0))
    e_sfc = 6.112 * np.exp(17.67 * (Td_sfc - 273.15) / np.maximum(Td_sfc - 29.65, 1.0))
    rh_sfc = np.clip((e_sfc / np.maximum(es_sfc, 1e-5)) * 100.0, 0.0, 100.0)
    Tc_sfc = T_sfc - 273.15
    Tw_sfc = (
        Tc_sfc * np.arctan(0.151977 * np.sqrt(rh_sfc + 8.313659))
        + np.arctan(Tc_sfc + rh_sfc) - np.arctan(rh_sfc - 1.676331)
        + 0.00391838 * (rh_sfc ** 1.5) * np.arctan(0.023101 * rh_sfc) - 4.686035
    )

    p_3d = p_1d[:, None, None]
    Tc_3d = T_3d - 273.15
    es_3d = 6.112 * np.exp(17.67 * Tc_3d / np.maximum(T_3d - 29.65, 1.0))
    e_3d = (p_3d * q_3d) / (0.62198 + q_3d)
    rh_3d = np.clip((e_3d / np.maximum(es_3d, 1e-5)) * 100.0, 0.0, 100.0)
    Tw_3d = (
        Tc_3d * np.arctan(0.151977 * np.sqrt(rh_3d + 8.313659))
        + np.arctan(Tc_3d + rh_3d) - np.arctan(rh_3d - 1.676331)
        + 0.00391838 * (rh_3d ** 1.5) * np.arctan(0.023101 * rh_3d) - 4.686035
    )

    above_sfc = p_3d <= p_sfc[None, :, :]
    max_tw_aloft = np.nanmax(np.where(above_sfc, Tw_3d, -99.0), axis=0)
    bl_mask = above_sfc & (p_3d >= (p_sfc[None, :, :] - 150.0))
    bl_n = np.sum(bl_mask & np.isfinite(rh_3d), axis=0)
    bl_rh = np.where(bl_n > 0, np.nansum(np.where(bl_mask, rh_3d, 0.0), axis=0) / np.maximum(bl_n, 1), np.nan)

    ptype = np.full_like(Tw_sfc, -1, dtype=int)
    active = bl_rh >= 70.0
    ptype = np.where(active & (Tw_sfc > 0.5), 0, ptype)
    ptype = np.where(active & (Tw_sfc <= 0.5) & (max_tw_aloft <= 0.5), 7, ptype)
    ptype = np.where(active & (Tw_sfc <= 0.0) & (max_tw_aloft > 1.5), 1, ptype)
    ptype = np.where(active & (Tw_sfc <= 0.0) & (max_tw_aloft > 0.5) & (max_tw_aloft <= 1.5), 5, ptype)
    return ptype


# --- DATA EXTRACTION ---
# --- BELOW-GROUND FILL FOR PRESSURE-LEVEL MAPS ---
# The SFCOA-HRRR file leaves 850/700mb blank wherever that level is underground (most of the West).
# Model pressure-level output used for maps like SPC's normally extrapolates there instead. We do the
# same: start from the nearest level above ground and come down to the target level using a standard
# 6.5 K/km lapse rate (temperature) and the hypsometric equation (height). Winds and moisture take the
# value from that nearest above-ground level. These are not real observations, only continuity.
_LAPSE_EXP = 0.190263          # Rd * 6.5 K/km / g
_RD_OVER_G = 287.05 / 9.80665


def _level_fields_filled(ds_main, z_dim, ds_level, level_hpa):
    names = [n for n in ('tmpk', 'hght', 'uwin', 'vwin', 'mixr') if n in ds_level.data_vars]
    out = {n: ds_level[n].values.astype(np.float64) for n in names}
    if 'tmpk' not in out or 'hght' not in out:
        return out
    need = ~(np.isfinite(out['tmpk']) & np.isfinite(out['hght']))
    if not need.any():
        return out
    p_all = ds_main[z_dim].values.astype(np.float64)
    p_hpa = p_all / 100.0 if np.nanmax(p_all) > 2000 else p_all
    # levels above the target (lower pressure), nearest first
    for k in [int(k) for k in np.argsort(-p_hpa) if p_hpa[k] < level_hpa - 0.5]:
        if not need.any():
            break
        lev_k = ds_main.isel({z_dim: k})
        t_k = lev_k['tmpk'].values.astype(np.float64)
        z_k = lev_k['hght'].values.astype(np.float64)
        use = need & np.isfinite(t_k) & np.isfinite(z_k)
        if not use.any():
            continue
        p_k = p_hpa[k]
        t_new = t_k * (level_hpa / p_k) ** _LAPSE_EXP
        z_new = z_k - _RD_OVER_G * 0.5 * (t_k + t_new) * np.log(level_hpa / p_k)
        out['tmpk'] = np.where(use, t_new, out['tmpk'])
        out['hght'] = np.where(use, z_new, out['hght'])
        for n in ('uwin', 'vwin', 'mixr'):
            if n in out:
                out[n] = np.where(use & ~np.isfinite(out[n]), lev_k[n].values.astype(np.float64), out[n])
        need &= ~use
    return out


def get_parameter_data(ds_main, param_key):
    cfg = PARAMETERS[param_key]

    if 'isobaric' in ds_main.dims:
        z_dim = 'isobaric'
    elif 'isobaricInhPa' in ds_main.dims:
        z_dim = 'isobaricInhPa'
    else:
        z_dim = 'pressure'

    if cfg.get('plot_type') == 'upper_air':
        level = cfg['level']
        target_level = level * 100 if ds_main[z_dim].values[0] > 2000 else level
        ds_level = ds_main.sel({z_dim: target_level}, method='nearest')
        # Fill points where this level is below ground (the SFCOA file leaves them blank), so contours
        # run continuously across the high terrain like SPC's maps.
        lv = _level_fields_filled(ds_main, z_dim, ds_level, level)

        u = lv['uwin'] * cfg['scale']
        v = lv['vwin'] * cfg['scale']
        speed = np.hypot(u, v)

        extra_fields = {}
        if 'height_levels' in cfg:
            extra_fields['heights_dam'] = lv['hght'] / 10.0
        if 'temp_levels' in cfg:
            extra_fields['temp_c'] = lv['tmpk'] - 273.15
        if cfg.get('plot_rh'):
            target_500 = 500 * 100 if ds_main[z_dim].values[0] > 2000 else 500
            ds_500 = ds_main.sel({z_dim: target_500}, method='nearest')
            rh_700 = mpcalc.relative_humidity_from_mixing_ratio(700 * units.hPa, lv['tmpk'] * units.K, lv['mixr']).to('percent')
            rh_500 = mpcalc.relative_humidity_from_mixing_ratio(500 * units.hPa, ds_500['tmpk'].values * units.K, ds_500['mixr'].values).to('percent')
            extra_fields['mean_rh'] = (rh_700.magnitude + rh_500.magnitude) / 2.0
        if cfg.get('dwpt_levels') is not None and 'mixr' in lv:
            q = lv['mixr'].astype(np.float64)
            if _mixr_is_g_per_kg(ds_main, z_dim):
                q = q * 0.001
            e_hpa = np.maximum(level * q / (0.62198 + q), 1e-3)
            ln_e = np.log(e_hpa / 6.112)
            extra_fields['dwpt_c'] = 243.5 * ln_e / (17.67 - ln_e)
        if cfg.get('plot_divergence'):
            u_metric = lv['uwin'] * units('m/s')
            v_metric = lv['vwin'] * units('m/s')
            dx, dy = mpcalc.lat_lon_grid_deltas(ds_main.longitude.values, ds_main.latitude.values)
            div = mpcalc.divergence(u_metric, v_metric, dx=dx, dy=dy) * 1e5
            extra_fields['divergence'] = ndimage.gaussian_filter(div.magnitude, sigma=15)

        return speed, (u, v), extra_fields

    elif param_key == '10m Wind Speed (kt)':
        u = ds_main['uwin_10m'].values * cfg['scale']
        v = ds_main['vwin_10m'].values * cfg['scale']
        return np.hypot(u, v), (u, v), None

    elif cfg.get('plot_type') == 'theta_e_composite':
        t2m = ds_main['tmpk_2m'].values * units.K
        d2m = ds_main['dwpk_2m'].values * units.K
        sp = ds_main['pres_sfc'].values * units.Pa
        theta_e = mpcalc.equivalent_potential_temperature(sp, t2m, d2m).to('kelvin')
        mslp = ds_main['pres_msl'].values * 0.01
        return (theta_e.magnitude, mslp), (ds_main['uwin_10m'].values * 1.94384, ds_main['vwin_10m'].values * 1.94384), None

    elif cfg.get('plot_type') == 'sfc_composite':
        t2m_f = (ds_main['tmpk_2m'].values - 273.15) * 9/5 + 32
        d2m_f = (ds_main['dwpk_2m'].values - 273.15) * 9/5 + 32
        mslp = ds_main['pres_msl'].values * 0.01
        return (t2m_f, d2m_f, mslp), (ds_main['uwin_10m'].values * 1.94384, ds_main['vwin_10m'].values * 1.94384), None

    elif cfg.get('plot_type') == 'composite':
        t2m = ds_main['tmpk_2m'].values * units.K
        d2m = ds_main['dwpk_2m'].values * units.K
        sp = ds_main['pres_sfc'].values * units.Pa
        theta = mpcalc.potential_temperature(sp, t2m).to('kelvin')
        mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(sp, t2m, mpcalc.relative_humidity_from_dewpoint(t2m, d2m)).to('g/kg')
        return (theta.magnitude, mixing_ratio.magnitude), None, None

    elif cfg.get('plot_type') == 'moisture_convergence':
        t2m = ds_main['tmpk_2m'].values * units.K
        d2m = ds_main['dwpk_2m'].values * units.K
        sp = ds_main['pres_sfc'].values * units.Pa
        u_ms = ds_main['uwin_10m'].values * units('m/s')
        v_ms = ds_main['vwin_10m'].values * units('m/s')
        mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(sp, t2m, mpcalc.relative_humidity_from_dewpoint(t2m, d2m)).to('g/kg')
        dx, dy = mpcalc.lat_lon_grid_deltas(ds_main.longitude.values, ds_main.latitude.values)
        mfc = -mpcalc.divergence(u_ms * mixing_ratio, v_ms * mixing_ratio, dx=dx, dy=dy) * 1e5
        mfc_smoothed = ndimage.gaussian_filter(mfc.magnitude, sigma=8)
        return (mixing_ratio.magnitude, mfc_smoothed), (ds_main['uwin_10m'].values * 1.94384, ds_main['vwin_10m'].values * 1.94384), None

    elif cfg.get('plot_type') == 'shear':
        scale = cfg.get('scale', 1.94384)
        u_candidates = cfg['var_u'] if isinstance(cfg['var_u'], list) else [cfg['var_u']]
        v_candidates = cfg['var_v'] if isinstance(cfg['var_v'], list) else [cfg['var_v']]
        u_var = next((v for v in u_candidates if v in ds_main.data_vars), None)
        v_var = next((v for v in v_candidates if v in ds_main.data_vars), None)

        if u_var is not None and v_var is not None:
            u_kt = ds_main[u_var].values * scale
            v_kt = ds_main[v_var].values * scale
        elif 'srw_layer_m' in cfg:
            z_bot, z_top = cfg['srw_layer_m']
            spatial_dims = [d for d in ds_main['uwin'].dims if d != z_dim]
            u_3d = ds_main['uwin'].transpose(z_dim, *spatial_dims).values
            v_3d = ds_main['vwin'].transpose(z_dim, *spatial_dims).values
            z_3d = ds_main['hght'].transpose(z_dim, *spatial_dims).values

            sfc_var = next((v for v in ['hght_sfc', 'orog', 'sfc_hght'] if v in ds_main.data_vars), None)
            z_sfc = ds_main[sfc_var].values if sfc_var else np.nanmin(z_3d, axis=0)
            z_agl = z_3d - z_sfc[None, :, :]

            in_layer = (z_agl >= z_bot) & (z_agl <= z_top)
            count = np.sum(in_layer, axis=0)
            u_mean = np.where(count > 0, np.nansum(np.where(in_layer, u_3d, 0.0), axis=0) / np.maximum(count, 1), np.nan)
            v_mean = np.where(count > 0, np.nansum(np.where(in_layer, v_3d, 0.0), axis=0) / np.maximum(count, 1), np.nan)

            if np.any(count == 0):
                z_mid = 0.5 * (z_bot + z_top)
                nearest_k = np.argmin(np.abs(z_agl - z_mid), axis=0)
                jj, ii = np.meshgrid(np.arange(u_mean.shape[0]), np.arange(u_mean.shape[1]), indexing='ij')
                u_mean = np.where(count > 0, u_mean, u_3d[nearest_k, jj, ii])
                v_mean = np.where(count > 0, v_mean, v_3d[nearest_k, jj, ii])

            u_kt = (u_mean - ds_main['supercell_right_mover_u'].values) * scale
            v_kt = (v_mean - ds_main['supercell_right_mover_v'].values) * scale
        else:
            raise KeyError(f"Variables {u_candidates} / {v_candidates} not found in dataset.")

        return np.hypot(u_kt, v_kt), (u_kt, v_kt), None

    elif cfg.get('plot_type') == 'cape_cin':
        cape = ds_main[cfg['var_name_cape']].values
        if cfg['var_name_cape'] == 'dcape':
            cape = np.abs(cape)
            if np.nanmax(cape) < 1.0:
                cape = _in_strips(compute_dcape_3d, ds_main, z_dim)
        try:
            cin = np.abs(ds_main[cfg['var_name_cin']].values)
        except KeyError:
            cin = np.zeros_like(cape)
        return (cape, cin), None, None

    elif cfg.get('plot_type') == 'esrh_composite':
        srh_candidates = cfg['var_srh'] if isinstance(cfg['var_srh'], list) else [cfg['var_srh']]
        srh_var = next((v for v in srh_candidates if v in ds_main.data_vars), None)
        if srh_var is None:
            raise KeyError(f"None of {srh_candidates} found in file.")
        srh = ds_main[srh_var].values
        eil_base = ds_main[cfg['var_eil_base']].values if cfg.get('plot_eil', True) else None
        return (srh, eil_base), (ds_main[cfg['var_u']].values * 1.94384, ds_main[cfg['var_v']].values * 1.94384), None

    elif cfg.get('plot_type') == 'composite_index':
        candidates = cfg['var_index'] if isinstance(cfg['var_index'], list) else [cfg['var_index']]
        found_var = next((v for v in candidates if v in ds_main.data_vars), None)

        def _compute_fallback_field(fb_type):
            ebwd = np.hypot(ds_main['effective_shear_u'].values, ds_main['effective_shear_v'].values)
            shr6 = np.hypot(ds_main['shear_0_6km_u'].values, ds_main['shear_0_6km_v'].values)
            mlcape = np.maximum(ds_main['ml_cape'].values, 0.0)
            mucape = np.maximum(ds_main['mu_cape'].values, 0.0)
            sbcape = np.maximum(ds_main['sb_cape'].values, 0.0)
            mlcin = np.abs(ds_main['ml_cinh'].values)
            mucin = np.abs(ds_main['mu_cinh'].values)
            lr75 = np.maximum(ds_main['lapse_rate_700_500hPa'].values, 0.0)
            lr3 = np.maximum(ds_main['lapse_rate_0_3km'].values, 0.0)

            if fb_type in ('scp_right', 'scp_left'):
                cin_term = np.where(mucin <= 40.0, 1.0, 40.0 / np.maximum(mucin, 1e-6))
                ebwd_term = np.where(ebwd < 10.0, 0.0, np.minimum(ebwd / 20.0, 1.5))
                if fb_type == 'scp_right':
                    srh = np.maximum(ds_main['srh_eff'].values, 0.0)
                    return (mucape / 1000.0) * (srh / 50.0) * ebwd_term * cin_term
                else:
                    srh_l_var = next((v for v in ['srh_eff_left', 'srh_0_3km_left'] if v in ds_main.data_vars), None)
                    srh_l = np.minimum(ds_main[srh_l_var].values, 0.0) if srh_l_var else -np.maximum(ds_main['srh_eff'].values, 0.0)
                    return (mucape / 1000.0) * (srh_l / 50.0) * ebwd_term * cin_term

            elif fb_type == 'stp_fixed':
                sblcl = ds_main['sb_lcl_hght_agl'].values
                srh1 = np.maximum(ds_main['srh_0_1km'].values, 0.0)
                lcl_term = np.clip((2000.0 - sblcl) / 1000.0, 0.0, 1.0)
                shr_term = np.where(shr6 < 12.5, 0.0, np.minimum(shr6 / 20.0, 1.5))
                return (sbcape / 1500.0) * lcl_term * (srh1 / 150.0) * shr_term

            elif fb_type in ('stp_eff', 'stp_500m', 'vtp'):
                mllcl = ds_main['ml_lcl_hght_agl'].values
                lcl_term = np.clip((2000.0 - mllcl) / 1000.0, 0.0, 1.0)
                cin_term = np.clip((200.0 - mlcin) / 150.0, 0.0, 1.0)
                ebwd_term = np.where(ebwd < 12.5, 0.0, np.minimum(ebwd / 20.0, 1.5))
                if fb_type == 'stp_500m':
                    srh500_var = next((v for v in ['srh_0_500m', 'esrh_0_500m'] if v in ds_main.data_vars), 'srh_eff')
                    srh_term = np.maximum(ds_main[srh500_var].values, 0.0) / 75.0
                else:
                    srh_term = np.maximum(ds_main['srh_eff'].values, 0.0) / 150.0
                val = (mlcape / 1500.0) * lcl_term * srh_term * ebwd_term * cin_term
                if fb_type == 'vtp':
                    cape3_var = next((v for v in ['cape_0_3km', 'ml_cape_0_3km', 'sb_cape_0_3km'] if v in ds_main.data_vars), None)
                    cape3 = np.maximum(ds_main[cape3_var].values, 0.0) if cape3_var else np.minimum(mlcape * 0.15, 250.0)
                    val = val * (cape3 / 50.0) * (lr3 / 6.5)
                return val

            elif fb_type in ('cprob_1', 'cprob_2'):
                srh1 = np.maximum(ds_main['srh_0_1km'].values, 0.0)
                z = -3.8 + 0.0006 * mlcape + 0.075 * shr6 + 0.006 * srh1 - 0.012 * mlcin
                if fb_type == 'cprob_2':
                    z = z + 1.5e-6 * (mlcape * srh1) - 0.15
                return np.where(mlcape > 50.0, 100.0 / (1.0 + np.exp(-z)), 0.0)

            elif fb_type == 'nstp':
                lr1_var = next((v for v in ['lapse_rate_0_1km', 'lapse_rate_0_3km'] if v in ds_main.data_vars), 'lapse_rate_0_3km')
                lr = np.maximum(ds_main[lr1_var].values, 0.0)
                cape3_var = next((v for v in ['cape_0_3km', 'ml_cape_0_3km', 'sb_cape_0_3km'] if v in ds_main.data_vars), None)
                cape3 = np.maximum(ds_main[cape3_var].values, 0.0) if cape3_var else np.minimum(mlcape * 0.15, 250.0)
                cin_term = np.clip((250.0 - mlcin) / 250.0, 0.0, 1.0)
                shr_term = np.clip((18.0 - shr6) / 18.0, 0.0, 1.0)
                return (lr / 9.0) * (cape3 / 100.0) * cin_term * shr_term

            elif fb_type in ('ship', 'sars_size', 'sars_base_pct', 'sars_sig_pct', 'lhp'):
                target_500 = 500 * 100 if ds_main[z_dim].values[0] > 2000 else 500
                t500_c = ds_main.sel({z_dim: target_500}, method='nearest')['tmpk'].values - 273.15
                t2m = ds_main['tmpk_2m'].values * units.K
                d2m = ds_main['dwpk_2m'].values * units.K
                sp = ds_main['pres_sfc'].values * units.Pa
                mr = mpcalc.mixing_ratio_from_relative_humidity(
                    sp, t2m, mpcalc.relative_humidity_from_dewpoint(t2m, d2m)
                ).to('g/kg').magnitude
                mr_cap = np.clip(mr, 0.0, 13.6)
                shr_cap = np.clip(shr6, 0.0, 27.0)
                t500_term = np.maximum(-t500_c, 5.5)
                ship = (mucape * mr_cap * lr75 * t500_term * shr_cap) / 42000000.0
                ship = np.where(mucape < 1300.0, ship * (mucape / 1300.0), ship)
                ship = np.where(lr75 < 5.8, ship * (lr75 / 5.8), ship)
                if fb_type == 'ship':
                    return ship
                elif fb_type == 'sars_size':
                    return np.where(mucape > 100.0, 0.5 + 1.35 * np.sqrt(np.maximum(ship, 0.0)), 0.0)
                elif fb_type == 'sars_base_pct':
                    return np.where(mucape > 100.0, np.clip(100.0 * (1.0 - np.exp(-0.65 * ship)), 0.0, 100.0), 0.0)
                elif fb_type == 'sars_sig_pct':
                    return np.where(mucape > 100.0, np.clip(100.0 * (1.0 - np.exp(-0.48 * np.maximum(ship - 0.3, 0.0))), 0.0, 100.0), 0.0)
                elif fb_type == 'lhp':
                    return np.where(mucape >= 400.0, (mucape / 2000.0) * (lr75 / 7.0) * (shr6 / 18.0) * 6.0, 0.0)

            elif fb_type in ('dcp', 'mbcp'):
                dcape = np.abs(ds_main['dcape'].values)
                if np.nanmax(dcape) < 1.0:
                    dcape = _in_strips(compute_dcape_3d, ds_main, z_dim)
                if fb_type == 'dcp':
                    u_mean6 = np.hypot(ds_main['uwin_10m'].values + 0.5 * ds_main['shear_0_6km_u'].values,
                                       ds_main['vwin_10m'].values + 0.5 * ds_main['shear_0_6km_v'].values)
                    return (dcape / 980.0) * (mucape / 2000.0) * (shr6 / 20.0) * (u_mean6 / 16.0)
                else:
                    return np.where((sbcape > 100.0) & (lr3 > 6.5),
                                    (sbcape / 2000.0) * (dcape / 800.0) * (lr3 / 7.5) * 4.0, 0.0)

            elif fb_type == 'craven_sigsig':
                return (mlcape * shr6) / 1000.0
            elif fb_type == 'brn':
                return np.where(mlcape >= 50.0, mlcape / (0.5 * np.maximum(shr6, 1.0) ** 2), np.nan)
            elif fb_type == 'mcsm':
                return np.where(mucape > 100.0, (mucape / 1500.0) * (shr6 / 15.0) * 35.0, 0.0)
            elif fb_type == 'esp':
                cape3_var = next((v for v in ['cape_0_3km', 'ml_cape_0_3km', 'sb_cape_0_3km'] if v in ds_main.data_vars), None)
                cape3 = np.maximum(ds_main[cape3_var].values, 0.0) if cape3_var else np.minimum(mlcape * 0.15, 250.0)
                return np.where(lr3 > 7.0, (cape3 / 50.0) * ((lr3 - 7.0) / 1.0), 0.0)
            elif fb_type == 'ehi1':
                return (sbcape * np.maximum(ds_main['srh_0_1km'].values, 0.0)) / 160000.0
            elif fb_type == 'ehi3':
                return (sbcape * np.maximum(ds_main['srh_0_3km'].values, 0.0)) / 160000.0
            elif fb_type == 'vgp3':
                shr3 = np.hypot(ds_main['shear_0_3km_u'].values, ds_main['shear_0_3km_v'].values)
                return np.sqrt(sbcape) * (shr3 / 3000.0)
            elif fb_type == 'snsq':
                return _in_strips(compute_snsq_3d, ds_main, z_dim)

            return np.zeros_like(mlcape)

        idx_val = ds_main[found_var].values.astype(np.float64) if found_var is not None else _compute_fallback_field(cfg.get('fallback_type'))

        if cfg.get('scale_thousands') and np.nanmax(idx_val) > 500.0:
            idx_val = idx_val / 1000.0
        if cfg.get('is_percent') and np.nanmax(idx_val) <= 1.05:
            idx_val = idx_val * 100.0
        if cfg.get('force_negative') and np.nanmax(idx_val) > 0 and np.nanmin(idx_val) >= 0:
            idx_val = -np.abs(idx_val)

        cin_val = None
        if cfg.get('fill_self'):
            cin_val = idx_val.copy()
        elif 'cin_var' in cfg:
            cin_candidates = cfg['cin_var'] if isinstance(cfg['cin_var'], list) else [cfg['cin_var']]
            c_var = next((v for v in cin_candidates if v in ds_main.data_vars), None)
            if c_var is not None:
                cin_val = ds_main[c_var].values.astype(np.float64)
                if not cfg.get('cin_is_percent'):
                    cin_val = np.abs(cin_val)
            elif 'fallback_cin' in cfg:
                cin_val = _compute_fallback_field(cfg['fallback_cin'])
            if cin_val is not None and cfg.get('cin_is_percent') and np.nanmax(cin_val) <= 1.05:
                cin_val = cin_val * 100.0

        wind_tuple = None
        if 'wind_u' in cfg and 'wind_v' in cfg:
            wind_tuple = (ds_main[cfg['wind_u']].values * 1.94384, ds_main[cfg['wind_v']].values * 1.94384)

        return (idx_val, cin_val), wind_tuple, None

    elif cfg.get('plot_type') == 'vort_mlcape3':
        u_ms = ds_main['uwin_10m'].values * units('m/s')
        v_ms = ds_main['vwin_10m'].values * units('m/s')
        dx, dy = mpcalc.lat_lon_grid_deltas(ds_main.longitude.values, ds_main.latitude.values)
        vort_1e5 = (mpcalc.vorticity(u_ms, v_ms, dx=dx, dy=dy) * 1e5).magnitude
        cape3_var = next((v for v in ['ml_cape_0_3km', 'cape_0_3km', 'ml3cape', 'mlcape_0_3km'] if v in ds_main.data_vars), None)
        cape_3km = np.maximum(ds_main[cape3_var].values, 0.0) if cape3_var is not None else _in_strips(compute_mlcape_0_3km, ds_main, z_dim)
        return (vort_1e5, cape_3km), (ds_main['uwin_10m'].values * 1.94384, ds_main['vwin_10m'].values * 1.94384), None

    elif cfg.get('plot_type') == 'moisture_field':
        candidates = cfg['var_field'] if isinstance(cfg['var_field'], list) else [cfg['var_field']]
        found_var = next((v for v in candidates if v in ds_main.data_vars), None)
        fb = cfg.get('fallback_type')

        if found_var is not None:
            field = ds_main[found_var].values.astype(np.float64)
            if fb == 'pwat' and np.nanmedian(field) > 5.0:
                field = field / 25.4
            elif fb == 'ml_mixr' and np.nanmedian(field) < 0.1:
                field = field * 1000.0
        else:
            p_1d = ds_main[z_dim].values.astype(np.float64)
            if p_1d[0] > 2000:
                p_1d = p_1d / 100.0
            spatial_dims = [d for d in ds_main['mixr'].dims if d != z_dim]
            q_3d = ds_main['mixr'].transpose(z_dim, *spatial_dims).values.astype(np.float64)
            if np.nanmedian(q_3d) > 0.1:
                q_3d = q_3d * 0.001
            if p_1d[0] < p_1d[-1]:
                p_1d = p_1d[::-1]
                q_3d = q_3d[::-1]

            p_sfc = ds_main['pres_sfc'].values.astype(np.float64) / 100.0
            t2m = ds_main['tmpk_2m'].values * units.K
            d2m = ds_main['dwpk_2m'].values * units.K
            sp = ds_main['pres_sfc'].values * units.Pa
            q_sfc = mpcalc.mixing_ratio_from_relative_humidity(
                sp, t2m, mpcalc.relative_humidity_from_dewpoint(t2m, d2m)
            ).to('kg/kg').magnitude

            p_3d = p_1d[:, None, None]
            if fb == 'pwat':
                pw_mm = np.zeros_like(p_sfc)
                for k in range(len(p_1d) - 1):
                    p_bot = np.minimum(p_1d[k], p_sfc)
                    p_top = np.minimum(p_1d[k + 1], p_sfc)
                    dp_pa = np.maximum(p_bot - p_top, 0.0) * 100.0
                    pw_mm += (0.5 * (q_3d[k] + q_3d[k + 1]) * dp_pa) / 9.80665
                field = pw_mm / 25.4
            else:
                in_ml = (p_3d <= p_sfc[None, :, :]) & (p_3d >= (p_sfc[None, :, :] - 100.0))
                cnt = np.sum(in_ml, axis=0)
                field = ((q_sfc + np.sum(np.where(in_ml, q_3d, 0.0), axis=0)) / (cnt + 1.0)) * 1000.0

        vec_mode = cfg.get('vector_mode')
        vec_data = None
        if vec_mode == 'quiver_850_mtrans':
            target_850 = 850 * 100 if ds_main[z_dim].values[0] > 2000 else 850
            ds_850 = ds_main.sel({z_dim: target_850}, method='nearest')
            q_850 = ds_850['mixr'].values.astype(np.float64)
            if np.nanmedian(q_850) < 0.1:
                q_850 = q_850 * 1000.0
            vec_data = (q_850 * ds_850['uwin'].values, q_850 * ds_850['vwin'].values)
        elif vec_mode == 'streamlines_sfc':
            vec_data = (ds_main['uwin_10m'].values, ds_main['vwin_10m'].values)

        return field, vec_data, None

    elif cfg.get('plot_type') == 'winter_ptype':
        return _in_strips(compute_ptype_spc, ds_main, z_dim), None, None

    elif cfg.get('plot_type') == 'les1':
        return _in_strips(compute_les1_fields, ds_main, z_dim), None, None

    elif cfg.get('plot_type') == 'les2':
        u_bl, v_bl = _in_strips(compute_bl_wind, ds_main, z_dim)
        dx, dy = mpcalc.lat_lon_grid_deltas(ds_main.longitude.values, ds_main.latitude.values)
        div_1e5 = (mpcalc.divergence(u_bl * units('m/s'), v_bl * units('m/s'), dx=dx, dy=dy) * 1e5).magnitude
        return div_1e5, (u_bl, v_bl), None

    elif cfg.get('plot_type') == 'dgz_composite':
        return _in_strips(compute_dgz_depth_and_rh, ds_main, z_dim), None, None

    else:
        return ds_main[cfg['var_name']].values * cfg['scale'], None, None


# --- SHARED DRAWING HELPERS ---
HRRR_DX_KM = 3.0

def _sigma_pts(cfg, wfo):
    """Gaussian smoothing width in grid points from a width in km (0 = off).
    Full CONUS uses 'smooth_sigma_km_conus' when set, since a whole-country view needs smoother lines."""
    km = cfg.get('smooth_sigma_km', 0.0)
    if wfo == 'Full CONUS':
        km = cfg.get('smooth_sigma_km_conus', km)
    return float(km) / HRRR_DX_KM


def _smooth(arr, sigma):
    """NaN-safe Gaussian smoothing; contours from raw 3-km HRRR are noisy and hard to read."""
    arr = np.asarray(arr, dtype=np.float64)
    if sigma <= 0:
        return arr
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return ndimage.gaussian_filter(arr, sigma=sigma)
    filled = np.where(nan_mask, 0.0, arr)
    weights = ndimage.gaussian_filter((~nan_mask).astype(np.float64), sigma=sigma)
    out = ndimage.gaussian_filter(filled, sigma=sigma) / np.maximum(weights, 1e-8)
    out[nan_mask] = np.nan
    return out


def _hatched_bands(ax, x, y, z, levels, colors, hatch='////', zorder=1.5, tint_alpha=0.0):
    """SPC-style colored diagonal hatching, one color per band, transparent between lines.
    tint_alpha > 0 adds a see-through fill in a lighter shade of each band's color underneath."""
    if tint_alpha > 0:
        tints = [tuple(0.5 + 0.5 * c for c in mcolors.to_rgb(col)) for col in colors]   # halfway to white
        ax.contourf(x, y, z, levels=levels, colors=tints, alpha=tint_alpha, zorder=zorder - 0.05)
    for lo, hi, col in zip(levels[:-1], levels[1:], colors):
        with matplotlib.rc_context({'hatch.color': col, 'hatch.linewidth': 0.9}):
            cs = ax.contourf(x, y, z, levels=[lo, hi], colors='none', hatches=[hatch], zorder=zorder)
        if hasattr(cs, 'set_hatchcolor'):  # newer Matplotlib
            cs.set_hatchcolor(col)


def _band_legend(fig, ax, levels, colors, label):
    """Solid color-box legend under the map, like SPC's (boxes labeled by their lower bound)."""
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels[:len(colors) + 1], cmap.N)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.04, shrink=0.45, aspect=30,
                      ticks=levels[:len(colors)], spacing='uniform', label=label)
    cb.ax.set_xticklabels([str(v) for v in levels[:len(colors)]])
    return cb


def _draw_sfc_barbs(ax, cfg, ds_active, _crop, lons_c, lats_c, skip_barb):
    """10 m wind barbs (kt) for maps that SPC draws with surface wind, when the map's cfg sets 'sfc_barbs'."""
    if not cfg.get('sfc_barbs'):
        return
    try:
        u = _crop(ds_active['uwin_10m'].values) * 1.94384
        v = _crop(ds_active['vwin_10m'].values) * 1.94384
    except Exception:
        return
    ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
             u[::skip_barb, ::skip_barb], v[::skip_barb, ::skip_barb],
             length=5, color=cfg.get('barb_color', 'black'), linewidth=0.6,
             transform=ccrs.PlateCarree(), zorder=3)


# --- CACHED FAST PNG RENDERER (Native Lambert Conformal Grid + Spatial Crop) ---
# SPEEDUP: no ttl. The file path is already part of the cache key, so a new NOMADS cycle gets
# new maps automatically; the old ttl=300 threw away every finished map every 5 minutes.
def _draw_map_png(file_path, parameter, wfo='Full CONUS'):
    from matplotlib.colors import LinearSegmentedColormap

    ds_active, lons, lats, x_proj, y_proj = get_domain_dataset_and_grid(file_path, wfo)
    data, wind_data, extra_fields = get_parameter_data(ds_active, parameter)

    extent = WFO_DOMAINS.get(wfo, [-126.0, -66.0, 23.0, 50.0])
    lon_min, lon_max, lat_min, lat_max = extent

    # Crop 2D grid to the selected sector (+buffer) for 15x faster plotting & zero contour artifacts
    ny, nx = lons.shape
    if wfo == 'Full CONUS':
        r0, r1, c0, c1 = 0, ny, 0, nx
        skip_fill = 4
        skip_barb = 45
    else:
        pad = 1.5
        mask = (
            (lons >= lon_min - pad) & (lons <= lon_max + pad) &
            (lats >= lat_min - pad) & (lats <= lat_max + pad)
        )
        if np.any(mask):
            rows, cols = np.where(mask)
            r0, r1 = max(0, rows.min() - 2), min(ny, rows.max() + 3)
            c0, c1 = max(0, cols.min() - 2), min(nx, cols.max() + 3)
        else:
            r0, r1, c0, c1 = 0, ny, 0, nx
        skip_fill = 1
        skip_barb = max(6, int(min(r1 - r0, c1 - c0) / 14))

    def _crop(arr):
        return arr[r0:r1, c0:c1]

    lons_c = _crop(lons)
    lats_c = _crop(lats)
    x_c = _crop(x_proj)
    y_c = _crop(y_proj)

    x_sub = x_c[::skip_fill, ::skip_fill]
    y_sub = y_c[::skip_fill, ::skip_fill]
    lons_sub = lons_c[::skip_fill, ::skip_fill]
    lats_sub = lats_c[::skip_fill, ::skip_fill]

    # Object-oriented Figure (thread/session safe — zero global plt state!)
    fig = Figure(figsize=(10.5, 8.2), dpi=RENDER_DPI)
    ax = fig.add_subplot(1, 1, 1, projection=PROJECTION)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # MEMORY: fixed 50m detail. Cartopy's default switches to the very large 10m files when zoomed in.
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=PARAMETERS[parameter].get('ocean_color', OCEAN_COLOR), zorder=1)
    # SPC look: thin red CWA outlines underneath, gray state lines drawn on top of them
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor=STATE_LINE_COLOR, linewidth=STATE_LINE_WIDTH, zorder=STATE_ZORDER)
    # Coastlines and country borders in the same gray (outlines Mexico; the states layer has no Mexican states)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor=STATE_LINE_COLOR, linewidth=STATE_LINE_WIDTH, zorder=STATE_ZORDER)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor=STATE_LINE_COLOR, linewidth=STATE_LINE_WIDTH, zorder=STATE_ZORDER)
    # Great Salt Lake and Lake Okeechobee, same gray as the states
    lakes = _near(load_extra_lakes(), lon_min, lon_max, lat_min, lat_max)
    if lakes:
        ax.add_geometries(lakes, ccrs.PlateCarree(), edgecolor=STATE_LINE_COLOR, facecolor='none',
                          linewidth=STATE_LINE_WIDTH, zorder=STATE_ZORDER)
    # U.S. counties: lightest gray, thinnest line, underneath everything else on the base map
    counties = load_county_geometries('20m' if wfo == 'Full CONUS' else '5m')
    if wfo != 'Full CONUS':
        counties = _near(counties, lon_min, lon_max, lat_min, lat_max)
    if counties:
        ax.add_geometries(counties, ccrs.PlateCarree(), edgecolor=COUNTY_LINE_COLOR, facecolor='none',
                          linewidth=COUNTY_LINE_WIDTH, zorder=COUNTY_ZORDER)

    if _CWA_GEOMETRY_CACHE is not None:
        if wfo == 'Full CONUS':
            geoms_to_draw = _CWA_GEOMETRY_CACHE
        else:
            bbox_poly = shapely_box(lon_min - 1.0, lat_min - 1.0, lon_max + 1.0, lat_max + 1.0)
            geoms_to_draw = [g for g in _CWA_GEOMETRY_CACHE if g.intersects(bbox_poly)]
        if geoms_to_draw:
            ax.add_geometries(
                geoms_to_draw, ccrs.PlateCarree(),
                edgecolor=CWA_LINE_COLOR, facecolor='none', linewidth=CWA_LINE_WIDTH, alpha=1.0, zorder=CWA_ZORDER
            )

    cfg = PARAMETERS[parameter]

    # Note: All contour / contourf calls plot directly in native (x_sub, y_sub) Lambert Conformal meters
    # without transform=ccrs.PlateCarree(), preventing Cartopy curvilinear wrap lines!
    if cfg['plot_type'] == 'composite':
        theta, mixing_ratio = data
        mr_sub = _crop(mixing_ratio)[::skip_fill, ::skip_fill]
        th_sub = _crop(theta)[::skip_fill, ::skip_fill]
        if cfg.get('mr_hatch'):
            _hatched_bands(ax, x_sub, y_sub, mr_sub, cfg['mr_levels'], cfg['mr_colors'], zorder=1.5)
            _band_legend(fig, ax, cfg['mr_levels'], cfg['mr_colors'], 'Mixing Ratio (g/kg)')
        else:
            cf_mr = ax.contourf(x_sub, y_sub, mr_sub, levels=cfg['mr_levels'], colors=cfg['mr_colors'], alpha=0.6, zorder=1.5)
            fig.colorbar(cf_mr, ax=ax, orientation='horizontal', pad=0.04, shrink=0.72, label='Mixing Ratio (g/kg)')
        if cfg.get('mr_contour_levels') is not None:
            cs_mrl = ax.contour(x_sub, y_sub, mr_sub, levels=cfg['mr_contour_levels'], colors=cfg.get('mr_contour_color', '#1c92fc'),
                                linestyles='dashed', linewidths=0.9, zorder=1.8)
            ax.clabel(cs_mrl, inline=True, fmt='%d', fontsize=7.5)
        th_c = cfg.get('theta_color', 'red')
        cs_theta = ax.contour(x_sub, y_sub, th_sub, levels=np.arange(240, 360, 2), colors=th_c, linewidths=1.2, zorder=2)
        ax.clabel(cs_theta, inline=True, fmt='%d', fontsize=8, colors=th_c)
        _draw_sfc_barbs(ax, cfg, ds_active, _crop, lons_c, lats_c, skip_barb)

    elif cfg['plot_type'] == 'sfc_composite':
        t2m_f, d2m_f, mslp = data
        u_val, v_val = wind_data
        sig = _sigma_pts(cfg, wfo)
        t_s = _smooth(_crop(t2m_f), sig)[::skip_fill, ::skip_fill]
        td_s = _smooth(_crop(d2m_f), sig)[::skip_fill, ::skip_fill]
        p_s = _smooth(_crop(mslp), sig * 1.5)[::skip_fill, ::skip_fill]

        # 1. Hatched dewpoint shading >= 56 F, with SPC-style legend
        _hatched_bands(ax, x_sub, y_sub, td_s, cfg['dwpt_fill_levels'], cfg['dwpt_fill_colors'], zorder=1.5)
        _band_legend(fig, ax, cfg['dwpt_fill_levels'], cfg['dwpt_fill_colors'], 'Dewpoint (°F)')

        # 2. Dewpoint: blue dashed every 4 F
        cs_td = ax.contour(x_sub, y_sub, td_s, levels=cfg['dwpt_levels'], colors=cfg['dwpt_color'],
                           linestyles='dashed', linewidths=cfg.get('dwpt_linewidth', 1.5), zorder=2.0)
        ax.clabel(cs_td, inline=True, fmt='%d', fontsize=8, colors=cfg['dwpt_color'])

        # 3. Temperature: solid every 5 F, colored by range, plus a bold 32 F line
        cold_max, mild_max = cfg['temp_cold_max'], cfg['temp_mild_max']
        c_cold, c_mild, c_warm = cfg['temp_colors']
        t_levels = list(cfg['temp_levels'])
        t_colors = [c_cold if t <= cold_max else (c_mild if t <= mild_max else c_warm) for t in t_levels]
        cs_t = ax.contour(x_sub, y_sub, t_s, levels=t_levels, colors=t_colors, linewidths=cfg.get('temp_linewidth', 1.0), zorder=2.1)
        ax.clabel(cs_t, inline=True, fmt='%d', fontsize=8)
        cs_frz = ax.contour(x_sub, y_sub, t_s, levels=[32], colors=cfg['freeze_color'], linewidths=cfg.get('freeze_linewidth', 2.6), zorder=2.2)
        ax.clabel(cs_frz, inline=True, fmt='%d', fontsize=8.5)

        # 4. MSLP: black solid every 4 hPa
        cs_mslp = ax.contour(x_sub, y_sub, p_s, levels=cfg['mslp_levels'], colors='black', linewidths=ISOBAR_WIDTH, zorder=ISOBAR_ZORDER)
        ax.clabel(cs_mslp, inline=True, fmt='%d', fontsize=9, colors='black')

        # 5. 10 m wind barbs
        u_c, v_c = _crop(u_val), _crop(v_val)
        ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
                 u_c[::skip_barb, ::skip_barb], v_c[::skip_barb, ::skip_barb],
                 length=5, color=cfg['barb_color'], linewidth=0.6, transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'theta_e_composite':
        theta_e, mslp = data
        u_val, v_val = wind_data
        sig = _sigma_pts(cfg, wfo)
        te_s = _smooth(_crop(theta_e), sig)[::skip_fill, ::skip_fill]
        p_s = _smooth(_crop(mslp), sig * 1.5)[::skip_fill, ::skip_fill]

        # 1. Theta-E: dashed every 2 K, colored by value like SPC (no fill)
        te_levels = list(cfg['te_levels'])
        def _te_color(v):
            for upper, col in cfg['te_color_breaks']:
                if v <= upper:
                    return col
        te_colors = [_te_color(v) for v in te_levels]
        cs_te = ax.contour(x_sub, y_sub, te_s, levels=te_levels, colors=te_colors,
                           linestyles='dashed', linewidths=1.1, zorder=2.0)
        ax.clabel(cs_te, inline=True, fmt='%d', fontsize=8)

        # 2. MSLP: black solid every 2 hPa
        cs_mslp = ax.contour(x_sub, y_sub, p_s, levels=cfg['mslp_levels'], colors='black', linewidths=ISOBAR_WIDTH, zorder=ISOBAR_ZORDER)
        ax.clabel(cs_mslp, inline=True, fmt='%d', fontsize=9, colors='black')

        # 3. 10 m wind barbs
        u_c, v_c = _crop(u_val), _crop(v_val)
        ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
                 u_c[::skip_barb, ::skip_barb], v_c[::skip_barb, ::skip_barb],
                 length=5, color=cfg['barb_color'], linewidth=0.6, transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'moisture_convergence':
        mr, mconv = data
        u_val, v_val = wind_data
        if cfg.get('mr_fill', True):
            cf_mr = ax.contourf(x_sub, y_sub, _crop(mr)[::skip_fill, ::skip_fill], levels=cfg['mr_levels'],
                                colors=cfg['mr_colors'], alpha=0.6, zorder=1.5)
            fig.colorbar(cf_mr, ax=ax, orientation='horizontal', pad=0.04, shrink=0.72, label='Mixing Ratio (g/kg)')
        if cfg.get('mr_contour_levels') is not None:
            cs_mrl = ax.contour(x_sub, y_sub, _smooth(_crop(mr), 1.5)[::skip_fill, ::skip_fill], levels=cfg['mr_contour_levels'],
                                colors='black', linewidths=0.7, zorder=1.8)
            ax.clabel(cs_mrl, inline=True, fmt='%d', fontsize=7.5, colors='black')
        mc_c = cfg.get('mconv_color', 'darkblue')
        cs_mc = ax.contour(x_sub, y_sub, _crop(mconv)[::skip_fill, ::skip_fill], levels=cfg['mconv_levels'],
                           colors=mc_c, linewidths=1.5, zorder=2)
        ax.clabel(cs_mc, inline=True, fmt='%d', fontsize=9, colors=mc_c)
        u_c, v_c = _crop(u_val), _crop(v_val)
        ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
                 u_c[::skip_barb, ::skip_barb], v_c[::skip_barb, ::skip_barb],
                 length=5, color=cfg.get('barb_color', 'black'), linewidth=0.6, transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'custom_contour':
        data_values = _crop(data)
        sigma_val = cfg.get('smoothing', 0)
        if sigma_val > 0:
            nan_mask = np.isnan(data_values)
            if np.any(nan_mask):
                data_values = ndimage.gaussian_filter(np.where(nan_mask, 0.0, data_values), sigma=sigma_val) / np.maximum(
                    ndimage.gaussian_filter(np.where(nan_mask, 0.0, 1.0), sigma=sigma_val), 1e-8)
                data_values[nan_mask] = np.nan
            else:
                data_values = ndimage.gaussian_filter(data_values, sigma=sigma_val)
        if cfg.get('hatch_levels'):
            _hatched_bands(ax, x_sub, y_sub, data_values[::skip_fill, ::skip_fill],
                           cfg['hatch_levels'], cfg['hatch_colors'], zorder=1.5)
            if cfg.get('hatch_legend'):
                _band_legend(fig, ax, cfg['hatch_levels'], cfg['hatch_colors'], cfg['hatch_legend'])
        cs = ax.contour(x_sub, y_sub, data_values[::skip_fill, ::skip_fill], levels=cfg['levels'],
                        colors=cfg['colors'], linestyles=cfg.get('linestyles', 'solid'),
                        linewidths=cfg.get('linewidths', 1.4), zorder=2)
        ax.clabel(cs, inline=True, fmt=cfg.get('label_fmt', '%g'), fontsize=9)

    elif cfg['plot_type'] == 'shear':
        mag_kt = _crop(data)
        u_kt, v_kt = _crop(wind_data[0]), _crop(wind_data[1])
        sigma_val = cfg.get('smoothing', 0)
        if sigma_val > 0:
            nan_mask = np.isnan(mag_kt)
            if np.any(nan_mask):
                if 'Effective' in parameter:
                    mag_kt = ndimage.gaussian_filter(np.where(nan_mask, 0.0, mag_kt), sigma=sigma_val)
                else:
                    mag_kt = ndimage.gaussian_filter(np.where(nan_mask, 0.0, mag_kt), sigma=sigma_val) / np.maximum(
                        ndimage.gaussian_filter(np.where(nan_mask, 0.0, 1.0), sigma=sigma_val), 1e-8)
                    mag_kt[nan_mask] = np.nan
        cs = ax.contour(x_sub, y_sub, mag_kt[::skip_fill, ::skip_fill], levels=cfg['levels'],
                        colors=cfg['colors'], linestyles=cfg.get('linestyles', 'solid'),
                        linewidths=cfg.get('linewidths', 1.4), zorder=2)
        ax.clabel(cs, inline=True, fmt='%d', fontsize=8.5)
        ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
                 u_kt[::skip_barb, ::skip_barb], v_kt[::skip_barb, ::skip_barb],
                 length=5, color=cfg.get('barb_color', 'peru'), linewidth=0.65, transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'esrh_composite':
        srh, eil_base = data
        u_kt, v_kt = _crop(wind_data[0]), _crop(wind_data[1])
        srh_vals = _crop(srh)
        sigma_val = cfg.get('smoothing', 0)
        if sigma_val > 0:
            nan_mask = np.isnan(srh_vals)
            if np.any(nan_mask):
                if 'Effective' in parameter:
                    srh_vals = ndimage.gaussian_filter(np.where(nan_mask, 0.0, srh_vals), sigma=sigma_val)
                else:
                    srh_vals = ndimage.gaussian_filter(np.where(nan_mask, 0.0, srh_vals), sigma=sigma_val) / np.maximum(
                        ndimage.gaussian_filter(np.where(nan_mask, 0.0, 1.0), sigma=sigma_val), 1e-8)
                    srh_vals[nan_mask] = np.nan
            else:
                srh_vals = ndimage.gaussian_filter(srh_vals, sigma=sigma_val)

        if cfg.get('plot_eil', True) and eil_base is not None:
            eil_vals = _crop(eil_base)
            if cfg.get('eil_hatch'):
                _hatched_bands(ax, x_sub, y_sub, eil_vals[::skip_fill, ::skip_fill],
                               cfg['eil_levels'], cfg['eil_colors'], zorder=1.5)
                _band_legend(fig, ax, cfg['eil_levels'], cfg['eil_colors'], 'Effective Inflow Base (m AGL)')
            else:
                cf_eil = ax.contourf(x_sub, y_sub, eil_vals[::skip_fill, ::skip_fill],
                                     levels=cfg['eil_levels'], colors=cfg['eil_colors'], alpha=0.38, zorder=1.5)
                cbar = fig.colorbar(cf_eil, ax=ax, orientation='horizontal', pad=0.04, shrink=0.55,
                                    ticks=cfg.get('eil_ticks', [10, 250, 1000]),
                                    label='Effective Inflow Base (m AGL)')
                cbar.ax.set_xticklabels(cfg.get('eil_ticklabels', ['10', '250', '1000+']))

        cs_srh = ax.contour(x_sub, y_sub, srh_vals[::skip_fill, ::skip_fill], levels=cfg['srh_levels'],
                            colors=cfg['srh_colors'], linestyles=cfg.get('srh_linestyles', 'solid'),
                            linewidths=cfg.get('srh_linewidths', 1.4), zorder=2)
        ax.clabel(cs_srh, inline=True, fmt='%d', fontsize=8.5)
        ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
                 u_kt[::skip_barb, ::skip_barb], v_kt[::skip_barb, ::skip_barb],
                 length=5, color=cfg.get('barb_color', 'peru'), linewidth=0.65, transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'composite_index':
        idx_val, cin_val = data
        idx_vals = _crop(idx_val)
        sigma_val = cfg.get('smoothing', 0)
        if sigma_val > 0:
            nan_mask = np.isnan(idx_vals)
            if np.any(nan_mask):
                idx_vals = ndimage.gaussian_filter(np.where(nan_mask, 0.0, idx_vals), sigma=sigma_val) / np.maximum(
                    ndimage.gaussian_filter(np.where(nan_mask, 0.0, 1.0), sigma=sigma_val), 1e-8)
                idx_vals[nan_mask] = np.nan
            else:
                idx_vals = ndimage.gaussian_filter(idx_vals, sigma=sigma_val)
        idx_sub = idx_vals[::skip_fill, ::skip_fill]

        if cin_val is not None:
            cin_vals = _crop(cin_val)
            if cfg.get('cin_hatch'):
                _hatched_bands(ax, x_sub, y_sub, cin_vals[::skip_fill, ::skip_fill],
                               cfg['cin_levels'], cfg['cin_colors'], zorder=1.5)
                _band_legend(fig, ax, cfg['cin_levels'], cfg['cin_colors'], cfg.get('cin_label', 'CIN (J/kg)'))
            else:
                cf_cin = ax.contourf(x_sub, y_sub, cin_vals[::skip_fill, ::skip_fill],
                                     levels=cfg['cin_levels'], colors=cfg['cin_colors'],
                                     alpha=cfg.get('cin_alpha', 0.42), zorder=1.5)
                cbar = fig.colorbar(cf_cin, ax=ax, orientation='horizontal', pad=0.04, shrink=0.58,
                                    ticks=cfg['cin_ticks'], label=cfg.get('cin_label', 'CIN (J/kg)'))
                cbar.ax.set_xticklabels(cfg['cin_ticklabels'])

        if cfg.get('show_max_h', False):
            local_max = (idx_sub == ndimage.maximum_filter(idx_sub, size=18)) & (idx_sub >= 1.5)
            my, mx = np.where(local_max)
            for r_i, c_i in zip(my, mx):
                lo_v, la_v = lons_sub[r_i, c_i], lats_sub[r_i, c_i]
                if lon_min <= lo_v <= lon_max and lat_min <= la_v <= lat_max:
                    ax.text(lo_v, la_v, f"H\n{idx_sub[r_i, c_i]:.1f}", color='white',
                            fontsize=7.5, fontweight='bold', ha='center', va='center',
                            transform=ccrs.PlateCarree(), zorder=4)

        if not cfg.get('no_contours', False):
            cs_idx = ax.contour(x_sub, y_sub, idx_sub, levels=cfg['levels'],
                                colors=cfg['colors'], linestyles=cfg.get('linestyles', 'solid'),
                                linewidths=cfg.get('linewidths', 1.4), zorder=2)
            ax.clabel(cs_idx, inline=True, fmt='%g', fontsize=8.5)

        if wind_data is not None:
            u_kt, v_kt = _crop(wind_data[0]), _crop(wind_data[1])
            ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
                     u_kt[::skip_barb, ::skip_barb], v_kt[::skip_barb, ::skip_barb],
                     length=5, color=cfg.get('barb_color', 'peru'), linewidth=0.65, transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'vort_mlcape3':
        vort_1e5, cape_3km = data
        u_kt, v_kt = _crop(wind_data[0]), _crop(wind_data[1])
        vort_sub = ndimage.gaussian_filter(np.nan_to_num(_crop(vort_1e5), nan=0.0), sigma=cfg.get('vort_smoothing', 4.0))[::skip_fill, ::skip_fill]
        cape_sub = ndimage.gaussian_filter(np.nan_to_num(_crop(cape_3km), nan=0.0), sigma=cfg.get('cape_smoothing', 1.5))[::skip_fill, ::skip_fill]
        cs_vort = ax.contour(x_sub, y_sub, vort_sub, levels=cfg['vort_levels'], colors=cfg.get('vort_color', '#0099ff'), linewidths=1.0, zorder=2)
        ax.clabel(cs_vort, inline=True, fmt='%d', fontsize=8)
        cs_cape3 = ax.contour(x_sub, y_sub, cape_sub, levels=cfg['cape3_levels'],
                              colors=cfg['cape3_colors'], linewidths=cfg['cape3_linewidths'], zorder=2.2)
        ax.clabel(cs_cape3, inline=True, fmt='%d', fontsize=8.5)
        ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
                 u_kt[::skip_barb, ::skip_barb], v_kt[::skip_barb, ::skip_barb],
                 length=5, color=cfg.get('barb_color', 'peru'), linewidth=0.65, transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'moisture_field':
        field_vals = _crop(data)
        sigma_val = cfg.get('smoothing', 0)
        if sigma_val > 0:
            field_vals = ndimage.gaussian_filter(np.nan_to_num(field_vals, nan=0.0), sigma=sigma_val)
        field_sub = field_vals[::skip_fill, ::skip_fill]
        if cfg.get('fill_hatch'):
            _hatched_bands(ax, x_sub, y_sub, field_sub, cfg['fill_levels'], cfg['fill_colors'], zorder=1.5)
            _band_legend(fig, ax, cfg['fill_levels'], cfg['fill_colors'], cfg['fill_label'])
        else:
            cf_m = ax.contourf(x_sub, y_sub, field_sub, levels=cfg['fill_levels'],
                               colors=cfg['fill_colors'], alpha=cfg.get('fill_alpha', 0.55), zorder=1.5)
            cbar = fig.colorbar(cf_m, ax=ax, orientation='horizontal', pad=0.04, shrink=0.58,
                                ticks=cfg['fill_ticks'], label=cfg['fill_label'])
            cbar.ax.set_xticklabels(cfg['fill_ticklabels'])
        cs_m = ax.contour(x_sub, y_sub, field_sub, levels=cfg['levels'],
                          colors=cfg.get('contour_color', 'black'), linewidths=cfg.get('contour_width', 1.0), zorder=2)
        ax.clabel(cs_m, inline=True, fmt='%g', fontsize=8.5)

        vec_mode = cfg.get('vector_mode')
        if vec_mode == 'quiver_850_mtrans' and wind_data is not None:
            uq, vq = _crop(wind_data[0]), _crop(wind_data[1])
            skip_q = 45 if wfo == 'Full CONUS' else max(6, skip_barb)
            ax.quiver(lons_c[::skip_q, ::skip_q], lats_c[::skip_q, ::skip_q],
                      uq[::skip_q, ::skip_q], vq[::skip_q, ::skip_q],
                      color=cfg.get('vector_color', '#8b4513'), scale=3200, width=0.0028, headwidth=4.0, headlength=4.5,
                      transform=ccrs.PlateCarree(), zorder=3)
        elif vec_mode == 'streamlines_sfc' and wind_data is not None:
            u_c = np.nan_to_num(_crop(wind_data[0]), nan=0.0)
            v_c = np.nan_to_num(_crop(wind_data[1]), nan=0.0)
            u_rot, v_rot = PROJECTION.transform_vectors(ccrs.PlateCarree(), lons_c, lats_c, u_c, v_c)
            x_1d = np.linspace(x_c.min(), x_c.max(), 110)
            y_1d = np.linspace(y_c.min(), y_c.max(), 110)
            x_reg, y_reg = np.meshgrid(x_1d, y_1d)
            tree = cKDTree(np.column_stack([x_c.ravel(), y_c.ravel()]))
            _, idx = tree.query(np.column_stack([x_reg.ravel(), y_reg.ravel()]), k=1)
            ax.streamplot(x_1d, y_1d, u_rot.ravel()[idx].reshape(x_reg.shape),
                          v_rot.ravel()[idx].reshape(x_reg.shape),
                          density=1.6, color=cfg.get('vector_color', '#8b4513'), linewidth=0.85, arrowsize=1.0, zorder=3)

    elif cfg['plot_type'] == 'winter_ptype':
        ptype = _crop(data)
        step_txt = 28 if wfo == 'Full CONUS' else max(5, int(min(ptype.shape) / 18))
        color_map = cfg.get('ptype_colors') or {
            0: '#888888', 1: '#cc0000', 2: '#cc0000', 3: '#cc0000',
            4: '#9900cc', 5: '#9900cc', 6: '#9900cc',
            7: '#00aadd', 8: '#0066cc', 9: '#0044aa', 10: '#555555'
        }
        for r_i in range(0, ptype.shape[0], step_txt):
            for c_i in range(0, ptype.shape[1], step_txt):
                code = int(ptype[r_i, c_i])
                if 0 <= code <= 10:
                    lo_v, la_v = lons_c[r_i, c_i], lats_c[r_i, c_i]
                    if lon_min <= lo_v <= lon_max and lat_min <= la_v <= lat_max:
                        ax.text(lo_v, la_v, str(code), color=color_map.get(code, '#555555'),
                                fontsize=cfg.get('ptype_fontsize', 8.0),
                                fontweight='normal' if cfg.get('ptype_colors') or code == 0 else 'bold',
                                ha='center', va='center', transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'les1':
        bl_rh, dt_sfc_850, sbcape = data
        sigma_val = cfg.get('smoothing', 1.5)
        rh_s = ndimage.gaussian_filter(np.nan_to_num(_crop(bl_rh), nan=0.0), sigma=sigma_val)[::skip_fill, ::skip_fill]
        dt_s = ndimage.gaussian_filter(np.nan_to_num(_crop(dt_sfc_850), nan=0.0), sigma=sigma_val)[::skip_fill, ::skip_fill]
        cape_s = ndimage.gaussian_filter(np.nan_to_num(_crop(sbcape), nan=0.0), sigma=sigma_val)[::skip_fill, ::skip_fill]
        cf_rh = ax.contourf(x_sub, y_sub, rh_s, levels=cfg['rh_fill_levels'],
                            colors=cfg['rh_fill_colors'], alpha=0.55, zorder=1.5)
        cbar = fig.colorbar(cf_rh, ax=ax, orientation='horizontal', pad=0.04, shrink=0.55,
                            ticks=[70, 80, 90], label='Boundary-Layer RH (%)')
        cbar.ax.set_xticklabels(['70', '80', '90'])
        cs_rh = ax.contour(x_sub, y_sub, rh_s, levels=cfg['rh_levels'],
                           colors='black', linestyles='dotted', linewidths=0.8, zorder=2)
        ax.clabel(cs_rh, inline=True, fmt='%d', fontsize=7.5)
        cs_dt = ax.contour(x_sub, y_sub, dt_s, levels=cfg['dt_levels'], colors=cfg.get('dt_color', '#ff7700'), linewidths=1.4, zorder=2.2)
        ax.clabel(cs_dt, inline=True, fmt='%d', fontsize=8.5)
        cs_cape = ax.contour(x_sub, y_sub, cape_s, levels=cfg['cape_levels'], colors=cfg.get('cape_color', 'red'), linewidths=1.1, zorder=2.4)
        ax.clabel(cs_cape, inline=True, fmt='%d', fontsize=8.5)

    elif cfg['plot_type'] == 'les2':
        div_1e5 = _crop(data)
        u_bl, v_bl = _crop(wind_data[0]), _crop(wind_data[1])
        div_sub = ndimage.gaussian_filter(np.nan_to_num(div_1e5, nan=0.0), sigma=cfg.get('smoothing', 2.5))[::skip_fill, ::skip_fill]
        cs_conv = ax.contour(x_sub, y_sub, div_sub, levels=cfg['conv_levels'], colors=cfg.get('conv_color', 'red'), linewidths=1.2, zorder=2.2)
        ax.clabel(cs_conv, inline=True, fmt='%d', fontsize=8.5)
        ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
                 (u_bl * 1.94384)[::skip_barb, ::skip_barb], (v_bl * 1.94384)[::skip_barb, ::skip_barb],
                 length=4.8, color=cfg.get('barb_color', 'peru'), linewidth=0.6, transform=ccrs.PlateCarree(), zorder=2.5)
        u_c, v_c = np.nan_to_num(u_bl, nan=0.0), np.nan_to_num(v_bl, nan=0.0)
        u_rot, v_rot = PROJECTION.transform_vectors(ccrs.PlateCarree(), lons_c, lats_c, u_c, v_c)
        x_1d = np.linspace(x_c.min(), x_c.max(), 110)
        y_1d = np.linspace(y_c.min(), y_c.max(), 110)
        x_reg, y_reg = np.meshgrid(x_1d, y_1d)
        tree = cKDTree(np.column_stack([x_c.ravel(), y_c.ravel()]))
        _, idx = tree.query(np.column_stack([x_reg.ravel(), y_reg.ravel()]), k=1)
        ax.streamplot(x_1d, y_1d, u_rot.ravel()[idx].reshape(x_reg.shape),
                      v_rot.ravel()[idx].reshape(x_reg.shape),
                      density=1.6, color='black', linewidth=0.85, arrowsize=1.0, zorder=3)

    elif cfg['plot_type'] == 'dgz_composite':
        dgz_depth, dgz_rh = data
        sigma_val = cfg.get('smoothing', 1.5)
        depth_s = ndimage.gaussian_filter(np.nan_to_num(_crop(dgz_depth), nan=0.0), sigma=sigma_val)[::skip_fill, ::skip_fill]
        rh_s = ndimage.gaussian_filter(np.nan_to_num(_crop(dgz_rh), nan=0.0), sigma=sigma_val)[::skip_fill, ::skip_fill]
        if cfg.get('rh_fill_hatch'):
            _hatched_bands(ax, x_sub, y_sub, rh_s, [90, 105], [cfg.get('rh_fill_color', '#048a04')], zorder=1.5)
        else:
            ax.contourf(x_sub, y_sub, rh_s, levels=[90, 105], colors=['#00cc00'], alpha=0.35, zorder=1.5)
        cs_rh = ax.contour(x_sub, y_sub, rh_s, levels=cfg['rh_levels'],
                           colors=cfg['rh_colors'], linewidths=cfg['rh_linewidths'], zorder=2)
        ax.clabel(cs_rh, inline=True, fmt='%d', fontsize=8.0)
        cs_depth = ax.contour(x_sub, y_sub, depth_s, levels=cfg['depth_levels'],
                              colors=cfg['depth_colors'], linewidths=cfg['depth_linewidths'], zorder=2.3)
        ax.clabel(cs_depth, inline=True, fmt='%d', fontsize=8.5)

    elif cfg['plot_type'] == 'contour':
        start, stop, step = cfg['contour_range']
        clevs = np.arange(start, stop, step)
        cs = ax.contour(x_sub, y_sub, _crop(data)[::skip_fill, ::skip_fill], levels=clevs,
                        colors=cfg['color'], linewidths=cfg.get('linewidth', 1.5), zorder=ISOBAR_ZORDER)
        ax.clabel(cs, inline=True, fmt='%d', fontsize=9)
        _draw_sfc_barbs(ax, cfg, ds_active, _crop, lons_c, lats_c, skip_barb)

    elif cfg['plot_type'] == 'filled':
        start, stop, step = cfg['contour_range']
        clevs = np.arange(start, stop, step)
        data_sub = _crop(data)[::skip_fill, ::skip_fill]
        cf = ax.contourf(x_sub, y_sub, data_sub, levels=clevs, cmap=cfg['cmap'], alpha=0.7, zorder=2)
        fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.04, shrink=0.72, label=cfg['title'])
        cs = ax.contour(x_sub, y_sub, data_sub, levels=clevs, colors='black', linewidths=0.3, alpha=0.5, zorder=2)
        ax.clabel(cs, inline=True, fmt='%d', fontsize=8, colors='black')
        if wind_data:
            u_c, v_c = _crop(wind_data[0]), _crop(wind_data[1])
            sb = 15 if wfo == 'Full CONUS' else max(4, skip_barb // 2)
            ax.barbs(lons_c[::sb, ::sb], lats_c[::sb, ::sb], u_c[::sb, ::sb], v_c[::sb, ::sb],
                     length=5, color='black', linewidth=0.6, transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'upper_air':
        u_c, v_c = _crop(wind_data[0]), _crop(wind_data[1])
        if cfg.get('plot_rh') and 'mean_rh' in extra_fields:
            rh_sub = _crop(extra_fields['mean_rh'])[::skip_fill, ::skip_fill]
            if cfg.get('rh_hatch'):
                _hatched_bands(ax, x_sub, y_sub, rh_sub, [70, 101], ['#048a04'], zorder=1.5)
                _band_legend(fig, ax, [70, 101], ['#048a04'], '700-500mb Mean RH (%)')
                cs_rh = ax.contour(x_sub, y_sub, rh_sub, levels=[70, 80, 90], colors='#04ce04', linewidths=1.2, zorder=2.2)
                ax.clabel(cs_rh, inline=True, fmt='%d', fontsize=8, colors='#04ce04')
            else:
                ax.contourf(x_sub, y_sub, rh_sub, levels=[70, 100], colors=['#a1d99b'], alpha=0.5, zorder=2)
                cs_rh = ax.contour(x_sub, y_sub, rh_sub, levels=[70], colors='darkgreen', linewidths=1.2, zorder=2.2)
                ax.clabel(cs_rh, inline=True, fmt='70%% RH', fontsize=8, colors='darkgreen')
        elif cfg.get('isotach_levels'):
            # SPC style: hatched isotach bands + legend, thin navy outlines
            speed_sub = _crop(data)[::skip_fill, ::skip_fill]
            _hatched_bands(ax, x_sub, y_sub, speed_sub, cfg['isotach_levels'], cfg['isotach_colors'], zorder=1.5,
                           tint_alpha=cfg.get('isotach_tint', 0.0))
            _band_legend(fig, ax, cfg['isotach_levels'], cfg['isotach_colors'], 'Isotachs (kt)')
            if cfg.get('isotach_outline', '#144e8c'):
                ax.contour(x_sub, y_sub, speed_sub, levels=cfg['isotach_levels'][:-1],
                           colors=cfg.get('isotach_outline', '#144e8c'), linewidths=0.6, zorder=1.6)
        elif cfg.get('speed_fill', True):
            speed_sub = _crop(data)[::skip_fill, ::skip_fill]
            start, stop, step = cfg['contour_range']
            clevs = np.arange(start, stop, step)
            cf = ax.contourf(x_sub, y_sub, speed_sub, levels=clevs, cmap=cfg['cmap'], alpha=0.7, zorder=2)
            fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.04, shrink=0.72, label=cfg['title'])
            cs = ax.contour(x_sub, y_sub, speed_sub, levels=clevs, colors='black', linewidths=0.3, alpha=0.5, zorder=2)
            ax.clabel(cs, inline=True, fmt='%d', fontsize=8, colors='black')

        if 'height_levels' in cfg and 'heights_dam' in extra_fields:
            hght_sub = ndimage.gaussian_filter(_crop(extra_fields['heights_dam']), sigma=3)[::skip_fill, ::skip_fill]
            cs_h = ax.contour(x_sub, y_sub, hght_sub, levels=cfg['height_levels'], colors='black', linewidths=HEIGHT_LINE_WIDTH, zorder=ISOBAR_ZORDER)
            ax.clabel(cs_h, inline=True, fmt='%d', fontsize=9, colors='black')

        if 'temp_levels' in cfg and 'temp_c' in extra_fields:
            temp_sub = ndimage.gaussian_filter(_crop(extra_fields['temp_c']), sigma=3)[::skip_fill, ::skip_fill]
            t_levels = cfg['temp_levels']
            cold_c, warm_c = cfg.get('temp_cold_color', 'blue'), cfg.get('temp_warm_color', 'red')
            t_colors = [cold_c if t <= 0 else warm_c for t in t_levels] if cfg.get('temp_dynamic_color') else warm_c
            cs_t = ax.contour(x_sub, y_sub, temp_sub, levels=t_levels, colors=t_colors, linestyles='dashed', linewidths=1.0, zorder=2.5)
            ax.clabel(cs_t, inline=True, fmt='%d', fontsize=8)

        if cfg.get('dwpt_levels') is not None and 'dwpt_c' in extra_fields:
            td_sub = ndimage.gaussian_filter(_crop(extra_fields['dwpt_c']), sigma=3)[::skip_fill, ::skip_fill]
            cs_td = ax.contour(x_sub, y_sub, td_sub, levels=cfg['dwpt_levels'], colors=cfg.get('dwpt_color', '#048a04'),
                               linewidths=1.0, zorder=2.5)
            ax.clabel(cs_td, inline=True, fmt='%d', fontsize=8)

        if 'plot_divergence' in cfg and 'divergence' in extra_fields:
            div_sub = _crop(extra_fields['divergence'])[::skip_fill, ::skip_fill]
            div_c = cfg.get('div_color', 'magenta')
            cs_d = ax.contour(x_sub, y_sub, div_sub, levels=cfg['div_levels'], colors=div_c, linewidths=1.5, zorder=2.5)
            ax.clabel(cs_d, inline=True, fmt='%d', fontsize=9, colors=div_c)

        ax.barbs(lons_c[::skip_barb, ::skip_barb], lats_c[::skip_barb, ::skip_barb],
                 u_c[::skip_barb, ::skip_barb], v_c[::skip_barb, ::skip_barb],
                 length=5, color=cfg.get('barb_color', 'black'), linewidth=0.6, transform=ccrs.PlateCarree(), zorder=3)

    elif cfg['plot_type'] == 'cape_cin':
        cape, cin = data
        sigma_val = cfg.get('smoothing', 0)
        cape_clean = np.nan_to_num(_crop(cape)[::skip_fill, ::skip_fill], nan=0.0)
        cin_clean = np.nan_to_num(_crop(cin)[::skip_fill, ::skip_fill], nan=0.0)
        cape_plot = ndimage.gaussian_filter(cape_clean, sigma=sigma_val) if sigma_val > 0 else cape_clean
        cin_plot = ndimage.gaussian_filter(cin_clean, sigma=sigma_val) if sigma_val > 0 else cin_clean

        if cfg.get('cin_hatch'):
            _hatched_bands(ax, x_sub, y_sub, cin_plot, cfg['cin_levels'], cfg['cin_colors'], zorder=1.5)
            _band_legend(fig, ax, cfg['cin_levels'], cfg['cin_colors'], 'CIN (J/kg)')
        else:
            cf_cin = ax.contourf(x_sub, y_sub, cin_plot, levels=cfg['cin_levels'],
                                 colors=cfg['cin_colors'], alpha=0.5, zorder=1.5)
            cbar = fig.colorbar(cf_cin, ax=ax, orientation='horizontal', pad=0.04, shrink=0.55,
                                ticks=[25, 100], label='CIN (J/kg)')
            cbar.ax.set_xticklabels(['25', '100'])
        c_levels = cfg['cape_levels']
        c_colors = ['saddlebrown' if val < 800 else 'red' for val in c_levels] if cfg.get('cape_dynamic_colors') else cfg['cape_colors']
        cs_cape = ax.contour(x_sub, y_sub, cape_plot, levels=c_levels, colors=c_colors,
                             linewidths=cfg.get('cape_linewidths', 1.2), zorder=2)
        ax.clabel(cs_cape, inline=True, fmt='%d', fontsize=9)

    if 'time' in ds_active.coords:
        run_time = pd.to_datetime(ds_active.coords['time'].values).strftime('%Y-%m-%d %H:%M UTC')
    elif 'valid_time' in ds_active.coords:
        run_time = pd.to_datetime(ds_active.coords['valid_time'].values).strftime('%Y-%m-%d %H:%M UTC')
    else:
        run_time = 'N/A'

    ax.set_title(
        f"SPC Mesoanalysis: {cfg['title']}\nDomain: {wfo}  |  Run Time: {run_time}",
        fontsize=11.5, fontweight='bold', pad=10
    )

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.18, dpi=RENDER_DPI)
    fig.clear()
    return buf.getvalue()


# MEMORY: only MAX_CONCURRENT_RENDERS new maps are drawn at once; other requests wait their turn
# instead of piling up in memory. Maps already drawn come straight from the cache and never wait.
MAX_CONCURRENT_RENDERS = 1


def _release_freed_memory():
    """Return freed memory to the operating system. Python and NumPy keep freed blocks by default,
    so memory use only ratchets upward after each big map, and the host counts that against the limit."""
    try:
        import ctypes
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


@st.cache_data(max_entries=64, show_spinner=False)
def render_map_png(file_path, parameter, wfo='Full CONUS'):
    with _shared_state()['render_slots']:
        try:
            return _draw_map_png(file_path, parameter, wfo)
        finally:
            _release_freed_memory()


# --- LIVE GREEN STATUS BANNER ---
def render_green_status_banner(file_basename, hour_offset, checked_epoch_ms, checked_utc_str):
    (st.iframe if hasattr(st, 'iframe') else components.html)(
        f"""
        <div style="
            background: linear-gradient(90deg, #14532d 0%, #166534 100%);
            border: 1px solid #22c55e;
            border-radius: 8px;
            padding: 10px 16px;
            color: #f0fdf4;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            font-size: 13.5px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        ">
            <div>
                <span style="display:inline-block; width:10px; height:10px; background:#4ade80; border-radius:50%; margin-right:8px; box-shadow:0 0 6px #4ade80;"></span>
                <strong>AUTO-POLL ACTIVE</strong>  |
                Active File: <code style="background:rgba(0,0,0,0.3); padding:2px 6px; border-radius:4px; color:#bbf7d0;">{file_basename}</code>
                 (Offset: <strong>{hour_offset:+d}h</strong>)
            </div>
            <div style="margin-top: 2px;">
                Last Check: <strong>{checked_utc_str}</strong>
                (<span id="elapsed-txt">0m 00s ago</span>)  |
                Next Check in: <strong id="countdown-txt" style="color:#86efac;">05:00</strong>
            </div>
        </div>
        <script>
            const checkedMs = {checked_epoch_ms};
            function updateTimers() {{
                const nowMs = Date.now();
                const elapsedSec = Math.max(0, Math.floor((nowMs - checkedMs) / 1000));
                const elM = Math.floor(elapsedSec / 60);
                const elS = elapsedSec % 60;
                const elSpan = document.getElementById('elapsed-txt');
                if (elSpan) elSpan.textContent = elM + 'm ' + String(elS).padStart(2, '0') + 's ago';

                const remSec = Math.max(0, 300 - (elapsedSec % 300));
                const rM = Math.floor(remSec / 60);
                const rS = remSec % 60;
                const cdSpan = document.getElementById('countdown-txt');
                if (cdSpan) cdSpan.textContent = String(rM).padStart(2, '0') + ':' + String(rS).padStart(2, '0');
            }}
            updateTimers();
            setInterval(updateTimers, 1000);
        </script>
        """,
        height=56,
    )


# --- TOP CONTROL BAR & TIMELINE SLIDER ---
MENU_TREE = {
    cat: [p for p in items if p in PARAMETERS]
    for cat, items in SPC_CATEGORIES.items()
}
MENU_TREE = {cat: items for cat, items in MENU_TREE.items() if items}

col1, col2, col3 = st.columns([1.1, 1.5, 1.4])
with col1:
    menu_sel = st.selectbox("SPC Menu", list(MENU_TREE.keys()), key="spc_menu_sel")
with col2:
    param_sel = st.selectbox("Parameter", MENU_TREE[menu_sel], key="spc_param_sel")
with col3:
    wfo_sel = st.selectbox("Sector / WFO", list(WFO_DOMAINS.keys()), key="spc_wfo_sel")


def _offset_label(off):
    """Slider labels show the real cycle time, e.g. '-3h · 16Z' or 'Latest · 19Z'."""
    try:
        runs, _, _ = list_available_runs()
        cyc = runs[min(-off, len(runs) - 1)][2]
        return f"Latest · {cyc}Z" if off == 0 else f"{off:+d}h · {cyc}Z"
    except Exception:
        return "Latest" if off == 0 else f"{off:+d}h"


s_col1, s_col2 = st.columns([3.2, 0.8])
with s_col1:
    # Past analyses only. SFCOA-HRRR has no forecast hours on NOMADS, so +1..+6 had nothing to load.
    hour_offset = st.select_slider(
        "Analysis time",
        options=list(range(-MAX_PAST_HOURS, 1)),
        value=0,
        format_func=_offset_label,
        key="spc_hour_slider",
        help="Previous hourly SFCOA-HRRR analyses. Older hours load in the background after the latest map appears."
    )
with s_col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Check NOMADS Now", width='stretch'):
        list_available_runs.clear()
        st.rerun()


def memory_readout():
    """Server memory now and peak since start (Linux), to see how close the app runs to its limit."""
    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB on Linux
        now_mb = None
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    now_mb = int(line.split()[1]) / 1024.0
                    break
        disk_mb = sum(os.path.getsize(p) for p in glob.glob("*_spc_post.t*z.sfcoa_hrrr.*.nc")) / 1e6
        txt = f"Server memory: {now_mb:,.0f} MB now · {peak_mb:,.0f} MB peak · data files on disk: {disk_mb:,.0f} MB"
        st.caption(txt)
    except Exception:
        pass


def show_map_fit_to_screen(png_bytes):
    """Fit the map to the browser window: never wider than the page, never taller than the
    visible screen below the controls, so the whole map is visible without scrolling."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center;">
          <img src="data:image/png;base64,{b64}" alt="SPC mesoanalysis map"
               style="max-width:100%; max-height:max(420px, calc(100vh - 290px));
                      width:auto; height:auto; object-fit:contain;">
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- AUTO-POLL FRAGMENT + MAP DISPLAY ---
def _render_dashboard():
    try:
        nc_file, offset_note, checked_epoch_ms, checked_utc_str = download_sfcoa(hour_offset)
        render_green_status_banner(os.path.basename(nc_file), hour_offset, checked_epoch_ms, checked_utc_str)
        if offset_note:
            st.info(offset_note)

        with st.spinner(f"Rendering {param_sel} ({wfo_sel})..."):
            png_bytes = render_map_png(nc_file, param_sel, wfo_sel)
        show_map_fit_to_screen(png_bytes)

        # Now that the user has a map, get the older hours ready in the background
        start_background_prefetch(param_sel, wfo_sel)
        memory_readout()

    except Exception as e:
        st.error(f"Error loading or rendering map: {e}")
        st.code(traceback.format_exc())


if hasattr(st, "fragment"):
    @st.fragment(run_every=300)
    def live_polling_section():
        _render_dashboard()
    live_polling_section()
else:
    _render_dashboard()
