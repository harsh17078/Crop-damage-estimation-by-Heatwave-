"""
make_district_centroids.py - build data/external/district_centroids.csv WITHOUT Google Earth Engine
==================================================================================================

B3 needs one (lat, lon) per district to read IMD's 1-degree Tmax grid. This script creates that file from the
district names already in your project (reports/exposure/exposure_table.csv), so the names match by construction.

TWO METHODS
  A) geocode (default)   looks up "<district> district, <state>, India" on OpenStreetMap (Nominatim).
                         needs internet + `pip install geopy`. Each result is checked against its state's bounding box.
  B) boundaries          --boundaries path/to/districts.geojson|.shp|.gpkg  (e.g. GADM India level 2, or any district
                         boundary file you already have). needs `pip install geopandas`. Exact geometric centroids;
                         names are matched to your project names (fuzzy, every non-exact match is printed for review).

Accuracy note: IMD Tmax is a 1-degree grid (~110 km). A district location that is off by 10-30 km (town vs. geometric
centroid) does not change which grid cells are used in any meaningful way.

RUN (VS Code terminal, project root)
  pip install pandas geopy
  python scripts/make_district_centroids.py                       # method A
  python scripts/make_district_centroids.py --boundaries data/external/gadm41_IND_2.json   # method B
Outputs
  data/external/district_centroids.csv          District, State, lat, lon   (what B3 reads)
  data/external/district_centroids_REVIEW.csv   every row with source, query / matched name and a flag
If some districts fail, the script lists them; add them by hand (Google Maps -> right-click -> copy lat, lon) to
data/external/district_centroids_manual.csv with columns District,lat,lon and run the script again: manual rows win.
"""

import argparse
import difflib
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, "data")) and os.path.abspath(os.getcwd()) != ROOT:
    os.chdir(ROOT)

TABLE = "reports/exposure/exposure_table.csv"
OUT = "data/external/district_centroids.csv"
REVIEW = "data/external/district_centroids_REVIEW.csv"
MANUAL = "data/external/district_centroids_manual.csv"
CACHE = "data/external/_geocode_cache.csv"
# (lat_min, lat_max, lon_min, lon_max) - generous boxes used only to reject obviously wrong geocoder hits
STATE_BOX = {"punjab": (29.5, 32.6, 73.8, 77.0), "haryana": (27.6, 31.0, 74.4, 77.7),
             "uttarpradesh": (23.8, 30.5, 77.0, 84.7)}


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def in_box(state, lat, lon):
    b = STATE_BOX.get(norm(state))
    return True if b is None else (b[0] <= lat <= b[1] and b[2] <= lon <= b[3])


def project_districts(table):
    if not os.path.exists(table):
        sys.exit(f"[ERROR] {table} not found. Run B1 first (it writes the table).")
    d = pd.read_csv(table)[["State", "District"]].drop_duplicates().sort_values(["State", "District"])
    return d.reset_index(drop=True)


# ----------------------------------------------------------------- method A: geocoding
def geocode_one(geocode, district, state):
    """returns dict(lat, lon, query, matched, flag). `geocode` is a callable(query) -> object with latitude/longitude."""
    last = None
    for q in (f"{district} district, {state}, India", f"{district}, {state}, India"):
        loc = geocode(q)
        if loc is None:
            continue
        lat, lon = float(loc.latitude), float(loc.longitude)
        last = {"lat": lat, "lon": lon, "query": q, "matched": getattr(loc, "address", "")}
        if in_box(state, lat, lon):
            last["flag"] = "ok"
            return last
        last["flag"] = "OUTSIDE_STATE_BOX"
    return last or {"lat": float("nan"), "lon": float("nan"), "query": "", "matched": "", "flag": "NOT_FOUND"}


def run_geocoding(dists):
    try:
        from geopy.extra.rate_limiter import RateLimiter
        from geopy.geocoders import Nominatim
    except ImportError:
        sys.exit("[ERROR] geopy missing. Run: pip install geopy")
    geo = Nominatim(user_agent="wheat-heat-stress-minor-project", timeout=15)
    rl = RateLimiter(geo.geocode, min_delay_seconds=1.2, max_retries=2, error_wait_seconds=5, swallow_exceptions=True)
    cache = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
    done = {(r.District, r.State): r._asdict() for r in cache.itertuples(index=False)} if len(cache) else {}
    rows = []
    for i, r in enumerate(dists.itertuples(index=False), 1):
        key = (r.District, r.State)
        if key in done and done[key].get("flag") == "ok":
            rows.append(done[key])
            continue
        print(f"  [{i}/{len(dists)}] geocoding {r.District}, {r.State} ...")
        g = geocode_one(rl, r.District, r.State)
        rows.append({"District": r.District, "State": r.State, "source": "geocode", **g})
        pd.DataFrame(rows).to_csv(CACHE, index=False)     # save progress
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- method B: boundary file
def run_boundaries(dists, path):
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("[ERROR] geopandas missing. Run: pip install geopandas   (or use the default geocoding method)")
    g = gpd.read_file(path)
    low = {c.lower(): c for c in g.columns}
    pick = lambda *n: next((low[x] for x in n if x in low), None)
    nc = pick("name_2", "adm2_name", "district", "dtname", "dist_name", "name")
    sc = pick("name_1", "adm1_name", "state", "stname", "st_nm", "state_name")
    if nc is None:
        sys.exit(f"[ERROR] cannot find a district-name column in {path}: {list(g.columns)}")
    print(f"boundary file: {len(g)} polygons | district column '{nc}' | state column '{sc}'")
    if g.crs is None:
        g = g.set_crs(4326)
    proj = g.to_crs("EPSG:7755")                      # India LCC (metres) so centroids are not distorted
    pts = gpd.GeoSeries(proj.geometry.centroid, crs="EPSG:7755").to_crs(4326)
    g = g.assign(_lat=pts.y.values, _lon=pts.x.values, _dn=g[nc].map(norm), _sn=g[sc].map(norm) if sc else "")
    rows = []
    for r in dists.itertuples(index=False):
        cand = g[g["_sn"] == norm(r.State)] if sc else g
        if cand.empty:
            cand = g
        exact = cand[cand["_dn"] == norm(r.District)]
        if len(exact):
            m, flag = exact.iloc[0], "ok"
        else:
            best = difflib.get_close_matches(norm(r.District), cand["_dn"].tolist(), n=1, cutoff=0.75)
            if best:
                m, flag = cand[cand["_dn"] == best[0]].iloc[0], "FUZZY_MATCH_CHECK"
            else:
                rows.append({"District": r.District, "State": r.State, "source": "boundaries", "lat": float("nan"),
                             "lon": float("nan"), "query": "", "matched": "", "flag": "NOT_FOUND"})
                continue
        lat, lon = float(m["_lat"]), float(m["_lon"])
        if flag == "ok" and not in_box(r.State, lat, lon):
            flag = "OUTSIDE_STATE_BOX"
        rows.append({"District": r.District, "State": r.State, "source": "boundaries", "lat": lat, "lon": lon,
                     "query": r.District, "matched": str(m[nc]), "flag": flag})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- common
def apply_manual(res):
    if not os.path.exists(MANUAL):
        return res
    man = pd.read_csv(MANUAL)
    low = {c.lower(): c for c in man.columns}
    if not {"district", "lat", "lon"} <= set(low):
        print(f"[WARNING] {MANUAL} needs columns District,lat,lon - ignored")
        return res
    n = 0
    for r in man.itertuples(index=False):
        d, la, lo = getattr(r, low["district"]), getattr(r, low["lat"]), getattr(r, low["lon"])
        idx = res.index[res["District"].map(norm) == norm(d)]
        for i in idx:
            res.loc[i, ["lat", "lon", "source", "flag"]] = [float(la), float(lo), "manual", "ok"]
            n += 1
    print(f"manual overrides applied: {n}")
    return res


def main():
    ap = argparse.ArgumentParser(description="Build data/external/district_centroids.csv without Earth Engine")
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--boundaries", default=None, help="district boundary file (geojson/shp/gpkg); default = geocoding")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dists = project_districts(a.table)
    print(f"{len(dists)} districts in the project table: {dists['State'].value_counts().to_dict()}")
    res = run_boundaries(dists, a.boundaries) if a.boundaries else run_geocoding(dists)
    res = apply_manual(res)
    res.to_csv(REVIEW, index=False)
    good = res[(res["flag"] == "ok") & res["lat"].notna()]
    good[["District", "State", "lat", "lon"]].round(5).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(good)} of {len(res)} districts | review file: {REVIEW}")
    bad = res[~res.index.isin(good.index)]
    if len(bad):
        print("\nNEED ATTENTION (not written to the centroid file):")
        print(bad[["District", "State", "flag", "matched"]].to_string(index=False))
        print(f"\nFix: add rows 'District,lat,lon' to {MANUAL} (look the place up on Google Maps) and rerun this script.")
    print("\nEyeball check (should be plausible for each state):")
    print(good.groupby("State")[["lat", "lon"]].agg(["min", "max"]).round(2).to_string())
    print("\nNext: python scripts/B3_imd_validation.py --centroids data/external/district_centroids.csv")


if __name__ == "__main__":
    main()