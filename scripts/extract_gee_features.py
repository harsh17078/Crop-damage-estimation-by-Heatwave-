"""
======================================================================
PHASE 1 / TRACK A
GOOGLE EARTH ENGINE FEATURE EXTRACTION
======================================================================

Project:
    Crop Damage Estimation Due to Heat Waves

Research Question:
    Can multi-source satellite-derived vegetation, thermal,
    and meteorological features be used with machine learning
    to estimate crop damage caused by heat waves?

Purpose:
    Extract district-year X features from Google Earth Engine.

Target:
    These features will later be merged with DES wheat yield data
    and used to calculate yield anomaly / crop damage labels.

======================================================================
DATA SOURCES
======================================================================

1. MODIS MOD13Q1
   Dataset:
       MODIS/061/MOD13Q1

   Feature:
       NDVI

2. MODIS MOD09A1
   Dataset:
       MODIS/061/MOD09A1

   Feature:
       NDMI

3. MODIS MOD11A2
   Dataset:
       MODIS/061/MOD11A2

   Feature:
       LST

4. CHIRPS DAILY
   Dataset:
       UCSB-CHG/CHIRPS/DAILY

   Feature:
       Rainfall
       Extreme rainfall / wet-day streak (ADDED)

5. ERA5-LAND DAILY_AGGR
   Dataset:
       ECMW/ERA5_LAND/DAILY_AGGR

   Features:
       Tmax
       Tmin
       Soil Moisture
       Heatwave Days
       Cumulative Heat Stress
       Peak Tmax

6. MODIS MCD12Q1
   Dataset:
       MODIS/061/MCD12Q1/2010_01_01

   Feature:
       Cropland mask

7. FAO GAUL 2015 LEVEL 2
   Dataset:
       FAO/GAUL/2015/level2

   Used for:
       District boundaries

======================================================================
PROJECT PERIOD
======================================================================

2003 - 2024

This matches the DES yield data period being prepared.

======================================================================
DISTRICTS
======================================================================

Requested:
    43

Available in GAUL 2015:
    41

Unavailable in GAUL 2015:
    Barnala
    Fazilka

These two districts are NOT replaced with another district.

======================================================================
GAUL NAME MAPPINGS
======================================================================

Project name              GAUL name

Ferozepur                 Firozpur
Sri Muktsar Sahib         Muktsar
Sonipat                   Sonepat
Yamunanagar                Yamuna Nagar
Budaun                     Badaun
Kanpur Nagar               Kanpur

======================================================================
FEATURES
======================================================================

BEFORE:
    NDVI_before
    NDMI_before
    LST_before

DURING:
    NDVI_during
    NDMI_during
    LST_during

AFTER:
    NDVI_after
    NDMI_after
    LST_after

CHANGE:
    dNDVI
    dNDMI
    dLST
    dNDVI_after
    dNDMI_after

CLIMATE:
    Tmax_during_C
    Tmin_during_C
    SoilMoisture_during
    Rainfall_before_mm
    Rainfall_during_mm

HEAT:
    HeatWaveDays_34C
    HeatWaveDays_36C
    HeatWaveDays_40C
    CumulativeHeatStress_C
    Tmax_max_during_C

EXCESS WATER (ADDED — mirrors the HEAT block, opposite failure mode,
confirmed necessary by the 2014 hailstorm CV-outlier investigation):
    ExtremeRainDays_20mm
    ExtremeRainDays_50mm
    Rainfall_max_daily_mm
    Rainfall_p95_mm
    WetDayStreak_max

DISTRICT STATISTICS:
    mean
    standard deviation
    10th percentile

======================================================================
COMMANDS
======================================================================

1. District sanity check:

    python scripts/extract_gee_features.py --sanity-check

2. Dataset availability check:

    python scripts/extract_gee_features.py --check-datasets

3. Run extraction:

    python scripts/extract_gee_features.py --run

4. Explicit project period:

    python scripts/extract_gee_features.py --run \
        --start-year 2003 \
        --end-year 2024

======================================================================
"""


# ======================================================================
# IMPORTS
# ======================================================================

import argparse
import time
import ee


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ID = "crop-heatwave-minor-project"


# ----------------------------------------------------------------------
# Project period
# ----------------------------------------------------------------------

START_YEAR = 2003
END_YEAR = 2024


# ----------------------------------------------------------------------
# Heatwave thresholds
# ----------------------------------------------------------------------

HEATWAVE_THRESHOLDS_C = [
    34.0,
    36.0,
    40.0,
]


# ----------------------------------------------------------------------
# Reference temperature for cumulative heat stress
# ----------------------------------------------------------------------

CUMULATIVE_HEAT_REF_TEMP_C = 30.0


# ----------------------------------------------------------------------
# Extreme rainfall thresholds (ADDED)
# ----------------------------------------------------------------------
# Mirrors HEATWAVE_THRESHOLDS_C, but for excess-water/hail-associated
# events - added after the 2014 CV-outlier investigation confirmed a
# documented 20-day hailstorm/unseasonal-rain event across Punjab,
# Haryana and UP (Feb 24 - Mar 14, 2014) was invisible to the original
# heat-only feature set.

EXTREME_RAIN_THRESHOLDS_MM = [
    20.0,
    50.0,
]


# ----------------------------------------------------------------------
# Wet-day threshold, for the max consecutive wet-day streak (ADDED)
# ----------------------------------------------------------------------

WET_DAY_THRESHOLD_MM = 1.0


# ----------------------------------------------------------------------
# Export configuration
# ----------------------------------------------------------------------

EXPORT_FOLDER = "wheat_heatwave_dataset_v3"

EXPORT_DESCRIPTION_PREFIX = (
    "wheat_heatwave_features_v3"
)


# ----------------------------------------------------------------------
# Spatial scale
# ----------------------------------------------------------------------

EXPORT_SCALE = 500


# ----------------------------------------------------------------------
# Earth Engine tileScale
# ----------------------------------------------------------------------

TILE_SCALE = 4


# ----------------------------------------------------------------------
# Delay between export task submissions
# ----------------------------------------------------------------------

TASK_DELAY_SECONDS = 5


# ======================================================================
# PROJECT DISTRICTS
# ======================================================================

DISTRICT_LIST = {

    "Punjab": [

        "Ludhiana",
        "Patiala",
        "Sangrur",
        "Bathinda",
        "Ferozepur",
        "Moga",
        "Amritsar",
        "Sri Muktsar Sahib",
        "Fazilka",
        "Barnala",
        "Mansa",
        "Faridkot",
        "Gurdaspur",
        "Kapurthala",
        "Jalandhar",

    ],

    "Haryana": [

        "Hisar",
        "Sirsa",
        "Fatehabad",
        "Karnal",
        "Kurukshetra",
        "Kaithal",
        "Jind",
        "Rohtak",
        "Sonipat",
        "Panipat",
        "Yamunanagar",
        "Ambala",
        "Bhiwani",
        "Jhajjar",

    ],

    "Uttar Pradesh": [

        "Meerut",
        "Muzaffarnagar",
        "Agra",
        "Aligarh",
        "Bulandshahr",
        "Kanpur Nagar",
        "Etawah",
        "Jhansi",
        "Saharanpur",
        "Mathura",
        "Firozabad",
        "Etah",
        "Budaun",
        "Hardoi",

    ],

}


# ======================================================================
# STATE LIST
# ======================================================================

STATES = list(
    DISTRICT_LIST.keys()
)


# ======================================================================
# ALL PROJECT DISTRICTS
# ======================================================================

ALL_DISTRICTS = [

    district

    for districts in DISTRICT_LIST.values()

    for district in districts

]


# ======================================================================
# GAUL NAME MAPPING
# ======================================================================

GAUL_DISTRICT_MAP = {

    "Ferozepur":
        "Firozpur",

    "Sri Muktsar Sahib":
        "Muktsar",

    "Sonipat":
        "Sonepat",

    "Yamunanagar":
        "Yamuna Nagar",

    "Budaun":
        "Badaun",

    "Kanpur Nagar":
        "Kanpur",

}


# ======================================================================
# EARTH ENGINE INITIALIZATION
# ======================================================================

def initialize_earth_engine():

    """
    Initialize Google Earth Engine.
    """

    try:

        ee.Initialize(
            project=PROJECT_ID
        )

        print(
            "Earth Engine initialized successfully."
        )

    except Exception as e:

        print(
            "\nEarth Engine initialization failed."
        )

        print(
            "\nTry running:"
        )

        print(
            "    earthengine authenticate"
        )

        print(
            "\nThen run this script again."
        )

        print(
            "\nOriginal error:"
        )

        print(e)

        raise


# ======================================================================
# LOAD EARTH ENGINE DATASETS
# ======================================================================

def get_collections():

    """
    Create Earth Engine ImageCollections.

    This function MUST be called after ee.Initialize().
    """

    # ------------------------------------------------------------------
    # MODIS NDVI
    # ------------------------------------------------------------------

    NDVI_COL = (

        ee.ImageCollection(
            "MODIS/061/MOD13Q1"
        )

        .select(
            "NDVI"
        )

    )


    # ------------------------------------------------------------------
    # MODIS Surface Reflectance
    # ------------------------------------------------------------------

    MOD09_COL = (

        ee.ImageCollection(
            "MODIS/061/MOD09A1"
        )

    )


    # ------------------------------------------------------------------
    # MODIS LST
    # ------------------------------------------------------------------

    LST_COL = (

        ee.ImageCollection(
            "MODIS/061/MOD11A2"
        )

        .select(
            "LST_Day_1km"
        )

    )


    # ------------------------------------------------------------------
    # CHIRPS rainfall
    # ------------------------------------------------------------------

    CHIRPS_COL = (

        ee.ImageCollection(
            "UCSB-CHG/CHIRPS/DAILY"
        )

        .select(
            "precipitation"
        )

    )


    # ------------------------------------------------------------------
    # ERA5 LAND
    # ------------------------------------------------------------------

    ERA5_COL = (

        ee.ImageCollection(
            "ECMWF/ERA5_LAND/DAILY_AGGR"
        )

    )


    return (

        NDVI_COL,

        MOD09_COL,

        LST_COL,

        CHIRPS_COL,

        ERA5_COL,

    )


# ======================================================================
# LOAD GAUL INDIA LEVEL 2
# ======================================================================

def get_gaul_india():

    """
    Load GAUL 2015 level-2 administrative boundaries for India.
    """

    districts = (

        ee.FeatureCollection(
            "FAO/GAUL/2015/level2"
        )

        .filter(
            ee.Filter.eq(
                "ADM0_NAME",
                "India"
            )
        )

        .filter(
            ee.Filter.inList(
                "ADM1_NAME",
                STATES
            )
        )

    )

    return districts


# ======================================================================
# GET PROJECT DISTRICTS
# ======================================================================

def get_districts():

    """
    Match project district names against GAUL 2015.

    Project district names are preserved in:
        PROJECT_DISTRICT

    GAUL district names remain in:
        ADM2_NAME
    """

    gaul = get_gaul_india()

    matched_collections = []


    # ------------------------------------------------------------------
    # Process each state and district
    # ------------------------------------------------------------------

    for state, project_districts in DISTRICT_LIST.items():

        for project_district in project_districts:

            gaul_name = GAUL_DISTRICT_MAP.get(
                project_district,
                project_district
            )


            matched = (

                gaul

                .filter(
                    ee.Filter.eq(
                        "ADM1_NAME",
                        state
                    )
                )

                .filter(
                    ee.Filter.eq(
                        "ADM2_NAME",
                        gaul_name
                    )
                )

            )


            matched = (

                matched

                .map(

                    lambda feature,
                    state_name=state,
                    district_name=project_district:

                        feature.set({

                            "PROJECT_STATE":
                                state_name,

                            "PROJECT_DISTRICT":
                                district_name,

                        })

                )

            )


            matched_collections.append(
                matched
            )


    # ------------------------------------------------------------------
    # Merge collections
    # ------------------------------------------------------------------

    if not matched_collections:

        return ee.FeatureCollection([])


    result = matched_collections[0]


    for collection in matched_collections[1:]:

        result = result.merge(
            collection
        )


    return result


# ======================================================================
# CROPLAND MASK
# ======================================================================

def get_crop_mask():

    """
    Create a broad cropland mask using MODIS MCD12Q1.

    IGBP class 12 = Croplands.

    IMPORTANT:
        This identifies cropland.
        It does NOT specifically identify wheat.
    """

    lc = (

        ee.Image(
            "MODIS/061/MCD12Q1/2010_01_01"
        )

        .select(
            "LC_Type1"
        )

    )


    return lc.eq(12)


# ======================================================================
# PHENOLOGICAL WINDOWS
# ======================================================================

def phase_windows(year):

    """
    Define BEFORE, DURING and AFTER periods.

    BEFORE:
        Dec 1 previous year
        Jan 31 current year

    DURING:
        Feb 15
        Apr 10

    AFTER:
        Apr 11
        Apr 30
    """

    before = (

        f"{year - 1}-12-01",

        f"{year}-01-31"

    )


    during = (

        f"{year}-02-15",

        f"{year}-04-10"

    )


    after = (

        f"{year}-04-11",

        f"{year}-04-30"

    )


    return (
        before,
        during,
        after
    )


# ======================================================================
# NDMI
# ======================================================================

def ndmi_image(
    MOD09_COL,
    start,
    end
):

    """
    Calculate NDMI.

    NDMI = (NIR - SWIR) / (NIR + SWIR)

    MOD09A1:
        sur_refl_b02 = NIR
        sur_refl_b06 = SWIR
    """

    img = (

        MOD09_COL

        .filterDate(
            start,
            end
        )

        .mean()

    )


    nir = (

        img

        .select(
            "sur_refl_b02"
        )

        .multiply(
            0.0001
        )

    )


    swir = (

        img

        .select(
            "sur_refl_b06"
        )

        .multiply(
            0.0001
        )

    )


    ndmi = (

        nir

        .subtract(
            swir
        )

        .divide(
            nir.add(swir)
        )

        .rename(
            "NDMI"
        )

    )


    return ndmi


# ======================================================================
# VEGETATION + THERMAL COMPOSITE
# ======================================================================

def veg_thermal_composite(
    NDVI_COL,
    MOD09_COL,
    LST_COL,
    start,
    end
):

    """
    Generate:

        NDVI
        NDMI
        LST
    """

    # ------------------------------------------------------------------
    # NDVI
    # ------------------------------------------------------------------

    ndvi = (

        NDVI_COL

        .filterDate(
            start,
            end
        )

        .mean()

        .multiply(
            0.0001
        )

        .rename(
            "NDVI"
        )

    )


    # ------------------------------------------------------------------
    # NDMI
    # ------------------------------------------------------------------

    ndmi = ndmi_image(

        MOD09_COL,

        start,

        end

    )


    # ------------------------------------------------------------------
    # LST
    # ------------------------------------------------------------------

    lst = (

        LST_COL

        .filterDate(
            start,
            end
        )

        .mean()

        .multiply(
            0.02
        )

        .subtract(
            273.15
        )

        .rename(
            "LST"
        )

    )


    return ee.Image.cat([

        ndvi,
        ndmi,
        lst,

    ])


# ======================================================================
# GRADUATED HEATWAVE DAYS
# ======================================================================

def graduated_heatwave_days(
    ERA5_COL,
    start,
    end
):

    """
    Calculate number of days exceeding:

        34 C
        36 C
        40 C
    """

    tmax_c = (

        ERA5_COL

        .select(
            "temperature_2m_max"
        )

        .filterDate(
            start,
            end
        )

        .map(

            lambda image:

                image.subtract(
                    273.15
                )

        )

    )


    bands = []


    for threshold in HEATWAVE_THRESHOLDS_C:

        band = (

            tmax_c

            .map(

                lambda image,
                threshold_value=threshold:

                    image.gt(
                        threshold_value
                    )

            )

            .sum()

            .rename(

                f"HeatWaveDays_"
                f"{int(threshold)}C"

            )

        )


        bands.append(
            band
        )


    return ee.Image.cat(
        bands
    )


# ======================================================================
# CUMULATIVE HEAT STRESS
# ======================================================================

def cumulative_heat_stress(
    ERA5_COL,
    start,
    end,
    ref_temp=CUMULATIVE_HEAT_REF_TEMP_C
):

    """
    Calculate cumulative heat stress.

    Daily heat stress:

        max(Tmax - reference temperature, 0)

    Then sum across the period.

    Output:
        CumulativeHeatStress_C
    """

    tmax_c = (

        ERA5_COL

        .select(
            "temperature_2m_max"
        )

        .filterDate(
            start,
            end
        )

        .map(

            lambda image:

                image.subtract(
                    273.15
                )

        )

    )


    excess = (

        tmax_c

        .map(

            lambda image:

                image

                .subtract(
                    ref_temp
                )

                .max(
                    0
                )

        )

    )


    return (

        excess

        .sum()

        .rename(
            "CumulativeHeatStress_C"
        )

    )


# ======================================================================
# PEAK TMAX
# ======================================================================

def peak_tmax(
    ERA5_COL,
    start,
    end
):

    """
    Maximum daily Tmax during the DURING period.
    """

    tmax_c = (

        ERA5_COL

        .select(
            "temperature_2m_max"
        )

        .filterDate(
            start,
            end
        )

        .map(

            lambda image:

                image.subtract(
                    273.15
                )

        )

    )


    return (

        tmax_c

        .max()

        .rename(
            "Tmax_max_during_C"
        )

    )


# ======================================================================
# GRADUATED EXTREME RAINFALL DAYS (ADDED)
# ======================================================================

def extreme_rain_days(
    CHIRPS_COL,
    start,
    end
):

    """
    Calculate number of days exceeding:

        20 mm
        50 mm

    Mirrors graduated_heatwave_days() but for excess-water events - added
    after the 2014 CV-outlier investigation confirmed a documented
    hailstorm/unseasonal-rain event across the study region was invisible
    to the heat-only feature set.
    """

    daily = (

        CHIRPS_COL

        .filterDate(
            start,
            end
        )

    )


    bands = []


    for threshold in EXTREME_RAIN_THRESHOLDS_MM:

        band = (

            daily

            .map(

                lambda image,
                threshold_value=threshold:

                    image.gt(
                        threshold_value
                    )

            )

            .sum()

            .rename(

                f"ExtremeRainDays_"
                f"{int(threshold)}mm"

            )

        )


        bands.append(
            band
        )


    return ee.Image.cat(
        bands
    )


# ======================================================================
# MAX DAILY RAINFALL (ADDED)
# ======================================================================

def max_daily_rainfall(
    CHIRPS_COL,
    start,
    end
):

    """
    Maximum single-day rainfall during the period.

    Detects an extreme downpour even if the seasonal TOTAL
    (Rainfall_during_mm) looks unremarkable.
    """

    return (

        CHIRPS_COL

        .filterDate(
            start,
            end
        )

        .max()

        .rename(
            "Rainfall_max_daily_mm"
        )

    )


# ======================================================================
# RAINFALL 95TH PERCENTILE - TEMPORAL (ADDED)
# ======================================================================

def rainfall_p95(
    CHIRPS_COL,
    start,
    end
):

    """
    95th percentile of DAILY rainfall values during the period.

    This is a TEMPORAL percentile per pixel (across the daily time
    series) - distinct from the spatial mean/stdDev/p10 combined reducer
    applied later at the district level in export_year(). More robust to
    a single-day outlier than the max, while still capturing how extreme
    the worst rain days got.
    """

    return (

        CHIRPS_COL

        .filterDate(
            start,
            end
        )

        .reduce(
            ee.Reducer.percentile([95])
        )

        .rename(
            "Rainfall_p95_mm"
        )

    )


# ======================================================================
# MAX CONSECUTIVE WET-DAY STREAK (ADDED)
# ======================================================================

def wet_day_streak_max(
    CHIRPS_COL,
    start,
    end,
    wet_threshold=WET_DAY_THRESHOLD_MM
):

    """
    Longest run of consecutive days with rainfall above wet_threshold.

    GEE has no built-in "consecutive count" reducer, so this threads a
    running per-pixel streak counter through the sorted daily images via
    ee.List.iterate(): the streak increments on a wet day and resets to
    0 on a dry day, with a second running image tracking the maximum
    streak seen so far.
    """

    daily_list = (

        CHIRPS_COL

        .filterDate(
            start,
            end
        )

        .sort(
            "system:time_start"
        )

        .toList(
            200
        )

    )


    initial_state = ee.Dictionary({

        "streak":
            ee.Image(0),

        "max_streak":
            ee.Image(0),

    })


    def accumulate(image, state):

        state = ee.Dictionary(
            state
        )

        prev_streak = ee.Image(
            state.get("streak")
        )

        prev_max = ee.Image(
            state.get("max_streak")
        )

        is_wet = (

            ee.Image(image)

            .gt(
                wet_threshold
            )

        )

        new_streak = (

            prev_streak

            .add(1)

            .multiply(
                is_wet
            )

        )

        new_max = prev_max.max(new_streak)

        return ee.Dictionary({

            "streak":
                new_streak,

            "max_streak":
                new_max,

        })


    final_state = ee.Dictionary(

        daily_list.iterate(
            accumulate,
            initial_state
        )

    )


    return (

        ee.Image(
            final_state.get("max_streak")
        )

        .rename(
            "WetDayStreak_max"
        )

    )


# ======================================================================
# CLIMATE FEATURES
# ======================================================================

def climate_features(
    ERA5_COL,
    CHIRPS_COL,
    before,
    during
):

    """
    Generate:

        Tmax_during_C
        Tmin_during_C
        SoilMoisture_during
        Rainfall_before_mm
        Rainfall_during_mm
        HeatWaveDays_34C
        HeatWaveDays_36C
        HeatWaveDays_40C
        CumulativeHeatStress_C
        Tmax_max_during_C
        ExtremeRainDays_20mm    (ADDED)
        ExtremeRainDays_50mm    (ADDED)
        Rainfall_max_daily_mm   (ADDED)
        Rainfall_p95_mm         (ADDED)
        WetDayStreak_max        (ADDED)
    """

    before_start, before_end = before

    during_start, during_end = during


    # ------------------------------------------------------------------
    # Tmax
    # ------------------------------------------------------------------

    tmax_mean = (

        ERA5_COL

        .select(
            "temperature_2m_max"
        )

        .filterDate(
            during_start,
            during_end
        )

        .mean()

        .subtract(
            273.15
        )

        .rename(
            "Tmax_during_C"
        )

    )


    # ------------------------------------------------------------------
    # Tmin
    # ------------------------------------------------------------------

    tmin_mean = (

        ERA5_COL

        .select(
            "temperature_2m_min"
        )

        .filterDate(
            during_start,
            during_end
        )

        .mean()

        .subtract(
            273.15
        )

        .rename(
            "Tmin_during_C"
        )

    )


    # ------------------------------------------------------------------
    # Soil moisture
    # ------------------------------------------------------------------

    soil_moisture = (

        ERA5_COL

        .select(
            "volumetric_soil_water_layer_1"
        )

        .filterDate(
            during_start,
            during_end
        )

        .mean()

        .rename(
            "SoilMoisture_during"
        )

    )


    # ------------------------------------------------------------------
    # Rainfall BEFORE
    # ------------------------------------------------------------------

    rainfall_before = (

        CHIRPS_COL

        .filterDate(
            before_start,
            before_end
        )

        .sum()

        .rename(
            "Rainfall_before_mm"
        )

    )


    # ------------------------------------------------------------------
    # Rainfall DURING
    # ------------------------------------------------------------------

    rainfall_during = (

        CHIRPS_COL

        .filterDate(
            during_start,
            during_end
        )

        .sum()

        .rename(
            "Rainfall_during_mm"
        )

    )


    # ------------------------------------------------------------------
    # Heatwave days
    # ------------------------------------------------------------------

    heatwave_days = graduated_heatwave_days(

        ERA5_COL,

        during_start,

        during_end

    )


    # ------------------------------------------------------------------
    # Cumulative heat
    # ------------------------------------------------------------------

    cumulative_heat = cumulative_heat_stress(

        ERA5_COL,

        during_start,

        during_end

    )


    # ------------------------------------------------------------------
    # Peak temperature
    # ------------------------------------------------------------------

    peak_temperature = peak_tmax(

        ERA5_COL,

        during_start,

        during_end

    )


    # ------------------------------------------------------------------
    # Extreme rain days (ADDED)
    # ------------------------------------------------------------------

    extreme_rain = extreme_rain_days(

        CHIRPS_COL,

        during_start,

        during_end

    )


    # ------------------------------------------------------------------
    # Max daily rainfall (ADDED)
    # ------------------------------------------------------------------

    max_rain = max_daily_rainfall(

        CHIRPS_COL,

        during_start,

        during_end

    )


    # ------------------------------------------------------------------
    # Rainfall 95th percentile (ADDED)
    # ------------------------------------------------------------------

    rain_p95 = rainfall_p95(

        CHIRPS_COL,

        during_start,

        during_end

    )


    # ------------------------------------------------------------------
    # Max wet-day streak (ADDED)
    # ------------------------------------------------------------------

    wet_streak = wet_day_streak_max(

        CHIRPS_COL,

        during_start,

        during_end

    )


    # ------------------------------------------------------------------
    # Combine all climate features
    # ------------------------------------------------------------------

    return ee.Image.cat([

        tmax_mean,

        tmin_mean,

        soil_moisture,

        rainfall_before,

        rainfall_during,

        heatwave_days,

        cumulative_heat,

        peak_temperature,

        extreme_rain,       # ADDED

        max_rain,            # ADDED

        rain_p95,             # ADDED

        wet_streak,            # ADDED

    ])


# ======================================================================
# BUILD COMPLETE YEAR FEATURE IMAGE
# ======================================================================

def build_feature_image(
    year,
    crop_mask,
    NDVI_COL,
    MOD09_COL,
    LST_COL,
    CHIRPS_COL,
    ERA5_COL
):

    """
    Create complete feature image for one year.
    """

    before, during, after = phase_windows(
        year
    )


    # ==================================================================
    # BEFORE
    # ==================================================================

    vt_before = (

        veg_thermal_composite(

            NDVI_COL,

            MOD09_COL,

            LST_COL,

            *before

        )

        .rename([

            "NDVI_before",

            "NDMI_before",

            "LST_before",

        ])

    )


    # ==================================================================
    # DURING
    # ==================================================================

    vt_during = (

        veg_thermal_composite(

            NDVI_COL,

            MOD09_COL,

            LST_COL,

            *during

        )

        .rename([

            "NDVI_during",

            "NDMI_during",

            "LST_during",

        ])

    )


    # ==================================================================
    # AFTER
    # ==================================================================

    vt_after = (

        veg_thermal_composite(

            NDVI_COL,

            MOD09_COL,

            LST_COL,

            *after

        )

        .rename([

            "NDVI_after",

            "NDMI_after",

            "LST_after",

        ])

    )


    # ==================================================================
    # CHANGE FEATURES
    # ==================================================================

    d_ndvi = (

        vt_during

        .select(
            "NDVI_during"
        )

        .subtract(

            vt_before.select(
                "NDVI_before"
            )

        )

        .rename(
            "dNDVI"
        )

    )


    d_ndmi = (

        vt_during

        .select(
            "NDMI_during"
        )

        .subtract(

            vt_before.select(
                "NDMI_before"
            )

        )

        .rename(
            "dNDMI"
        )

    )


    d_lst = (

        vt_during

        .select(
            "LST_during"
        )

        .subtract(

            vt_before.select(
                "LST_before"
            )

        )

        .rename(
            "dLST"
        )

    )


    d_ndvi_after = (

        vt_after

        .select(
            "NDVI_after"
        )

        .subtract(

            vt_during.select(
                "NDVI_during"
            )

        )

        .rename(
            "dNDVI_after"
        )

    )


    d_ndmi_after = (

        vt_after

        .select(
            "NDMI_after"
        )

        .subtract(

            vt_during.select(
                "NDMI_during"
            )

        )

        .rename(
            "dNDMI_after"
        )

    )


    # ==================================================================
    # VEGETATION + THERMAL STACK
    # ==================================================================

    veg_thermal_stack = ee.Image.cat([

        vt_before,

        vt_during,

        vt_after,

        d_ndvi,

        d_ndmi,

        d_lst,

        d_ndvi_after,

        d_ndmi_after,

    ])


    # ==================================================================
    # CLIMATE STACK
    # ==================================================================

    climate_stack = climate_features(

        ERA5_COL,

        CHIRPS_COL,

        before,

        during

    )


    # ==================================================================
    # COMPLETE STACK
    # ==================================================================

    full_stack = (

        ee.Image.cat([

            veg_thermal_stack,

            climate_stack,

        ])

        .updateMask(
            crop_mask
        )

    )


    return (

        full_stack

        .set(
            "year",
            year
        )

    )


# ======================================================================
# COMBINED REDUCER
# ======================================================================

def combined_reducer():

    """
    District-level statistics:

        mean
        standard deviation
        10th percentile
    """

    reducer = (

        ee.Reducer.mean()

        .combine(

            ee.Reducer.stdDev(),

            sharedInputs=True

        )

        .combine(

            ee.Reducer.percentile(
                [10]
            ),

            sharedInputs=True

        )

    )


    return reducer


# ======================================================================
# EXPORT ONE YEAR
# ======================================================================

def export_year(
    year,
    districts,
    crop_mask,
    NDVI_COL,
    MOD09_COL,
    LST_COL,
    CHIRPS_COL,
    ERA5_COL,
    scale=EXPORT_SCALE
):

    """
    Build and export district-level features for one year.
    """

    print()
    print(
        f"Preparing export for {year}..."
    )


    # ------------------------------------------------------------------
    # Build feature image
    # ------------------------------------------------------------------

    img = build_feature_image(

        year,

        crop_mask,

        NDVI_COL,

        MOD09_COL,

        LST_COL,

        CHIRPS_COL,

        ERA5_COL,

    )


    # ------------------------------------------------------------------
    # District reduction
    # ------------------------------------------------------------------

    reduced = (

        img

        .reduceRegions(

            collection=(

                districts.select([

                    "ADM1_NAME",

                    "ADM2_NAME",

                    "PROJECT_STATE",

                    "PROJECT_DISTRICT",

                ])

            ),

            reducer=combined_reducer(),

            scale=scale,

            tileScale=TILE_SCALE,

        )

        .map(

            lambda feature:

                feature.set(
                    "year",
                    year
                )

        )

    )


    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    task = ee.batch.Export.table.toDrive(

        collection=reduced,

        description=(

            f"{EXPORT_DESCRIPTION_PREFIX}_"
            f"{year}"

        ),

        folder=EXPORT_FOLDER,

        fileNamePrefix=(

            f"{EXPORT_DESCRIPTION_PREFIX}_"
            f"{year}"

        ),

        fileFormat="CSV",

    )


    task.start()


    print(
        f"Year {year}: export task started."
    )


    print(
        f"Export folder: {EXPORT_FOLDER}"
    )


    return task


# ======================================================================
# DISTRICT SANITY CHECK
# ======================================================================

def sanity_check(districts):

    """
    Compare requested project districts with GAUL 2015.
    """

    matched_features = (

        districts

        .select([

            "PROJECT_STATE",

            "PROJECT_DISTRICT",

            "ADM1_NAME",

            "ADM2_NAME",

        ])

        .getInfo()["features"]

    )


    matched_project_districts = set()

    matched_details = []


    # ------------------------------------------------------------------
    # Extract matched district information
    # ------------------------------------------------------------------

    for feature in matched_features:

        properties = feature["properties"]


        project_state = properties.get(
            "PROJECT_STATE"
        )


        project_district = properties.get(
            "PROJECT_DISTRICT"
        )


        gaul_state = properties.get(
            "ADM1_NAME"
        )


        gaul_district = properties.get(
            "ADM2_NAME"
        )


        matched_project_districts.add(
            project_district
        )


        matched_details.append({

            "state":
                project_state,

            "project":
                project_district,

            "gaul_state":
                gaul_state,

            "gaul":
                gaul_district,

        })


    # ------------------------------------------------------------------
    # Calculate missing
    # ------------------------------------------------------------------

    requested_set = set(
        ALL_DISTRICTS
    )


    missing = sorted(

        requested_set
        -
        matched_project_districts

    )


    # ==================================================================
    # PRINT REPORT
    # ==================================================================

    print()
    print("=" * 70)
    print("DISTRICT SANITY CHECK")
    print("=" * 70)

    print()

    print(
        f"Requested districts : "
        f"{len(requested_set)}"
    )

    print(
        f"Matched districts   : "
        f"{len(matched_project_districts)}"
    )

    print(
        f"Missing districts   : "
        f"{len(missing)}"
    )


    # ==================================================================
    # MAPPING REPORT
    # ==================================================================

    print()
    print("-" * 70)
    print("PROJECT → GAUL DISTRICT MAPPING")
    print("-" * 70)


    for item in sorted(

        matched_details,

        key=lambda x: (
            x["state"],
            x["project"]
        )

    ):

        print(

            f"{item['state']:18s} | "
            f"{item['project']:25s} → "
            f"{item['gaul']}"

        )


    # ==================================================================
    # MISSING REPORT
    # ==================================================================

    if missing:

        print()
        print("-" * 70)
        print("MISSING DISTRICTS")
        print("-" * 70)


        for district in missing:

            print(
                f"  - {district}"
            )


        print()
        print(
            "NOTE:"
        )


        print(
            "These districts do not have matching "
            "GAUL 2015 boundaries."
        )


        print(
            "They will NOT be assigned another "
            "district's geometry."
        )


    else:

        print()
        print(
            "All requested districts matched successfully."
        )


    print()
    print("=" * 70)


    return missing


# ======================================================================
# DATASET AVAILABILITY CHECK
# ======================================================================

def check_dataset_availability():

    """
    Check Earth Engine dataset accessibility.

    IMPORTANT:
        Different Earth Engine asset types require different
        constructors.

        ImageCollection  -> ee.ImageCollection()
        Image            -> ee.Image()
        FeatureCollection-> ee.FeatureCollection()

    GAUL is a FeatureCollection, NOT an ImageCollection.
    """

    print()
    print("=" * 70)
    print("DATASET AVAILABILITY CHECK")
    print("=" * 70)


    # ==================================================================
    # IMAGE COLLECTIONS
    # ==================================================================

    image_collections = {

        "MODIS NDVI":
            "MODIS/061/MOD13Q1",

        "MODIS Surface Reflectance":
            "MODIS/061/MOD09A1",

        "MODIS LST":
            "MODIS/061/MOD11A2",

        "CHIRPS":
            "UCSB-CHG/CHIRPS/DAILY",

        "ERA5-LAND":
            "ECMWF/ERA5_LAND/DAILY_AGGR",

    }


    # ==================================================================
    # IMAGES
    # ==================================================================

    images = {

        "MODIS Land Cover":
            "MODIS/061/MCD12Q1/2010_01_01",

    }


    # ==================================================================
    # FEATURE COLLECTIONS
    # ==================================================================

    feature_collections = {

        "GAUL Level 2":
            "FAO/GAUL/2015/level2",

    }


    # ==================================================================
    # CHECK IMAGE COLLECTIONS
    # ==================================================================

    for name, dataset_id in image_collections.items():

        try:

            collection = ee.ImageCollection(
                dataset_id
            )


            count = (

                collection

                .size()

                .getInfo()

            )


            print(

                f"[OK] {name:28s} "
                f"{dataset_id}"

            )


            print(

                f"     Images available: "
                f"{count}"

            )


        except Exception as e:

            print(

                f"[ERROR] {name:28s} "
                f"{dataset_id}"

            )


            print(
                f"        {e}"
            )


    # ==================================================================
    # CHECK IMAGES
    # ==================================================================

    for name, dataset_id in images.items():

        try:

            image = ee.Image(
                dataset_id
            )


            band_count = (

                image

                .bandNames()

                .size()

                .getInfo()

            )


            print(

                f"[OK] {name:28s} "
                f"{dataset_id}"

            )


            print(

                f"     Bands available: "
                f"{band_count}"

            )


        except Exception as e:

            print(

                f"[ERROR] {name:28s} "
                f"{dataset_id}"

            )


            print(
                f"        {e}"
            )


    # ==================================================================
    # CHECK FEATURE COLLECTIONS
    # ==================================================================

    for name, dataset_id in feature_collections.items():

        try:

            collection = ee.FeatureCollection(
                dataset_id
            )


            count = (

                collection

                .size()

                .getInfo()

            )


            print(

                f"[OK] {name:28s} "
                f"{dataset_id}"

            )


            print(

                f"     Features available: "
                f"{count}"

            )


        except Exception as e:

            print(

                f"[ERROR] {name:28s} "
                f"{dataset_id}"

            )


            print(
                f"        {e}"
            )


    print()
    print("=" * 70)


# ======================================================================
# PRINT CONFIGURATION
# ======================================================================

def print_configuration(
    start_year,
    end_year
):

    """
    Print extraction configuration.
    """

    print()
    print("=" * 70)
    print("GEE FEATURE EXTRACTION CONFIGURATION")
    print("=" * 70)

    print()

    print(
        f"Earth Engine project : "
        f"{PROJECT_ID}"
    )

    print(
        f"Start year           : "
        f"{start_year}"
    )

    print(
        f"End year             : "
        f"{end_year}"
    )

    print(
        f"Requested districts  : "
        f"{len(ALL_DISTRICTS)}"
    )

    print(
        f"Export scale         : "
        f"{EXPORT_SCALE} m"
    )

    print(
        f"Tile scale           : "
        f"{TILE_SCALE}"
    )

    print(
        f"Export folder        : "
        f"{EXPORT_FOLDER}"
    )

    print()

    print(
        f"Heatwave thresholds  : "
        f"{HEATWAVE_THRESHOLDS_C}"
    )

    print(
        f"Cumulative heat ref. : "
        f"{CUMULATIVE_HEAT_REF_TEMP_C} C"
    )

    print(
        f"Extreme rain thresh. : "
        f"{EXTREME_RAIN_THRESHOLDS_MM}"
    )

    print(
        f"Wet-day threshold    : "
        f"{WET_DAY_THRESHOLD_MM} mm"
    )

    print()

    print("=" * 70)


# ======================================================================
# MAIN
# ======================================================================

def main():

    # ==================================================================
    # ARGUMENT PARSER
    # ==================================================================

    parser = argparse.ArgumentParser(

        description=(

            "Extract district-level satellite, "
            "thermal and meteorological features "
            "using Google Earth Engine."

        )

    )


    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------

    parser.add_argument(

        "--sanity-check",

        action="store_true",

        help=(
            "Check project district names "
            "against GAUL 2015."
        )

    )


    # ------------------------------------------------------------------
    # Dataset check
    # ------------------------------------------------------------------

    parser.add_argument(

        "--check-datasets",

        action="store_true",

        help=(
            "Check configured Earth Engine datasets."
        )

    )


    # ------------------------------------------------------------------
    # Run extraction
    # ------------------------------------------------------------------

    parser.add_argument(

        "--run",

        action="store_true",

        help=(
            "Submit yearly Earth Engine export tasks."
        )

    )


    # ------------------------------------------------------------------
    # Start year
    # ------------------------------------------------------------------

    parser.add_argument(

        "--start-year",

        type=int,

        default=START_YEAR,

        help=(
            f"Start year. Default: {START_YEAR}"
        )

    )


    # ------------------------------------------------------------------
    # End year
    # ------------------------------------------------------------------

    parser.add_argument(

        "--end-year",

        type=int,

        default=END_YEAR,

        help=(
            f"End year. Default: {END_YEAR}"
        )

    )


    args = parser.parse_args()


    # ==================================================================
    # VALIDATE YEARS
    # ==================================================================

    if args.start_year > args.end_year:

        raise ValueError(

            "start-year cannot be greater "
            "than end-year."

        )


    # ==================================================================
    # INITIALIZE EARTH ENGINE
    # ==================================================================

    initialize_earth_engine()


    # ==================================================================
    # PRINT CONFIGURATION
    # ==================================================================

    print_configuration(

        args.start_year,

        args.end_year

    )


    # ==================================================================
    # LOAD DATASETS
    # ==================================================================

    print()
    print(
        "Loading Earth Engine datasets..."
    )


    (

        NDVI_COL,

        MOD09_COL,

        LST_COL,

        CHIRPS_COL,

        ERA5_COL,

    ) = get_collections()


    print(
        "Datasets loaded successfully."
    )


    # ==================================================================
    # DATASET CHECK
    # ==================================================================

    if args.check_datasets:

        check_dataset_availability()

        return


    # ==================================================================
    # LOAD DISTRICTS
    # ==================================================================

    print()
    print(
        "Loading district boundaries..."
    )


    districts = get_districts()


    # ==================================================================
    # SANITY CHECK
    # ==================================================================

    if args.sanity_check:

        sanity_check(
            districts
        )

        return


    # ==================================================================
    # RUN EXTRACTION
    # ==================================================================

    if args.run:

        missing = sanity_check(
            districts
        )


        # --------------------------------------------------------------
        # Number of matched districts
        # --------------------------------------------------------------

        matched_count = (

            len(ALL_DISTRICTS)

            -
            len(missing)

        )


        if matched_count == 0:

            raise RuntimeError(

                "No project districts matched "
                "GAUL 2015."

            )


        # ==============================================================
        # EXTRACTION HEADER
        # ==============================================================

        print()
        print("=" * 70)
        print("STARTING GEE FEATURE EXTRACTION")
        print("=" * 70)

        print()

        print(
            f"Years              : "
            f"{args.start_year}-"
            f"{args.end_year}"
        )

        print(
            f"Requested districts : "
            f"{len(ALL_DISTRICTS)}"
        )

        print(
            f"Matched districts   : "
            f"{matched_count}"
        )

        print(
            f"Skipped districts   : "
            f"{len(missing)}"
        )


        # ==============================================================
        # EXPECTED ROW COUNT
        # ==============================================================

        number_of_years = (

            args.end_year

            -
            args.start_year

            +
            1

        )


        expected_rows = (

            matched_count
            *
            number_of_years

        )


        print()

        print(

            "Expected district-year "
            "observations approximately: "
            f"{expected_rows}"

        )


        print()

        print(
            "One export task will be created "
            "for each year."
        )


        print(

            f"Total yearly export tasks: "
            f"{number_of_years}"

        )


        # ==============================================================
        # CROPLAND MASK
        # ==============================================================

        print()
        print(
            "Preparing cropland mask..."
        )


        crop_mask = get_crop_mask()


        print(
            "Cropland mask prepared."
        )


        # ==============================================================
        # YEAR-BY-YEAR EXTRACTION
        # ==============================================================

        for year in range(

            args.start_year,

            args.end_year + 1

        ):

            try:

                export_year(

                    year,

                    districts,

                    crop_mask,

                    NDVI_COL,

                    MOD09_COL,

                    LST_COL,

                    CHIRPS_COL,

                    ERA5_COL,

                    scale=EXPORT_SCALE,

                )


                print(

                    f"Waiting "
                    f"{TASK_DELAY_SECONDS} seconds "
                    f"before next task..."

                )


                time.sleep(
                    TASK_DELAY_SECONDS
                )


            except Exception as e:

                print()
                print("=" * 70)

                print(
                    f"ERROR WHILE SUBMITTING YEAR "
                    f"{year}"
                )

                print("=" * 70)

                print(e)

                print()

                print(
                    "Continuing with next year..."
                )


        # ==============================================================
        # FINISHED
        # ==============================================================

        print()
        print("=" * 70)
        print("ALL EXPORT TASKS SUBMITTED")
        print("=" * 70)

        print()

        print(
            f"Years submitted: "
            f"{args.start_year}-"
            f"{args.end_year}"
        )

        print(

            f"Number of yearly tasks: "
            f"{number_of_years}"

        )

        print()

        print(
            "Google Drive folder:"
        )

        print(
            f"    {EXPORT_FOLDER}"
        )

        print()

        print(
            "Earth Engine Tasks:"
        )

        print(
            "    https://code.earthengine.google.com/tasks"
        )

        print()

        print(
            "Matched districts:"
        )

        print(
            f"    {matched_count}"
        )

        print()

        print(
            "Unavailable districts:"
        )


        for district in missing:

            print(
                f"    - {district}"
            )


        print()

        print(
            "The unavailable districts were NOT "
            "assigned substitute geometries."
        )

        print()

        return


    # ==================================================================
    # NO ACTION
    # ==================================================================

    parser.print_help()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()