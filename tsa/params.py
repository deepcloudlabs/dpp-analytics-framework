"""Parameter tables for the synthetic textile lifecycle scenario.

All values are ILLUSTRATIVE and serve only to give the synthetic data a
plausible order of magnitude. Each table records a ``source`` tag: either the
BibTeX key of a literature/data source that informed the range, or
``"assumption"`` when the value is an author assumption. The synthetic data are
not calibrated to any life-cycle inventory database, and results obtained with
them must not be interpreted as industrial evidence.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Countries: grid carbon intensity (kg CO2e/kWh, location-based), renewable
# share of electricity generation, AWARE water-scarcity characterisation
# factor (m3 world-eq / m3), and an assumed industrial heat fuel mix.
# ---------------------------------------------------------------------------
COUNTRIES = {
    # grid_ef: Ember 2025, lifecycle gCO2e/kWh -> kg/kWh; renew: Ember 2025 renewable share of generation;
    # aware_na / aware_un: AWARE 2.0 annual country CFs (non-agricultural / unspecified); fuel: assumption
    "CN": dict(grid_ef=0.528, renew=0.365, aware_na=6.29, aware_un=22.3, fuel={"coal": 0.6, "gas": 0.3, "oil": 0.1}, region="asia"),
    "IN": dict(grid_ef=0.671, renew=0.240, aware_na=37.9, aware_un=36.2, fuel={"coal": 0.6, "gas": 0.1, "biomass": 0.3}, region="asia"),
    "BD": dict(grid_ef=0.696, renew=0.021, aware_na=6.88, aware_un=8.33, fuel={"gas": 0.9, "oil": 0.1}, region="asia"),
    "PK": dict(grid_ef=0.308, renew=0.470, aware_na=46.2, aware_un=47.0, fuel={"gas": 0.6, "coal": 0.2, "oil": 0.2}, region="asia"),
    "VN": dict(grid_ef=0.464, renew=0.453, aware_na=6.39, aware_un=11.0, fuel={"coal": 0.5, "biomass": 0.3, "oil": 0.2}, region="asia"),
    "TR": dict(grid_ef=0.476, renew=0.433, aware_na=25.1, aware_un=66.6, fuel={"gas": 0.8, "coal": 0.2}, region="eurasia"),
    "IT": dict(grid_ef=0.284, renew=0.488, aware_na=19.0, aware_un=46.4, fuel={"gas": 1.0}, region="eu"),
    "PT": dict(grid_ef=0.128, renew=0.809, aware_na=19.2, aware_un=42.5, fuel={"gas": 0.8, "biomass": 0.2}, region="eu"),
    "DE": dict(grid_ef=0.334, renew=0.585, aware_na=2.09, aware_un=2.31, fuel={"gas": 1.0}, region="eu"),
}
COUNTRY_SOURCE = "ember2026yearly; seitfudem2026data (AWARE 2.0)"

# CO2 emission factors of stationary combustion per MJ of fuel (IPCC 2006
# defaults, CO2 only; biomass CO2 reported as biogenic and excluded).
FUEL_EF = {"coal": 0.0946, "gas": 0.0561, "oil": 0.0774, "biomass": 0.0}
FUEL_SOURCE = "ipcc2006stationary"

# ---------------------------------------------------------------------------
# Fibres: cradle-to-gate climate impact (kg CO2e/kg fibre), blue water
# consumption (L/kg fibre), lot-to-lot variability (log-scale sigma).
# ---------------------------------------------------------------------------
MATERIALS = {
    # gwp: central value within verified literature ranges (manuscript Table A1); water: blue water L/kg
    "cotton_conv": dict(gwp=2.5, water=2000.0, sd=0.30, family="cellulosic", recycled=False, organic=False, staple=True),
    "cotton_org":  dict(gwp=1.1, water=2000.0, sd=0.35, family="cellulosic", recycled=False, organic=True, staple=True),
    "cotton_rec":  dict(gwp=0.6, water=4.0,   sd=0.25, family="cellulosic", recycled=True, organic=False, staple=True),
    "pes_virgin":  dict(gwp=4.5, water=19.7,  sd=0.12, family="polyester", recycled=False, organic=False, staple=False),
    "pes_rec":     dict(gwp=2.0, water=16.2,  sd=0.35, family="polyester", recycled=True, organic=False, staple=False),
    "pa_virgin":   dict(gwp=7.0, water=10.6,  sd=0.20, family="polyamide", recycled=False, organic=False, staple=False),
    "elastane":    dict(gwp=5.7, water=20.0,  sd=0.25, family="elastane", recycled=False, organic=False, staple=False),
    "viscose":     dict(gwp=3.0, water=30.0,  sd=0.35, family="cellulosic", recycled=False, organic=False, staple=True),
    "lyocell":     dict(gwp=0.8, water=20.0,  sd=0.40, family="cellulosic", recycled=False, organic=False, staple=True),
    "wool":        dict(gwp=20.0, water=100.0, sd=0.40, family="protein", recycled=False, organic=False, staple=True),
}
# country-specific blue water of cotton lint (L/kg), Textile Exchange (2026) Table 5-4; others use MATERIALS
FIBRE_WATER_COUNTRY = {("cotton_conv", "IN"): 3177.0, ("cotton_conv", "CN"): 1801.0, ("cotton_conv", "TR"): 1679.0,
                       ("cotton_org", "IN"): 3357.0, ("cotton_org", "TR"): 2418.0}
MATERIAL_SOURCE = ("textileexchange2026lifecotton; textileexchange2026lifepolyester; shen2010environmental; "
                   "vandervelden2014lca; feng2025carbon; wiedemann2015application; plasticseurope2022ecoprofile")

# Countries in which each fibre is sourced (weights), assumption.
FIBRE_ORIGINS = {
    "cotton_conv": {"IN": 0.4, "CN": 0.3, "PK": 0.2, "TR": 0.1},
    "cotton_org": {"IN": 0.7, "TR": 0.3},
    "cotton_rec": {"IN": 0.4, "PT": 0.3, "IT": 0.3},
    "pes_virgin": {"CN": 0.7, "IN": 0.2, "VN": 0.1},
    "pes_rec": {"CN": 0.5, "IT": 0.2, "TR": 0.3},
    "pa_virgin": {"CN": 0.6, "DE": 0.2, "IT": 0.2},
    "elastane": {"CN": 0.7, "TR": 0.3},
    "viscose": {"CN": 0.6, "IN": 0.4},
    "lyocell": {"DE": 0.5, "CN": 0.5},
    "wool": {"IT": 0.6, "TR": 0.4},
}

# ---------------------------------------------------------------------------
# Manufacturing processes (assumption unless stated; see code comments).
# ---------------------------------------------------------------------------
PROCESS = dict(
    # staple spinning electricity kWh/kg = a + b*Ne, spanning 1.3-6.8 kWh/kg (hasanbeigi2012review; sandin2019environmental)
    spin_staple_a=0.8, spin_staple_b=0.11,
    spin_filament=1.5,                 # texturing / filament processing (vandervelden2014lca: 1-1.7 kWh/kg)
    spin_waste_staple=(0.08, 0.14), spin_waste_filament=(0.005, 0.02),   # sandin2019environmental: 11 % / 0.5 %
    # fabric formation kWh/kg at 200 g/m2, scaled by (200/gsm)^0.5 (wovens 2.4-19.5, knits 0.13-1.2 kWh/kg)
    weave_ref=3.5, knit_ref=0.4, fabric_waste=(0.01, 0.06),
    wet_elec=(0.5, 1.5),               # kWh/kg (vandervelden2014lca: 0.36-1.25; sandin2019environmental: 0.7-0.8 per step)
    boiler_eff=(0.75, 0.90),           # assumption
    dryer_moisture=(0.5, 0.8),         # assumption
    dryer_eff=(0.40, 0.60),            # assumption
    latent_heat=2.6,                   # MJ per kg water evaporated incl. sensible heat (rounded)
    wet_loss=(0.02, 0.06),             # assumption
    # cut-make-trim electricity per garment (sandin2019environmental Table 3.5; beton2014environmental Table 48)
    cmt_kwh=dict(tshirt=0.29, jeans=1.31, sweater=0.29, jacket=3.92, dress=2.47),
    overhead=(0.05, 0.15),             # assumption
)

# Dye classes: liquor ratio (L/kg), number of baths, average temperature lift (K)
DYE_CLASSES = {
    # water consistent with roth2023best (batch dyeing of fabric 10-150 L/kg); salt 80-960 g/kg (reactive)
    "reactive": dict(lr=(6, 10), baths=(8, 12), dT=(40, 60), aux=(0.30, 0.90)),
    "disperse": dict(lr=(6, 10), baths=(5, 7), dT=(70, 95), aux=(0.05, 0.15)),
    "acid":     dict(lr=(8, 12), baths=(5, 7), dT=(60, 80), aux=(0.05, 0.12)),
    "vat_indigo": dict(lr=(2, 4), baths=(6, 10), dT=(20, 40), aux=(0.10, 0.30)),
}
DEPTH_OWF = {"light": (0.005, 0.015), "medium": (0.015, 0.035), "dark": (0.035, 0.08)}   # dyes 5-80 g/kg (roth2023best)
DEPTH_FACTOR = {"light": 0.85, "medium": 1.0, "dark": 1.2}   # scales baths/energy
FINISHES = {
    "none": dict(chem=(0.0, 0.005), pfas=False),
    "softener": dict(chem=(0.01, 0.03), pfas=False),
    "enzyme_wash": dict(chem=(0.02, 0.05), pfas=False),
    "wr_fluorinated": dict(chem=(0.03, 0.06), pfas=True),
    "wr_fluorine_free": dict(chem=(0.03, 0.06), pfas=False),
}

# Restricted-substance classes: illustrative limits (mg/kg) and base
# occurrence probabilities. SVHC threshold of 0.1 % w/w (1000 mg/kg) follows
# REACH Article 33; all other limits are illustrative.
SUBSTANCES = {
    "pfas": dict(limit=25.0, p_base=0.0),
    "arylamines": dict(limit=30.0, p_base=0.02),
    "formaldehyde": dict(limit=75.0, p_base=0.03),
    "apeo": dict(limit=100.0, p_base=0.03),
}
SVHC_THRESHOLD_MG_KG = 1000.0

# ---------------------------------------------------------------------------
# Transport emission factors (kg CO2e per tonne-km) and illustrative distances.
# ---------------------------------------------------------------------------
TRANSPORT_EF = {"sea": 0.01977, "road": 0.09502, "rail": 0.03470, "air": 0.66646}   # DESNZ 2025 WTW, kg CO2e/tkm
TRANSPORT_SOURCE = "desnz2025greenhouse"
REGION_DIST = {  # (km_sea, km_road) between regions, assumption
    ("asia", "asia"): (3000, 600),
    ("asia", "eurasia"): (9000, 800),
    ("asia", "eu"): (18000, 900),
    ("eurasia", "eu"): (3500, 1500),
    ("eurasia", "eurasia"): (0, 500),
    ("eu", "eu"): (0, 1200),
}
AIR_SHARE = 0.03  # share of garment shipments by air (assumption)

# ---------------------------------------------------------------------------
# Product categories (assumption).
# ---------------------------------------------------------------------------
CATEGORIES = {
    # mass (kg): beton2014environmental Annex 1 / sandin2019environmental; cut: beton2014environmental Table 18
    "tshirt": dict(route="knit", gsm=(140, 200), mass=(0.12, 0.25), ne=(24, 40), cut=(0.11, 0.17),
                   comps=[{"cotton_conv": 1.0}, {"cotton_org": 1.0}, {"cotton_conv": 0.6, "pes_virgin": 0.4},
                          {"cotton_rec": 0.3, "cotton_conv": 0.7}, {"pes_rec": 0.5, "cotton_org": 0.5}],
                   finishes=["softener", "none"]),
    "jeans": dict(route="woven", gsm=(300, 450), mass=(0.45, 0.70), ne=(8, 14), cut=(0.12, 0.18),
                  comps=[{"cotton_conv": 0.98, "elastane": 0.02}, {"cotton_org": 0.98, "elastane": 0.02},
                         {"cotton_conv": 0.78, "cotton_rec": 0.20, "elastane": 0.02},
                         {"cotton_conv": 0.70, "pes_rec": 0.28, "elastane": 0.02}],
                  finishes=["enzyme_wash"]),
    "sweater": dict(route="knit", gsm=(250, 400), mass=(0.30, 0.60), ne=(10, 20), cut=(0.08, 0.14),
                    comps=[{"wool": 1.0}, {"wool": 0.5, "pa_virgin": 0.5}, {"cotton_conv": 1.0},
                           {"viscose": 0.5, "pa_virgin": 0.5}],
                    finishes=["softener", "none"]),
    "jacket": dict(route="woven", gsm=(120, 250), mass=(0.43, 0.80), ne=(0, 0), cut=(0.13, 0.20),
                   comps=[{"pes_virgin": 1.0}, {"pes_rec": 1.0}, {"pa_virgin": 1.0}, {"pa_virgin": 0.5, "pes_rec": 0.5}],
                   finishes=["wr_fluorinated", "wr_fluorine_free"]),
    "dress": dict(route="woven", gsm=(100, 180), mass=(0.25, 0.45), ne=(30, 50), cut=(0.15, 0.22),
                  comps=[{"viscose": 1.0}, {"lyocell": 1.0}, {"viscose": 0.95, "elastane": 0.05}, {"pes_virgin": 1.0}],
                  finishes=["softener", "none"]),
}

# Countries hosting each manufacturing role (weights), assumption.
ROLE_COUNTRIES = {
    "spinning": {"CN": 0.3, "IN": 0.2, "PK": 0.1, "BD": 0.1, "VN": 0.1, "TR": 0.1, "IT": 0.05, "PT": 0.05},
    "fabric": {"CN": 0.3, "IN": 0.15, "BD": 0.15, "PK": 0.1, "VN": 0.1, "TR": 0.1, "IT": 0.05, "PT": 0.05},
    "wet": {"CN": 0.3, "IN": 0.15, "BD": 0.15, "PK": 0.1, "VN": 0.1, "TR": 0.1, "IT": 0.05, "PT": 0.05},
    "garment": {"CN": 0.25, "BD": 0.25, "VN": 0.15, "IN": 0.1, "TR": 0.15, "PT": 0.05, "IT": 0.05},
}

# End-of-life scenario per category (fractions of collected mass): assumption
EOL = dict(   # huygens2023technoscientific; kohler2021circular; eea2024management (ranges, simplified)
    collection_rate=(0.12, 0.38),     # share of products separately collected
    reuse_share=(0.50, 0.70),         # of collected (incl. export for reuse)
    recycling_share=(0.20, 0.30),     # of collected
    recycling_yield=(0.55, 0.85),     # mechanical recycling process yield (textileexchange2026lifecotton: 1/1.789)
)

# ---------------------------------------------------------------------------
# Reporting behaviour of facilities (data-generation model).
# ---------------------------------------------------------------------------
REPORTING = dict(
    metering_levels={"lot": 0.35, "facility": 0.40, "none": 0.25},
    measured_noise=0.03,          # log-sd of metered lot values
    estimate_bias_sd=0.25,        # log-sd of supplier estimates around truth
    estimate_mean_bias=0.10,      # supplier estimates tend to be optimistic (under-report by ~10 %)
    verification_levels={"third_party": 0.25, "second_party": 0.20, "self_declared": 0.40, "none": 0.15},
    renewable_evidence_prob=0.6,  # probability that on-site/contracted renewables are evidenced
)

# ---------------------------------------------------------------------------
# Secondary ("default") data used by baseline methods: generic process
# intensities at reference conditions (as a generic database would provide).
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    spin_kwh_per_kg=3.5,            # kazan2020life: derived 3.64 kWh/kg
    weave_kwh_per_kg=3.5,
    knit_kwh_per_kg=0.5,
    wet_elec_kwh_per_kg=1.0,
    wet_heat_mj_per_kg=25.0,
    wet_water_l_per_kg=100.0,       # roth2023best: 100-150 L/kg conventional jet reactive dyeing
    garment_kwh_per_kg=3.0,         # sandin2019environmental: 2.4-8.8 kWh/kg
    world_grid_ef=0.4605,           # ember2026yearly, world 2025
    default_heat_ef=0.0561 / 0.85,  # natural gas boiler
)
