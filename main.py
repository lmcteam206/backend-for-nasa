# server.py
import math
import random
import hashlib
import requests
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Meteor Impact Detailed API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- defaults & constants ---
DEFAULT_DENSITY = 3300  # kg/m3 (chondritic rock). Use 7870 for iron, etc.
EARTH_GRAVITY = 9.81  # m/s2
JOULES_PER_KT_TNT = 4.184e12
JOULES_PER_MT_TNT = 4.184e15
RHO_AIR = 1.225  # kg/m3 (sea level)
SPEED_OF_SOUND = 340  # m/s

# --- deterministic RNG ---
def seeded_random(lat, lon, angle_deg, diameter_m, velocity_km_s):
    key = f"{lat:.6f},{lon:.6f},{angle_deg:.2f},{float(diameter_m) if diameter_m else 'none'},{float(velocity_km_s) if velocity_km_s else 'none'}"
    h = hashlib.md5(key.encode()).hexdigest()
    seed = int(h[:8], 16)
    rnd = random.Random(seed)
    return rnd

# --- physics helpers ---
def volume_sphere(diameter_m):
    r = diameter_m / 2.0
    return (4.0 / 3.0) * math.pi * r**3

def mass_from_density(volume_m3, density_kg_m3):
    return volume_m3 * density_kg_m3

def kinetic_energy_j(mass_kg, velocity_m_s):
    return 0.5 * mass_kg * velocity_m_s**2

def approximate_airburst_altitude(diameter_m, velocity_m_s, angle_deg, density_kg_m3):
    base = 50000.0 / (math.sqrt(diameter_m) + 0.1)
    vel_factor = (velocity_m_s / 20000.0)
    density_factor = (density_kg_m3 / 3300.0)
    angle_factor = max(0.2, math.sin(math.radians(angle_deg)))
    alt = base * vel_factor / density_factor * angle_factor
    return max(50.0, min(120000.0, alt))

def crater_diameter_km(diameter_m, velocity_m_s, density_kg_m3):
    return 0.0013 * (diameter_m ** 0.78) * (velocity_m_s ** 0.44) * (density_kg_m3 / 3300.0) / 1000.0

def seismic_magnitude_from_energy(energy_j):
    try:
        return max(-1.0, round(math.log10(max(1.0, energy_j)) - 4.8, 2))
    except:
        return None

def blast_radius_by_overpressure(energy_j, overpressure_pa):
    if energy_j <= 0:
        return 0.0
    e_kt = energy_j / JOULES_PER_KT_TNT
    base_km = (e_kt ** (1/3.0)) * 1.0
    psi = overpressure_pa / 101325.0
    mult = 1.0 / (1.2 * (psi ** 0.3))
    return max(0.01, base_km * mult)

def thermal_radius_from_energy(energy_j):
    mt = energy_j / JOULES_PER_MT_TNT
    return max(0.01, (mt ** (1/4.0)) * 10.0)  # km

def tsunami_initial_height_m(energy_j, water_depth_m):
    if water_depth_m <= 0:
        return 0.0
    h = (energy_j ** 0.25) / (1e4 * (water_depth_m ** 0.25))
    return max(0.0, min(500.0, h))

# --- population lookup ---
# --- simplified population approximation ---
def fetch_population_nearby(lat, lon, radius_km, rnd=None):
    """
    Quick deterministic population estimate based on latitude, longitude, and radius.
    No external API calls, very fast.
    """
    if rnd is None:
        # Use a deterministic seed based on location
        seed = int((lat + lon) * 100000) & 0xffffffff
        rnd = random.Random(seed)
    
    # Base population density decreases from equator to poles
    lat_factor = max(0.1, math.cos(math.radians(lat))**2)  # 0.1..1
    base_density_per_km2 = 50.0 * lat_factor  # average density per km2
    
    # Small random variation for realism
    noise = 0.7 + rnd.random() * 0.6  # 0.7..1.3
    effective_density = base_density_per_km2 * noise
    
    # Population in the circular area
    area_km2 = math.pi * (radius_km ** 2)
    pop = effective_density * area_km2
    return max(0, int(round(pop)))



# --- main endpoint ---
@app.get("/impact_realistic")
def impact_realistic(
    lat: float = Query(..., description="latitude"),
    lon: float = Query(..., description="longitude"),
    angle_deg: float = Query(..., description="entry angle degrees (0=skimming,90=vertical)"),
    diameter_m: float = Query(None, description="diameter in meters (optional)"),
    velocity_km_s: float = Query(None, description="velocity in km/s (optional)"),
    density_kg_m3: float = Query(DEFAULT_DENSITY, description="material density (kg/m3)"),
    population_api_url: str = Query(None, description="optional population API URL"),
    population_api_key: str = Query(None, description="optional API key")
):
    rnd = seeded_random(lat, lon, angle_deg, diameter_m, velocity_km_s)
    if diameter_m is None:
        diameter_m = rnd.uniform(5.0, 300.0)
    if velocity_km_s is None:
        velocity_km_s = rnd.uniform(11.0, 72.0)
    velocity_m_s = velocity_km_s * 1000.0

    vol_m3 = volume_sphere(diameter_m)
    mass_kg = mass_from_density(vol_m3, density_kg_m3)
    energy_j = kinetic_energy_j(mass_kg, velocity_m_s)
    energy_kt = energy_j / JOULES_PER_KT_TNT
    energy_mt = energy_j / JOULES_PER_MT_TNT

    airburst_alt_m = approximate_airburst_altitude(diameter_m, velocity_m_s, angle_deg, density_kg_m3)
    will_airburst = airburst_alt_m > 1000.0 and diameter_m < 500.0
    luminous_energy_j = energy_j * 0.001 * (diameter_m / 100.0)
    sonic_boom_radius_km = max(0.1, (velocity_m_s / SPEED_OF_SOUND) * (diameter_m / 10.0))

    crater_km = crater_diameter_km(diameter_m, velocity_m_s, density_kg_m3)
    crater_depth_m = max(1.0, crater_km * 1000 * 0.2)
    seismic_mag = seismic_magnitude_from_energy(energy_j)
    ejecta_volume_km3 = max(0.0, (vol_m3 / 1e9) * (0.1 + (density_kg_m3 / 10000.0)))

    psi_values = [0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20]
    overpressure_pa_values = [p * 6894.76 for p in psi_values]
    blast_rings = {}
    for p_psi, p_pa in zip(psi_values, overpressure_pa_values):
        d_km = round(blast_radius_by_overpressure(energy_j, p_pa), 3)
        blast_rings[f"{p_psi}psi_km"] = d_km

    thermal_km = round(thermal_radius_from_energy(energy_j), 3)
    peak_overpressure_pa = (energy_j ** 0.2) / 100000.0
    peak_overpressure_psi = peak_overpressure_pa / 6894.76

    near_coast_score = ((abs(lat) + abs(lon)) % 10) / 10.0
    is_ocean_impact = rnd.random() < 0.4
    water_depth_m = 4000.0 if is_ocean_impact else 0.0
    tsunami_initial_m = tsunami_initial_height_m(energy_j, water_depth_m)
    tsunami_max_coastal_m = tsunami_initial_m * (1.0 + rnd.uniform(0.2, 10.0))
    tsunami_inundation_km = min(200.0, tsunami_max_coastal_m * 2.0 * (1.0 + rnd.random()))

    radii_km_to_sample = [0.5, 1, 3, 5, 10, 20, 50]
    populations = {}
    for r in radii_km_to_sample:
        p = fetch_population_nearby(lat, lon, r, rnd=rnd)
        populations[f"pop_within_{r}km"] = int(round(p))

    casualties = {}
    for key, ring_km in blast_rings.items():
        psi_label = float(key.replace("psi_km", ""))
        if psi_label >= 10: frac = 0.9
        elif psi_label >= 5: frac = 0.6
        elif psi_label >= 2: frac = 0.3
        elif psi_label >= 1: frac = 0.15
        elif psi_label >= 0.5: frac = 0.05
        else: frac = 0.01
        candidate_radii = sorted([0.5,1,3,5,10,20,50], key=lambda x: abs(x - (ring_km if ring_km>0 else 0.5)))
        found_pop = None
        for cr in candidate_radii:
            k = f"pop_within_{cr}km"
            if k in populations:
                found_pop = populations[k]
                break
        if found_pop is None:
            found_pop = populations.get("pop_within_50km", 0)
        estimated_deaths = int(round(found_pop * frac))
        casualties[f"deaths_in_{key}"] = estimated_deaths

    pop_in_thermal = populations.get("pop_within_10km", 0)
    thermal_death_frac = min(0.9, 0.02 + (energy_mt ** 0.1))
    thermal_deaths = int(round(pop_in_thermal * thermal_death_frac))

    pop_in_tsunami_zone = populations.get("pop_within_50km", 0) if is_ocean_impact else 0
    if tsunami_max_coastal_m <= 0.5:
        tsunami_deaths = int(round(pop_in_tsunami_zone * 0.001))
    elif tsunami_max_coastal_m <= 2.0:
        tsunami_deaths = int(round(pop_in_tsunami_zone * 0.02))
    else:
        tsunami_deaths = int(round(pop_in_tsunami_zone * 0.25))

    total_deaths_est = sum(v for v in casualties.values())
    total_deaths_est += thermal_deaths + tsunami_deaths
    low = int(max(0, total_deaths_est * 0.5))
    med = int(max(0, total_deaths_est))
    high = int(max(0, total_deaths_est * 2.5))

    injuries_low = int(low * 1.5)
    injuries_med = int(med * 2.0)
    injuries_high = int(high * 3.0)

    damage_index = min(1.0, math.log10(energy_j + 1) / 10.0)
    buildings_destroyed_percent = round(100.0 * damage_index * (0.2 + rnd.random() * 0.8), 1)
    roads_destroyed_km = int(round(50 * damage_index * (1 + rnd.random()*4)))
    bridges_destroyed = int(round(5 * damage_index * (1 + rnd.random()*6)))
    airports_destroyed = int(round(1 * damage_index * (rnd.random()>0.7 and 1 or 0)))

    soot_megatonnes = round((energy_mt ** 0.6) * (0.01 + rnd.random()*0.2), 3)
    dust_megatonnes = round((energy_mt ** 0.5) * (0.01 + rnd.random()*0.3), 3)
    global_temp_drop_c = round(min(5.0, 0.1 * soot_megatonnes ** 0.6), 3)

    local_pop_50km = populations.get("pop_within_50km", 0)
    GDP_per_capita = 10000.0
    economic_loss_usd = int(round(local_pop_50km * GDP_per_capita * damage_index * (1 + rnd.random()*2)))
    recovery_years = int(round(1 + (10 * damage_index) + (global_temp_drop_c * 2)))

    out = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request": {"lat": lat, "lon": lon, "angle_deg": angle_deg, "diameter_m": diameter_m, "velocity_km_s": velocity_km_s, "density_kg_m3": density_kg_m3},
            "notes": "Deterministic approx 100+ field summary."
        },
        "physics": {
            "volume_m3": vol_m3,
            "mass_kg": mass_kg,
            "kinetic_energy_j": energy_j,
            "kinetic_energy_kt_tnt": energy_kt,
            "kinetic_energy_mt_tnt": energy_mt,
            "luminous_energy_j": luminous_energy_j,
            "entry_velocity_m_s": velocity_m_s,
            "entry_velocity_km_s": velocity_km_s,
            "entry_angle_deg": angle_deg,
            "airburst_altitude_m": airburst_alt_m,
            "will_airburst": bool(will_airburst),
            "sonic_boom_radius_km": round(sonic_boom_radius_km, 3)
        },
        "crater": {
            "crater_diameter_km": round(crater_km, 4),
            "crater_depth_m": round(crater_depth_m, 1),
            "ejecta_volume_km3": round(ejecta_volume_km3, 4)
        },
        "blast": {
            "peak_overpressure_pa": round(peak_overpressure_pa, 3),
            "peak_overpressure_psi": round(peak_overpressure_psi, 3),
            "blast_rings_km": blast_rings,
            "thermal_radius_km": thermal_km
        },
        "seismic": {
            "seismic_magnitude": seismic_mag,
            "seismic_radius_km": round(max(1.0, seismic_mag * 50 if seismic_mag else 0.0), 1)
        },
        "tsunami": {
            "is_ocean_impact_estimate": bool(is_ocean_impact),
            "water_depth_m_used": water_depth_m,
            "tsunami_initial_offshore_m": round(tsunami_initial_m, 3),
            "tsunami_max_coastal_m": round(tsunami_max_coastal_m, 3),
            "tsunami_inundation_km": round(tsunami_inundation_km, 2),
            "tsunami_deaths_estimate": int(tsunami_deaths)
        },
        "population_samples": populations,
        "casualties": {
            "deaths_by_blast_ring": casualties,
            "thermal_deaths_estimate": int(thermal_deaths),
            "tsunami_deaths_estimate": int(tsunami_deaths),
            "deaths_estimate_low": int(low),
            "deaths_estimate_med": int(med),
            "deaths_estimate_high": int(high),
            "injuries_estimate_low": int(injuries_low),
            "injuries_estimate_med": int(injuries_med),
            "injuries_estimate_high": int(injuries_high)
        },
        "infrastructure": {
            "buildings_destroyed_percent": buildings_destroyed_percent,
            "roads_destroyed_km": roads_destroyed_km,
            "bridges_destroyed": bridges_destroyed,
            "airports_destroyed": airports_destroyed,
        },
        "environmental": {
            "soot_megatonnes": soot_megatonnes,
            "dust_megatonnes": dust_megatonnes,
            "global_temp_drop_c": global_temp_drop_c,
            "ozone_loss_percent_estimate": round(min(100, soot_megatonnes * 0.05), 2)
        },
        "economy_recovery": {
            "economic_loss_usd": economic_loss_usd,
            "estimated_recovery_years": recovery_years
        },
        "debug_and_assumptions": {
            "population_api_used": bool(population_api_url),
            "population_api_url": population_api_url,
            "population_api_key_provided": bool(population_api_key),
            "notes": "Approximate model for visualization/testing."
        },
        "extras": {
            "fragmentation_index": round( max(0.0, min(1.0, rnd.random() * (100.0 / (diameter_m + 1.0)))), 3),
            "luminous_duration_s": round(max(0.1, math.log10(max(1.0, energy_j)) * 0.05 + rnd.random()*2.0), 2),
            "shockwave_duration_s": round(max(0.1, (blast_rings.get("0.1psi_km",0.01) * 1000) / (velocity_m_s + 1.0)), 2),
            "fireball_radius_km": round(min(thermal_km*0.2, crater_km*0.5 + rnd.random()*1.0), 3),
            "fallout_radius_km": round(max(1.0, blast_rings.get("0.1psi_km", 1.0) * (1.0 + rnd.random())), 2),
            "air_ionization_index": round(min(100.0, math.log10(max(1, energy_j)) * 2.0 + rnd.random()*5.0), 2),
            "number_of_secondary_fires_est": int(round((local_pop_50km / 10000.0) * (0.1 + rnd.random()*2.0))),
            "water_contamination_index": round(min(100.0, dust_megatonnes * 0.3 + rnd.random()*5.0), 2)
        }
    }

    flat_count = sum(len(v) if isinstance(v, dict) else 1 for v in out.values())
    out["meta"]["approx_output_field_count"] = flat_count + sum(len(v) for v in out["extras"].keys())

    return out

@app.get("/TENheads")
def TENheads():
    NASA_API_KEY = "DEMO_KEY"  # replace with your key
    NEO_API_URL = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={NASA_API_KEY}&size=10"

    try:
        r = requests.get(NEO_API_URL, timeout=10)
        r.raise_for_status()
        neos = r.json().get("near_earth_objects", [])
    except Exception as e:
        return {"error": f"Failed to fetch NEOs: {str(e)}"}

    results = []

    for neo in neos:
        # Pick first close approach data if available
        cad = neo.get("close_approach_data")
        if cad and len(cad) > 0:
            cad0 = cad[0]
            lat = float(cad0.get("miss_distance", {}).get("astronomical", 0)) * 149597870.7  # AU to km approx
            lon = random.uniform(-180, 180)  # NEO API doesn't give lat/lon; randomize for simulation
            diameter_m = (
                (neo.get("estimated_diameter", {})
                 .get("meters", {})
                 .get("estimated_diameter_min", 5) +
                 neo.get("estimated_diameter", {})
                 .get("meters", {})
                 .get("estimated_diameter_max", 10)) / 2
            )
        else:
            lat = random.uniform(-90, 90)
            lon = random.uniform(-180, 180)
            diameter_m = random.uniform(5.0, 300.0)

        velocity_km_s = float(cad0.get("relative_velocity", {}).get("kilometers_per_second", 20)) if cad else random.uniform(11.0, 72.0)
        angle_deg = random.uniform(20, 80)  # random entry angle
        density_kg_m3 = DEFAULT_DENSITY

        # Call your impact_realistic function directly
        impact = impact_realistic(
            lat=lat,
            lon=lon,
            angle_deg=angle_deg,
            diameter_m=diameter_m,
            velocity_km_s=velocity_km_s,
            density_kg_m3=density_kg_m3
        )

        results.append({
            "name": neo.get("name", "Unnamed NEO"),
            "id": neo.get("id"),
            "impact_data": impact
        })

    return results
@app.get("/impact_by_name")
def impact_by_name(name: str):
    NASA_API_KEY = "DEMO_KEY"  # Replace with your actual key
    NEO_API_URL = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={NASA_API_KEY}&size=50" 

    try:
        r = requests.get(NEO_API_URL, timeout=10)
        r.raise_for_status()
        neos = r.json().get("near_earth_objects", [])
    except Exception as e:
        return {"error": f"Failed to fetch NEOs: {str(e)}"}

    # Search for the NEO by name
    neo = next((n for n in neos if n.get("name", "").lower() == name.lower()), None)
    if not neo:
        return {"error": f"NEO with name '{name}' not found."}

    # Pick first close approach data if available
    cad = neo.get("close_approach_data")
    if cad and len(cad) > 0:
        cad0 = cad[0]
        lat = float(cad0.get("miss_distance", {}).get("astronomical", 0)) * 149597870.7  # AU to km approx
        lon = random.uniform(-180, 180)  # NEO API doesn't provide lat/lon
        diameter_m = (
            (neo.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_min", 5) +
             neo.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max", 10)) / 2
        )
        velocity_km_s = float(cad0.get("relative_velocity", {}).get("kilometers_per_second", 20))
    else:
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        diameter_m = random.uniform(5.0, 300.0)
        velocity_km_s = random.uniform(11.0, 72.0)

    angle_deg = random.uniform(20, 80)
    density_kg_m3 = DEFAULT_DENSITY

    # Generate impact data
    impact = impact_realistic(
        lat=lat,
        lon=lon,
        angle_deg=angle_deg,
        diameter_m=diameter_m,
        velocity_km_s=velocity_km_s,
        density_kg_m3=density_kg_m3
    )

    return {
        "name": neo.get("name", "Unnamed NEO"),
        "id": neo.get("id"),
        "impact_data": impact
    }

# uvicorn server3:app --reload --host 127.0.0.1 --port 8000
