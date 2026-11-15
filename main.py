#!/usr/bin/env python3
"""MagLevX - maglev train levitation transport lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Simulation only: the vehicle flagship that grows out of LevitationX
physics - EMS vs EDS suspension at train scale, linear-motor
propulsion to 430 km/h, aerodynamic drag and energy per passenger-km,
service + emergency braking, guideway switching and safe headway,
X-family speed governor zones, departure interlocks, curve ride
quality, a TinyML kNN regime classifier, Gemini Carbon 4 as DECIDER
(names-only, tighten-only), Muse as advisor (never overrides
HOLD), and Google Flow plan-only seats.

Every live-operation / real-hardware action is a permanent HOLD gate.
"""
import hashlib
import json
import math
import sys

# ---------------------------------------------------------------- constants
MU0 = 4.0 * math.pi * 1e-7          # vacuum permeability, H/m
G = 9.80665                         # standard gravity, m/s^2
RHO_AIR = 1.225                     # kg/m^3

VERSION = "1.0.0"

# Reference train (lab model, Transrapid-class numbers, SIM ONLY)
TRAIN = {
    "mass_kg": 240000.0,            # 4-car trainset
    "length_m": 100.0,
    "frontal_area_m2": 12.5,
    "cd": 0.24,                     # drag coefficient, streamlined nose
    "bogies": 16,
    "seats": 360,
    "vmax_kmh": 430.0,
    "levitation_kw_per_tonne": 1.0,  # EMS rule of thumb (LevitationX)
    "aux_power_kw": 150.0,
    "propulsion_efficiency": 0.90,
}

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "live_train_operation": "HOLD - no operation of any real train, ever",
    "real_hardware_control": "HOLD - no real magnets, motors, or guideway hardware control",
    "passenger_service_live": "HOLD - no live passenger service; nobody rides a simulation",
    "bypass_safety_interlock": "HOLD - safety interlocks are never bypassed",
    "exceed_speed_governor_live": "HOLD - the speed governor is never exceeded on live track",
    "guideway_switching_live": "HOLD - no switching of a real guideway beam",
    "emergency_brake_disable": "HOLD - emergency braking is never disabled",
    "claim_certified_system": "HOLD - no claim this simulation is a certified rail system",
    "commercial_service_claim": "HOLD - no commercial service claims from a lab model",
    "live_guideway_construction": "HOLD - no live guideway construction guidance",
    "high_voltage_propulsion_live": "HOLD - no live high-voltage propulsion work",
    "cryogenics_live_handling": "HOLD - no live cryogen handling for EDS superconducting magnets",
    "evacuation_drill_live": "HOLD - no live evacuation drills directed from a simulation",
    "freight_hazmat_live": "HOLD - no live hazmat freight planning on a passenger lab model",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. Suspension
# EMS (attractive, Transrapid style) at train scale: many magnet modules
# share the load. Same force law as LevitationX, vehicle-scale constants:
#   F_module = MU0 * N^2 * A * I^2 / (4 * gap^2)
EMS_N = 250.0          # coil turns per module
EMS_A = 0.35           # pole face area per module, m^2
EMS_MODULES = 128      # levitation magnet modules on the trainset
EMS_GAP_M = 0.010      # 10 mm working gap


def ems_module_force(current_a, gap_m):
    """Attractive force of one EMS module in newtons."""
    if gap_m <= 0:
        raise ValueError("gap must be positive")
    return MU0 * EMS_N ** 2 * EMS_A * current_a ** 2 / (4.0 * gap_m ** 2)


def ems_train_lift(current_a, gap_m=EMS_GAP_M):
    """Total EMS lift of the whole trainset in newtons."""
    return EMS_MODULES * ems_module_force(current_a, gap_m)


def ems_module_load_n(mass_kg=TRAIN["mass_kg"]):
    """Weight each module must carry: mg / modules."""
    return mass_kg * G / EMS_MODULES


def ems_equilibrium_current(mass_kg=TRAIN["mass_kg"], gap_m=EMS_GAP_M):
    """Per-module current so total EMS lift exactly balances weight."""
    k = MU0 * EMS_N ** 2 * EMS_A / 4.0
    return gap_m * math.sqrt(ems_module_load_n(mass_kg) / k)


# EDS (repulsive, superconducting JR-maglev style): force is zero at
# standstill and rises with speed; wheels carry the train below liftoff.
#   F = K_EDS * (1 - exp(-v / V0)) * (GAP_REF / gap)
EDS_K = 6.5e6          # N at full speed factor, reference gap
EDS_V0 = 40.0          # m/s speed scale
EDS_GAP_REF = 0.10     # m, EDS rides on a ~10 cm gap


def eds_train_lift(speed_mps, gap_m=EDS_GAP_REF):
    """Total repulsive EDS lift of the trainset in newtons."""
    if gap_m <= 0:
        raise ValueError("gap must be positive")
    if speed_mps < 0:
        raise ValueError("speed must be >= 0")
    return EDS_K * (1.0 - math.exp(-speed_mps / EDS_V0)) * (EDS_GAP_REF / gap_m)


def eds_liftoff_speed(mass_kg=TRAIN["mass_kg"], gap_m=EDS_GAP_REF):
    """Speed where EDS lift first equals weight (bisection, deterministic)."""
    target = mass_kg * G
    lo, hi = 0.0, 400.0
    if eds_train_lift(hi, gap_m) < target:
        return -1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if eds_train_lift(mid, gap_m) < target:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 4)


# EDS magnetic (levitation) drag: rises to a peak at low speed, then
# falls off - the tax EDS pays before aerodynamics take over.
EDS_DRAG_PEAK_N = 18000.0
EDS_DRAG_VPEAK = 30.0  # m/s


def eds_magnetic_drag(speed_mps):
    """Magnetic drag force (N): peak * (v/vp) * exp(1 - v/vp)."""
    if speed_mps < 0:
        raise ValueError("speed must be >= 0")
    if speed_mps == 0:
        return 0.0
    x = speed_mps / EDS_DRAG_VPEAK
    return EDS_DRAG_PEAK_N * x * math.exp(1.0 - x)


def suspension_comparison(mass_kg=TRAIN["mass_kg"]):
    """EMS vs EDS for the same trainset - the honest trade table."""
    return {
        "train_mass_kg": mass_kg,
        "weight_n": round(mass_kg * G, 1),
        "ems": {
            "principle": "attractive - train wraps the guideway, magnets pull up",
            "gap_m": EMS_GAP_M,
            "lift_at_standstill_n": round(ems_train_lift(ems_equilibrium_current(mass_kg)), 1),
            "needs": "fast gap feedback (unstable open-loop, Earnshaw class)",
            "wheels_below_liftoff": False,
            "levitation_power_kw": round(mass_kg / 1000.0 * TRAIN["levitation_kw_per_tonne"], 1),
        },
        "eds": {
            "principle": "repulsive - superconducting magnets push off guideway coils",
            "gap_m": EDS_GAP_REF,
            "lift_at_standstill_n": 0.0,
            "liftoff_speed_mps": eds_liftoff_speed(mass_kg),
            "liftoff_speed_kmh": round(eds_liftoff_speed(mass_kg) * 3.6, 1),
            "needs": "speed - no lift until the train is moving",
            "wheels_below_liftoff": True,
            "magnetic_drag_peak_n": EDS_DRAG_PEAK_N,
        },
        "gate": gate("live_train_operation"),
    }


# ================================================================ 2. Propulsion
# Linear synchronous motor: thrust accelerates the train against drag.
A_MAX = 0.9            # m/s^2 service acceleration (maglev-class)
JERK_MAX = 0.5         # m/s^3 ride-comfort jerk limit


def kmh_to_mps(kmh):
    return kmh / 3.6


def mps_to_kmh(mps):
    return mps * 3.6


def accel_time_to_speed(speed_mps, accel=A_MAX):
    """Constant-acceleration time from rest: t = v / a."""
    return speed_mps / accel


def accel_distance(speed_mps, accel=A_MAX):
    """Constant-acceleration distance from rest: d = v^2 / (2a)."""
    return speed_mps ** 2 / (2.0 * accel)


def thrust_for_accel(mass_kg, accel, speed_mps):
    """Motor thrust (N) = m*a + aerodynamic drag at that speed."""
    return mass_kg * accel + aero_drag_force(speed_mps)


def propulsion_profile(distance_m=430000.0, vmax_mps=None, a_max=A_MAX,
                       jerk=JERK_MAX, dt=0.1):
    """Jerk-limited trapezoidal run over a route. Deterministic sim.

    State machine: accelerate (accel ramps at jerk limit) -> cruise ->
    brake so the train stops exactly at the destination. Returns
    travel time, top speed, and distance covered.
    """
    if vmax_mps is None:
        vmax_mps = kmh_to_mps(TRAIN["vmax_kmh"])
    v, a, x, t = 0.0, 0.0, 0.0, 0.0
    top = 0.0
    while True:
        # distance needed to stop from current v at a_max (with jerk ramp)
        stop_dist = v * v / (2.0 * a_max) + v * (a_max / jerk) * 0.5
        remaining = distance_m - x
        if remaining <= 0.01 and v <= 0.05:
            break
        if stop_dist >= remaining:
            target_a = -a_max
        elif v < vmax_mps:
            target_a = a_max
        else:
            target_a = 0.0
        # jerk-limit the acceleration change
        da = target_a - a
        max_da = jerk * dt
        a += max(-max_da, min(max_da, da))
        v = max(0.0, v + a * dt)
        if v > vmax_mps:
            v = vmax_mps
        x += v * dt
        t += dt
        top = max(top, v)
        if t > 20000:  # safety bound, never reached on sane inputs
            break
    return {
        "distance_m": round(x, 1),
        "route_m": distance_m,
        "travel_time_s": round(t, 1),
        "travel_time_min": round(t / 60.0, 2),
        "top_speed_mps": round(top, 3),
        "top_speed_kmh": round(mps_to_kmh(top), 1),
        "a_max_mps2": a_max,
        "jerk_max_mps3": jerk,
    }


# ================================================================ 3. Aero + energy
def aero_drag_force(speed_mps):
    """Aerodynamic drag: F = 0.5 * rho * Cd * A * v^2."""
    return 0.5 * RHO_AIR * TRAIN["cd"] * TRAIN["frontal_area_m2"] * speed_mps ** 2


def cruise_power_kw(speed_mps, method="eds"):
    """Total wall power at steady cruise (kW).

    Mechanical power = (aero drag + magnetic drag) * v, divided by
    propulsion efficiency, plus levitation and auxiliary power.
    """
    mag = eds_magnetic_drag(speed_mps) if method == "eds" else 0.0
    mech_kw = (aero_drag_force(speed_mps) + mag) * speed_mps / 1000.0
    lev_kw = TRAIN["mass_kg"] / 1000.0 * TRAIN["levitation_kw_per_tonne"]
    return mech_kw / TRAIN["propulsion_efficiency"] + lev_kw + TRAIN["aux_power_kw"]


def energy_per_passenger_km(speed_mps, method="eds", seats=TRAIN["seats"]):
    """kWh per passenger-km at cruise = power / (speed * passengers)."""
    return cruise_power_kw(speed_mps, method) / (mps_to_kmh(speed_mps) * seats)


# Reference figures - LABELED ESTIMATES for comparison only, not measured.
ENERGY_REFERENCE = {
    "maglevx_model_430kmh": None,  # filled from the model in the report fn
    "high_speed_rail_est": 0.035,
    "car_petrol_per_person_est": 0.12,
    "short_haul_air_est": 0.18,
    "note": "ESTIMATES for classroom comparison - not measured data",
}


def energy_comparison(speed_mps=None):
    if speed_mps is None:
        speed_mps = kmh_to_mps(TRAIN["vmax_kmh"])
    out = dict(ENERGY_REFERENCE)
    out["maglevx_model_430kmh"] = round(energy_per_passenger_km(speed_mps), 4)
    out["units"] = "kWh per passenger-km"
    return out


# ================================================================ 4. Braking
SERVICE_DECEL = 0.8    # m/s^2
EMERGENCY_DECEL = 1.3  # m/s^2
REGEN_SHARE = 0.65     # fraction of brake energy recovered at high speed


def braking_distance(speed_mps, decel):
    """Stopping distance at constant decel: d = v^2 / (2 * decel)."""
    return speed_mps ** 2 / (2.0 * decel)


def braking_energy_kwh(speed_mps, mass_kg=TRAIN["mass_kg"]):
    """Kinetic energy at speed: 0.5 m v^2, in kWh."""
    return 0.5 * mass_kg * speed_mps ** 2 / 3.6e6


def braking_report(speed_mps=None):
    if speed_mps is None:
        speed_mps = kmh_to_mps(TRAIN["vmax_kmh"])
    e = braking_energy_kwh(speed_mps)
    return {
        "from_kmh": round(mps_to_kmh(speed_mps), 1),
        "service_decel_mps2": SERVICE_DECEL,
        "service_distance_m": round(braking_distance(speed_mps, SERVICE_DECEL), 1),
        "emergency_decel_mps2": EMERGENCY_DECEL,
        "emergency_distance_m": round(braking_distance(speed_mps, EMERGENCY_DECEL), 1),
        "kinetic_energy_kwh": round(e, 1),
        "regen_share": REGEN_SHARE,
        "recovered_kwh": round(e * REGEN_SHARE, 1),
        "eddy_heat_kwh": round(e * (1.0 - REGEN_SHARE), 1),
        "split_note": "regenerative braking returns energy to the grid; "
                      "eddy-current brakes turn the rest into heat in the guideway",
        "gate": gate("emergency_brake_disable"),
    }


# ================================================================ 5. Switching + headway
SWITCH_CYCLE_S = 12.0      # movable guideway beam: move + lock + verify (model)
HEADWAY_MARGIN_S = 20.0    # signalling / reaction margin (model)


def safe_headway(speed_mps):
    """Minimum safe separation: emergency stop + margin + train length."""
    dist = (braking_distance(speed_mps, EMERGENCY_DECEL)
            + speed_mps * HEADWAY_MARGIN_S + TRAIN["length_m"])
    return {
        "speed_kmh": round(mps_to_kmh(speed_mps), 1),
        "headway_distance_m": round(dist, 1),
        "headway_time_s": round(dist / speed_mps, 1) if speed_mps > 0 else -1.0,
        "emergency_stop_m": round(braking_distance(speed_mps, EMERGENCY_DECEL), 1),
    }


def line_capacity(speed_mps):
    """Trains per hour per direction at minimum safe headway."""
    h = safe_headway(speed_mps)
    if h["headway_time_s"] <= 0:
        return -1.0
    return round(3600.0 / h["headway_time_s"], 1)


def switch_report():
    return {
        "switch_cycle_s": SWITCH_CYCLE_S,
        "sequence": ["route request", "beam moves", "mechanical lock",
                     "lock verified", "signal clears"],
        "rule": "a following train may not be routed until lock is verified",
        "gate": gate("guideway_switching_live"),
    }


# ================================================================ 6. Governor + interlocks
GOVERNOR_ZONES = {
    "STATION_APPROACH": 60.0,    # km/h
    "URBAN": 160.0,
    "SUBURBAN": 250.0,
    "OPEN_CORRIDOR": 430.0,
}
ABSOLUTE_CEILING_KMH = 430.0


def governor_check(zone, speed_kmh):
    """X-family governor: is this speed legal in this zone?"""
    limit = GOVERNOR_ZONES[zone]
    return {
        "zone": zone,
        "limit_kmh": limit,
        "speed_kmh": speed_kmh,
        "ok": speed_kmh <= min(limit, ABSOLUTE_CEILING_KMH),
        "gate": gate("exceed_speed_governor_live"),
    }


INTERLOCKS = [
    "doors_closed", "guideway_clear", "suspension_healthy", "power_ok",
    "wind_within_limit", "switch_locked", "brakes_tested", "cabin_pressure_ok",
]


def departure_check(state):
    """All 8 interlocks must be True before a simulated departure."""
    failed = [k for k in INTERLOCKS if not state.get(k, False)]
    return {
        "interlocks": len(INTERLOCKS),
        "failed": failed,
        "all_pass": not failed,
        "verdict": "DEPART (sim)" if not failed else "HOLD AT PLATFORM (sim)",
        "gate": gate("bypass_safety_interlock"),
    }


def nominal_state():
    return {k: True for k in INTERLOCKS}


# ================================================================ 7. Ride quality
COMFORT_LIMIT_MPS2 = 1.0   # lateral acceleration comfort limit (model)


def lateral_accel(speed_mps, radius_m, bank_rad=0.0):
    """Effective lateral accel in a banked curve: v^2/R cos(b) - g sin(b)."""
    return speed_mps ** 2 / radius_m * math.cos(bank_rad) - G * math.sin(bank_rad)


def min_curve_radius(speed_mps, limit=COMFORT_LIMIT_MPS2):
    """Smallest unbanked radius keeping lateral accel at the limit: R = v^2/a."""
    return speed_mps ** 2 / limit


def ride_report(speed_mps=None):
    if speed_mps is None:
        speed_mps = kmh_to_mps(TRAIN["vmax_kmh"])
    r = min_curve_radius(speed_mps)
    return {
        "speed_kmh": round(mps_to_kmh(speed_mps), 1),
        "comfort_limit_mps2": COMFORT_LIMIT_MPS2,
        "min_radius_unbanked_m": round(r, 1),
        "accel_at_min_radius": round(lateral_accel(speed_mps, r), 4),
        "accel_banked_6deg": round(lateral_accel(speed_mps, r, math.radians(6.0)), 4),
        "note": "at 430 km/h, curves must be many kilometres wide - "
                "that is why maglev corridors are built straight",
    }


# ================================================================ 8. kNN regime classifier
# Features: (crosswind m/s, load %, speed % of vmax) -> regime label
_KNN_TRAIN = [
    ((2, 50, 50), "safe"), ((5, 60, 70), "safe"), ((3, 80, 60), "safe"),
    ((8, 50, 80), "safe"), ((0, 40, 40), "safe"), ((6, 70, 90), "safe"),
    ((12, 80, 90), "marginal"), ((15, 60, 100), "marginal"),
    ((10, 90, 85), "marginal"), ((14, 85, 95), "marginal"),
    ((11, 70, 100), "marginal"),
    ((22, 90, 100), "no-go"), ((25, 100, 100), "no-go"),
    ((20, 80, 110), "no-go"), ((28, 95, 90), "no-go"),
    ((18, 100, 105), "no-go"),
]


class RegimeClassifier:
    """kNN (k=3) over (crosswind, load, speed) features."""

    def __init__(self, k=3):
        self.k = k
        self.train = list(_KNN_TRAIN)

    @staticmethod
    def _dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def classify(self, features):
        ranked = sorted(self.train, key=lambda t: self._dist(features, t[0]))
        votes = {}
        for _, label in ranked[: self.k]:
            votes[label] = votes.get(label, 0) + 1
        return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def training_accuracy(self):
        hits = sum(1 for f, label in self.train if self.classify(f) == label)
        return round(hits / len(self.train), 4)


SCENARIOS = {
    "calm_commute": (4, 55, 60),
    "busy_windy_corridor": (13, 75, 95),
    "storm_full_speed": (24, 95, 100),
}


def regime_report():
    clf = RegimeClassifier()
    return {
        "training_accuracy": clf.training_accuracy(),
        "scenarios": {name: clf.classify(f) for name, f in SCENARIOS.items()},
    }


# ================================================================ 9. Advisory seats
def _trace(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def carbon4_decide(topic="maglevx-demo"):
    """Gemini Carbon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-carbon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Carbon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "can_override_hold": False,
        "tighten_only": True,
        "decision": "PROCEED (sandbox simulation only)",
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="maglevx-demo"):
    """Muse advisor seat: advisory only, authority none."""
    return {
        "seat": "Muse",
        "role": "advisor",
        "authority": "none - advisory only",
        "can_override_hold": False,
        "live_calls": 0,
        "advice": "Keep every live rail operation behind its HOLD gate.",
        "trace_sha256_16": _trace({"seat": "claude-advisor", "topic": topic}),
    }


def google_flow_plan():
    """Google Flow: plan-only. Nothing renders, nothing is sent."""
    return {
        "seat": "Google Flow",
        "flows": ["FlowX task graph", "Workspace Studio drafts",
                  "Filmmaking 5-shot storyboard", "DataX data-flow"],
        "rendered": False,
        "live": 0,
        "plan": ["EMS vs EDS explainer", "0-430 acceleration story",
                 "braking-distance chart", "guideway switch hero shot"],
    }


def brains_report():
    return {"decider": carbon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    v_max = kmh_to_mps(TRAIN["vmax_kmh"])
    chk("kmh_mps_roundtrip", abs(mps_to_kmh(kmh_to_mps(430.0)) - 430.0) < 1e-9)
    chk("ems_module_force_positive", ems_module_force(50.0, EMS_GAP_M) > 0)
    chk("ems_gap_inverse_square",
        abs(ems_module_force(50.0, 0.010) / ems_module_force(50.0, 0.020) - 4.0) < 1e-9)
    i_eq = ems_equilibrium_current()
    chk("ems_equilibrium_balances_weight",
        abs(ems_train_lift(i_eq) - TRAIN["mass_kg"] * G) < 1.0)
    chk("ems_lifts_at_standstill", ems_train_lift(i_eq, EMS_GAP_M) > 0)
    chk("eds_zero_at_standstill", eds_train_lift(0.0) == 0.0)
    chk("eds_lift_rises_with_speed", eds_train_lift(100.0) > eds_train_lift(20.0))
    v_lo = eds_liftoff_speed()
    chk("eds_liftoff_exists", v_lo > 0)
    chk("eds_liftoff_balances_weight",
        abs(eds_train_lift(v_lo) - TRAIN["mass_kg"] * G) < 50.0)
    chk("eds_drag_zero_at_rest", eds_magnetic_drag(0.0) == 0.0)
    chk("eds_drag_peaks_at_vpeak",
        abs(eds_magnetic_drag(EDS_DRAG_VPEAK) - EDS_DRAG_PEAK_N) < 1e-6)
    chk("eds_drag_falls_after_peak", eds_magnetic_drag(v_max) < EDS_DRAG_PEAK_N)
    sc = suspension_comparison()
    chk("suspension_ems_no_wheels", sc["ems"]["wheels_below_liftoff"] is False)
    chk("suspension_eds_needs_wheels", sc["eds"]["wheels_below_liftoff"] is True)
    chk("accel_time_formula", abs(accel_time_to_speed(v_max) - v_max / A_MAX) < 1e-12)
    chk("accel_distance_formula",
        abs(accel_distance(v_max) - v_max ** 2 / (2.0 * A_MAX)) < 1e-9)
    chk("thrust_exceeds_drag_at_cruise_accel",
        thrust_for_accel(TRAIN["mass_kg"], A_MAX, v_max) > aero_drag_force(v_max))
    prof = propulsion_profile()
    chk("profile_reaches_vmax", abs(prof["top_speed_kmh"] - 430.0) < 1.0)
    chk("profile_covers_route", abs(prof["distance_m"] - 430000.0) < 500.0)
    chk("profile_time_plausible", 3300.0 < prof["travel_time_s"] < 4500.0)
    chk("drag_quadruples_when_speed_doubles",
        abs(aero_drag_force(100.0) / aero_drag_force(50.0) - 4.0) < 1e-9)
    chk("cruise_power_positive", cruise_power_kw(v_max) > 0)
    chk("cruise_power_rises_with_speed",
        cruise_power_kw(v_max) > cruise_power_kw(kmh_to_mps(200.0)))
    ec = energy_comparison()
    chk("energy_per_pax_km_positive", ec["maglevx_model_430kmh"] > 0)
    chk("energy_model_below_air_estimate",
        ec["maglevx_model_430kmh"] < ec["short_haul_air_est"])
    chk("braking_service_formula",
        abs(braking_distance(v_max, SERVICE_DECEL) - v_max ** 2 / (2 * SERVICE_DECEL)) < 1e-9)
    chk("braking_emergency_shorter",
        braking_distance(v_max, EMERGENCY_DECEL) < braking_distance(v_max, SERVICE_DECEL))
    br = braking_report()
    chk("braking_split_sums", abs(br["recovered_kwh"] + br["eddy_heat_kwh"]
                                   - br["kinetic_energy_kwh"]) < 0.2)
    chk("braking_gate_hold", br["gate"] == "HOLD")
    hw = safe_headway(v_max)
    chk("headway_exceeds_emergency_stop", hw["headway_distance_m"] > hw["emergency_stop_m"])
    chk("capacity_positive", line_capacity(v_max) > 0)
    chk("switch_gate_hold", switch_report()["gate"] == "HOLD")
    chk("governor_station_ok", governor_check("STATION_APPROACH", 50.0)["ok"] is True)
    chk("governor_urban_violation", governor_check("URBAN", 300.0)["ok"] is False)
    chk("governor_corridor_ok", governor_check("OPEN_CORRIDOR", 430.0)["ok"] is True)
    chk("governor_ceiling", governor_check("OPEN_CORRIDOR", 431.0)["ok"] is False)
    chk("interlocks_eight", len(INTERLOCKS) == 8)
    chk("departure_nominal_passes", departure_check(nominal_state())["all_pass"] is True)
    bad = nominal_state(); bad["wind_within_limit"] = False
    chk("departure_wind_blocks", departure_check(bad)["all_pass"] is False)
    chk("curve_radius_formula",
        abs(min_curve_radius(v_max) - v_max ** 2 / COMFORT_LIMIT_MPS2) < 1e-9)
    chk("curve_accel_at_min_radius",
        abs(lateral_accel(v_max, min_curve_radius(v_max)) - COMFORT_LIMIT_MPS2) < 1e-9)
    chk("banking_reduces_accel",
        lateral_accel(v_max, min_curve_radius(v_max), math.radians(6.0))
        < lateral_accel(v_max, min_curve_radius(v_max)))
    rr = regime_report()
    chk("classifier_accuracy", rr["training_accuracy"] >= 0.9)
    chk("classifier_calm_safe", rr["scenarios"]["calm_commute"] == "safe")
    chk("classifier_storm_nogo", rr["scenarios"]["storm_full_speed"] == "no-go")
    chk("hold_gates_14", len(HOLD_GATES) == 14 and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_passenger_hold", gate("passenger_service_live") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    b = brains_report()
    chk("carbon4_no_override", b["decider"]["can_override_hold"] is False)
    chk("claude_no_override", b["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", b["flow"]["rendered"] is False and b["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    print("=" * 68)
    print("MagLevX - maglev train levitation transport lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | SIM ONLY")
    print("=" * 68)
    v_max = kmh_to_mps(TRAIN["vmax_kmh"])
    print("\n[1] Suspension: EMS vs EDS at train scale (240 t trainset)")
    sc = suspension_comparison()
    print(f"    EMS: gap {sc['ems']['gap_m']} m, lift at standstill "
          f"{sc['ems']['lift_at_standstill_n']:,.0f} N, needs fast feedback")
    print(f"         equilibrium current per module = {ems_equilibrium_current():.2f} A")
    print(f"    EDS: gap {sc['eds']['gap_m']} m, lift at standstill 0 N, "
          f"liftoff at {sc['eds']['liftoff_speed_kmh']} km/h (wheels until then)")
    print(f"    EDS magnetic drag peaks at {EDS_DRAG_PEAK_N:,.0f} N "
          f"@ {EDS_DRAG_VPEAK} m/s, falls to {eds_magnetic_drag(v_max):,.0f} N at cruise")
    print("\n[2] Propulsion: linear motor, LA -> Las Vegas sim corridor (430 km)")
    prof = propulsion_profile()
    print(f"    0 -> 430 km/h in {accel_time_to_speed(v_max):.1f} s over "
          f"{accel_distance(v_max):,.0f} m (a = {A_MAX} m/s^2, jerk = {JERK_MAX} m/s^3)")
    print(f"    full run: {prof['travel_time_min']} min, top speed {prof['top_speed_kmh']} km/h")
    print(f"    thrust at cruise accel = {thrust_for_accel(TRAIN['mass_kg'], A_MAX, v_max):,.0f} N")
    print("\n[3] Aerodynamics + energy at 430 km/h")
    print(f"    aero drag = {aero_drag_force(v_max):,.0f} N (dominant at speed)")
    print(f"    cruise power = {cruise_power_kw(v_max):,.0f} kW | "
          f"energy = {energy_comparison()['maglevx_model_430kmh']} kWh/passenger-km (model)")
    print(f"    references (ESTIMATES): HSR {ENERGY_REFERENCE['high_speed_rail_est']}, "
          f"car {ENERGY_REFERENCE['car_petrol_per_person_est']}, air {ENERGY_REFERENCE['short_haul_air_est']}")
    print("\n[4] Braking from 430 km/h")
    br = braking_report()
    print(f"    service ({SERVICE_DECEL} m/s^2): {br['service_distance_m']:,.0f} m | "
          f"emergency ({EMERGENCY_DECEL} m/s^2): {br['emergency_distance_m']:,.0f} m")
    print(f"    kinetic energy {br['kinetic_energy_kwh']} kWh -> regen {br['recovered_kwh']} kWh, "
          f"eddy heat {br['eddy_heat_kwh']} kWh")
    print("\n[5] Guideway switching + safe headway")
    hw = safe_headway(v_max)
    print(f"    switch cycle = {SWITCH_CYCLE_S} s (move, lock, verify) - gate HOLD")
    print(f"    headway at 430 km/h: {hw['headway_distance_m']:,.0f} m = {hw['headway_time_s']} s "
          f"-> {line_capacity(v_max)} trains/hour/direction")
    print("\n[6] Speed governor + departure interlocks")
    for zone, limit in GOVERNOR_ZONES.items():
        print(f"    {zone:18s} limit {limit:.0f} km/h")
    dep = departure_check(nominal_state())
    print(f"    interlocks: {dep['interlocks']}/8 pass on nominal state -> {dep['verdict']}")
    bad = nominal_state(); bad["wind_within_limit"] = False
    print(f"    with high wind: {departure_check(bad)['verdict']} (failed: "
          f"{departure_check(bad)['failed']})")
    print("\n[7] Ride quality in curves at 430 km/h")
    rr = ride_report()
    print(f"    min unbanked radius for {COMFORT_LIMIT_MPS2} m/s^2 comfort = "
          f"{rr['min_radius_unbanked_m']:,.0f} m | banked 6 deg: {rr['accel_banked_6deg']} m/s^2")
    print("\n[8] Regime classifier (TinyML kNN)")
    rep = regime_report()
    print(f"    training accuracy = {rep['training_accuracy']}")
    for name, label in rep["scenarios"].items():
        print(f"    {name:20s} -> {label}")
    print("\n[9] Advisory seats")
    print(f"    decider: {carbon4_decide()['decision']} trace={carbon4_decide()['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {claude_advisor()['can_override_hold']}")
    print(f"    flow rendered = {google_flow_plan()['rendered']} live = {google_flow_plan()['live']}")
    print(f"\n[10] HOLD gates: {len(HOLD_GATES)}")
    for k in HOLD_GATES:
        print(f"    HOLD  {k}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "suspension":
        print(json.dumps(suspension_comparison(), indent=2, sort_keys=True))
        return 0
    if cmd == "propulsion":
        print(json.dumps(propulsion_profile(), indent=2, sort_keys=True))
        return 0
    if cmd == "braking":
        print(json.dumps(braking_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "headway":
        print(json.dumps(safe_headway(kmh_to_mps(TRAIN["vmax_kmh"])), indent=2, sort_keys=True))
        print(json.dumps(switch_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "governor":
        for zone in GOVERNOR_ZONES:
            print(json.dumps(governor_check(zone, GOVERNOR_ZONES[zone]), sort_keys=True))
        print(json.dumps(departure_check(nominal_state()), indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(regime_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for k, v in HOLD_GATES.items():
            print(f"{k}: {v}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "live_train_operation"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
