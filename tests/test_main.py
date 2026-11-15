import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as mx


VMAX = mx.kmh_to_mps(430.0)


def test_kmh_mps_roundtrip(): assert abs(mx.mps_to_kmh(mx.kmh_to_mps(430.0)) - 430.0) < 1e-9
def test_ems_module_force_positive(): assert mx.ems_module_force(50.0, 0.010) > 0
def test_ems_inverse_square():
    assert abs(mx.ems_module_force(50.0, 0.010) / mx.ems_module_force(50.0, 0.020) - 4.0) < 1e-9
def test_ems_equilibrium_balances():
    i = mx.ems_equilibrium_current()
    assert abs(mx.ems_train_lift(i) - mx.TRAIN["mass_kg"] * mx.G) < 1.0
def test_ems_lifts_at_standstill(): assert mx.ems_train_lift(mx.ems_equilibrium_current()) > 0
def test_ems_gap_shrink_pulls_harder():
    assert mx.ems_module_force(50.0, 0.008) > mx.ems_module_force(50.0, 0.010)
def test_eds_standstill_zero(): assert mx.eds_train_lift(0.0) == 0.0
def test_eds_rises_with_speed(): assert mx.eds_train_lift(100.0) > mx.eds_train_lift(20.0) > 0
def test_eds_liftoff():
    v = mx.eds_liftoff_speed()
    assert v > 0 and abs(mx.eds_train_lift(v) - mx.TRAIN["mass_kg"] * mx.G) < 50.0
def test_eds_drag_rest(): assert mx.eds_magnetic_drag(0.0) == 0.0
def test_eds_drag_peak(): assert abs(mx.eds_magnetic_drag(mx.EDS_DRAG_VPEAK) - mx.EDS_DRAG_PEAK_N) < 1e-6
def test_eds_drag_falls(): assert mx.eds_magnetic_drag(VMAX) < mx.EDS_DRAG_PEAK_N
def test_suspension_wheels():
    sc = mx.suspension_comparison()
    assert sc["ems"]["wheels_below_liftoff"] is False and sc["eds"]["wheels_below_liftoff"] is True
def test_suspension_gate(): assert mx.suspension_comparison()["gate"] == "HOLD"
def test_accel_time(): assert abs(mx.accel_time_to_speed(VMAX) - VMAX / mx.A_MAX) < 1e-12
def test_accel_distance(): assert abs(mx.accel_distance(VMAX) - VMAX ** 2 / (2 * mx.A_MAX)) < 1e-9
def test_thrust(): assert mx.thrust_for_accel(mx.TRAIN["mass_kg"], mx.A_MAX, VMAX) > mx.aero_drag_force(VMAX)
def test_profile_vmax():
    p = mx.propulsion_profile()
    assert abs(p["top_speed_kmh"] - 430.0) < 1.0
def test_profile_distance():
    p = mx.propulsion_profile()
    assert abs(p["distance_m"] - 430000.0) < 500.0
def test_profile_time():
    p = mx.propulsion_profile()
    assert 3300.0 < p["travel_time_s"] < 4500.0
def test_drag_quadratic(): assert abs(mx.aero_drag_force(100.0) / mx.aero_drag_force(50.0) - 4.0) < 1e-9
def test_drag_zero(): assert mx.aero_drag_force(0.0) == 0.0
def test_cruise_power():
    assert mx.cruise_power_kw(VMAX) > mx.cruise_power_kw(mx.kmh_to_mps(200.0)) > 0
def test_energy_comparison():
    ec = mx.energy_comparison()
    assert 0 < ec["maglevx_model_430kmh"] < ec["short_haul_air_est"]
def test_braking_service(): assert abs(mx.braking_distance(VMAX, mx.SERVICE_DECEL) - VMAX ** 2 / 1.6) < 1e-9
def test_braking_emergency_shorter():
    assert mx.braking_distance(VMAX, mx.EMERGENCY_DECEL) < mx.braking_distance(VMAX, mx.SERVICE_DECEL)
def test_braking_energy_formula():
    assert abs(mx.braking_energy_kwh(VMAX) - 0.5 * mx.TRAIN["mass_kg"] * VMAX ** 2 / 3.6e6) < 1e-9
def test_braking_split():
    br = mx.braking_report()
    assert abs(br["recovered_kwh"] + br["eddy_heat_kwh"] - br["kinetic_energy_kwh"]) < 0.2
def test_braking_gate(): assert mx.braking_report()["gate"] == "HOLD"
def test_headway():
    hw = mx.safe_headway(VMAX)
    assert hw["headway_distance_m"] > hw["emergency_stop_m"]
def test_capacity(): assert mx.line_capacity(VMAX) > 0
def test_switch_gate(): assert mx.switch_report()["gate"] == "HOLD"
def test_governor_ok(): assert mx.governor_check("STATION_APPROACH", 50.0)["ok"] is True
def test_governor_violation(): assert mx.governor_check("URBAN", 300.0)["ok"] is False
def test_governor_corridor(): assert mx.governor_check("OPEN_CORRIDOR", 430.0)["ok"] is True
def test_governor_ceiling(): assert mx.governor_check("OPEN_CORRIDOR", 431.0)["ok"] is False
def test_interlocks_count(): assert len(mx.INTERLOCKS) == 8
def test_departure_nominal(): assert mx.departure_check(mx.nominal_state())["all_pass"] is True
def test_departure_wind_blocks():
    s = mx.nominal_state(); s["wind_within_limit"] = False
    d = mx.departure_check(s)
    assert d["all_pass"] is False and d["failed"] == ["wind_within_limit"]
def test_departure_missing_blocks():
    s = mx.nominal_state(); del s["doors_closed"]
    assert mx.departure_check(s)["all_pass"] is False
def test_curve_radius(): assert abs(mx.min_curve_radius(VMAX) - VMAX ** 2 / mx.COMFORT_LIMIT_MPS2) < 1e-9
def test_curve_accel_at_radius():
    assert abs(mx.lateral_accel(VMAX, mx.min_curve_radius(VMAX)) - mx.COMFORT_LIMIT_MPS2) < 1e-9
def test_banking_helps():
    r = mx.min_curve_radius(VMAX)
    assert mx.lateral_accel(VMAX, r, math.radians(6.0)) < mx.lateral_accel(VMAX, r)
def test_classifier_accuracy(): assert mx.regime_report()["training_accuracy"] >= 0.9
def test_classifier_safe(): assert mx.regime_report()["scenarios"]["calm_commute"] == "safe"
def test_classifier_marginal(): assert mx.regime_report()["scenarios"]["busy_windy_corridor"] == "marginal"
def test_classifier_nogo(): assert mx.regime_report()["scenarios"]["storm_full_speed"] == "no-go"
def test_holds_count(): assert len(mx.HOLD_GATES) == 14
def test_holds_all_hold(): assert all(v.startswith("HOLD") for v in mx.HOLD_GATES.values())
def test_gate(): assert mx.gate("passenger_service_live") == "HOLD" and mx.gate("run_simulation") == "PROCEED"
def test_hold_demo_exit_code(): assert mx.main(["hold-demo", "live_train_operation"]) == 2
def test_hold_demo_proceed_code(): assert mx.main(["hold-demo", "run_simulation"]) == 0
def test_carbon4():
    d = mx.carbon4_decide()
    assert d["can_override_hold"] is False and d["live_calls"] == 0 and len(d["trace_sha256_16"]) == 16
def test_claude(): assert mx.claude_advisor()["can_override_hold"] is False
def test_flow(): 
    f = mx.google_flow_plan()
    assert f["rendered"] is False and f["live"] == 0
def test_checks_all_pass():
    checks = mx.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]
