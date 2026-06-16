#!/usr/bin/env python3
from __future__ import annotations

DEFAULT_POP_SIZE = 8
DEFAULT_N_GEN = 5

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_HAND_PY = _HERE / "3_finger_hand.py"

BOUND_PART1_LENGTH_Z_MM = (40.0, 130.0)
BOUND_PART1_WIDTH_Y_MM = (20.0, 50.0)
BOUND_P2_HZ_MM = (30.0, 100.0)
BOUND_AZIMUTH_FINGER1_DEG = (30.0, 180.0)
BOUND_AZIMUTH_FINGER2_DEG = (-180.0, -30.0)

SIM_TIME_S = 10.0
LOG_ALL_EVALS = True

HOLD_REQUIRED_S = 5.0
ENERGY_PENALTY_J = 1e9

CONTACT_OBJECTIVE_MODE = "phalanx"

RESULT_PARETO_CSV = _HERE / "results_GRIP_TASK_pareto.csv"
RESULT_EVALS_CSV = _HERE / "results_GRIP_TASK_evaluations.csv"
RESULT_META_CSV = _HERE / "results_GRIP_TASK_meta.csv"


def _load_hand_module():
    spec = importlib.util.spec_from_file_location("hand3optim", _HAND_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить {_HAND_PY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hand3optim"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model_and_data(
    hand,
    *,
    part1_length_z_m,
    part1_width_y_m,
    p2_hx_m,
    p2_hy_m,
    p2_hz_m,
    finger_azimuths_deg,
    mm,
):
    import mujoco

    hand.mm = mm
    base_xml = str(_HERE / "base_xml.xml")

    spec = mujoco.MjSpec.from_file(base_xml)
    spec.option.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    n_fingers = len(finger_azimuths_deg)
    finger_ring_r = 30.0 * mm
    euler_ref_deg = [-90.0, 0.0, 90.0]
    roots = []
    finger_quats = []
    for azimuth_deg in finger_azimuths_deg:
        rad = np.deg2rad(float(azimuth_deg))
        roots.append(
            np.array(
                [finger_ring_r * np.cos(rad), finger_ring_r * np.sin(rad), 0.0],
                dtype=float,
            )
        )
        finger_quats.append(
            hand.root_quat_finger_on_ring_deg(float(azimuth_deg), tuple(euler_ref_deg))
        )

    for idx in range(1, n_fingers + 1):
        hand.add_finger(
            spec,
            roots[idx - 1],
            finger_quats[idx - 1],
            str(idx),
            part1_length_z_m,
            part1_width_y_m,
            p2_hx_m,
            p2_hy_m,
            p2_hz_m,
        )

    rot_size = np.array([25 * mm, 35 * mm, 30 * mm]) / 2
    positions = np.array(roots)
    mins = positions.min(axis=0) - rot_size
    maxs = positions.max(axis=0) + rot_size
    ctr = (mins + maxs) / 2
    ext = (maxs - mins) / 2

    bbox_geom = spec.worldbody.add_geom(
        name="fingers_bbox",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(ext),
        pos=list(ctr),
        euler=[0, 0, 90],
        rgba=[0, 1, 0, 0.2],
    )
    bbox_geom.contype = 0
    bbox_geom.conaffinity = 0

    cyl_radius = 27 * mm
    cyl_half_h = 32 * mm
    cyl_center = np.array([0.0 * mm, 0.0 * mm, -50 * mm])
    cyl_euler_deg = [90.0, 0.0, 0.0]

    cyl_body = spec.worldbody.add_body(
        name=hand.GRASP_OBJECT_BODY_NAME,
        pos=list(cyl_center),
        euler=list(cyl_euler_deg),
    )
    cyl_body.gravcomp = 1.0
    cyl_body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    cyl_body.add_geom(
        name=hand.GRASP_OBJECT_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cyl_radius, cyl_half_h, 0.0],
        rgba=[0.85, 0.65, 0.15, 1.0],
        friction=[1.0, 1.0, 1.0],
        condim=6,
        density=650.0,
    )

    hand.add_hand_finger_motor_actuators(
        spec, hand.AK10_V2_JOINT_MAX_TORQUE_NM, n_fingers
    )
    _ = spec.compile()
    xml = spec.to_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def _phalange_geom_id_set(model, n_fingers: int):
    import mujoco

    gids: set[int] = set()
    for i in range(1, int(n_fingers) + 1):
        for suffix in ("PART1", "PART2"):
            gid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"ID{i}_{suffix}_geom"
            )
            if gid >= 0:
                gids.add(gid)
    return gids


def simulate_grasp_metrics(
    hand,
    x: np.ndarray,
    *,
    mm: float = 1e-3,
    sim_time_s: float = SIM_TIME_S,
) :
    import mujoco

    part1_L_mm, part1_Wy_mm, p2_hz_mm, az1_deg, az2_deg = [float(v) for v in x]
    part1_len_m = part1_L_mm * 1e-3
    part1_wy_m = part1_Wy_mm * 1e-3
    p2_hz_m = p2_hz_mm * 1e-3
    p2_hy_m = part1_wy_m / 2.0
    p2_hx_m = float(hand.OPT_P2_HX)
    azimuths = [az1_deg, az2_deg, 0.0]

    try:
        model, data = build_model_and_data(
            hand,
            part1_length_z_m=part1_len_m,
            part1_width_y_m=part1_wy_m,
            p2_hx_m=p2_hx_m,
            p2_hy_m=p2_hy_m,
            p2_hz_m=p2_hz_m,
            finger_azimuths_deg=azimuths,
            mm=mm,
        )
    except Exception:
        return 0.0, 0.0, False, {}

    n_fingers = 3
    grasp_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, hand.GRASP_OBJECT_GEOM_NAME
    )
    if grasp_geom_id < 0:
        return 0.0, 0.0, False, {}

    finger_geom_ids = hand.finger_geom_id_set(model, n_fingers)
    phalange_geom_ids = _phalange_geom_id_set(model, n_fingers)

    gravity_nominal = np.asarray(model.opt.gravity, dtype=np.float64).copy()
    model.opt.gravity[:] = 0.0

    cylinder_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, hand.GRASP_OBJECT_BODY_NAME
    )
    if cylinder_bid < 0:
        return 0.0, 0.0, False, {}

    motor_act_ids: list[int] = []
    joint_dofadr: list[int] = []
    for fi in range(1, n_fingers + 1):
        for part in ("PART1", "PART2"):
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_ID{fi}_{part}"
            )
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"ID{fi}_{part}_joint"
            )
            if aid < 0 or jid < 0:
                return 0.0, 0.0, False, {}
            motor_act_ids.append(aid)
            joint_dofadr.append(int(model.jnt_dofadr[jid]))

    dt = float(model.opt.timestep)
    max_steps = int(sim_time_s / dt) + 2

    mujoco.mj_forward(model, data)

    grasp_hold = False
    grasp_achieved = False
    first_grasp_time_s: float | None = None
    hold_contact_time_s = 0.0
    phalanx_touch_integral = 0.0
    manifold_integral = 0.0
    force_integral_Ns = 0.0
    max_phalanges_simultaneous = 0
    energy_J = 0.0
    force_buf = np.zeros(6, dtype=np.float64)

    for _step in range(max_steps):
        if data.time >= sim_time_s:
            break

        closing_allowed = data.time >= float(hand.GRASP_MOTOR_START_DELAY_S)

        if grasp_hold:
            cmd_tau = 0.0
        elif hand.contact_grasp_object_with_any_finger(
            data, grasp_geom_id, finger_geom_ids
        ):
            grasp_hold = True
            grasp_achieved = True
            if first_grasp_time_s is None:
                first_grasp_time_s = float(data.time)
            cmd_tau = 0.0
            model.opt.gravity[:] = gravity_nominal
            model.body_gravcomp[cylinder_bid] = 0.0
            mujoco.mj_forward(model, data)
        elif closing_allowed:
            cmd_tau = float(hand.GRASP_CLOSE_TORQUE_NM) * float(
                hand.GRASP_CLOSE_TORQUE_SIGN
            )
        else:
            cmd_tau = 0.0

        for aid in motor_act_ids:
            data.ctrl[aid] = cmd_tau

        for aid, dof_i in zip(motor_act_ids, joint_dofadr, strict=True):
            tau = float(data.ctrl[aid])
            qd = float(data.qvel[dof_i])
            energy_J += max(0.0, tau * qd) * dt

        mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            return 0.0, 0.0, False, {}

        if grasp_hold and hand.contact_grasp_object_with_any_finger(
            data, grasp_geom_id, finger_geom_ids
        ):
            hold_contact_time_s += dt

        touching_phalanges: set[int] = set()
        n_manifolds_finger_cyl = 0
        for ci in range(data.ncon):
            g1 = int(data.contact[ci].geom1)
            g2 = int(data.contact[ci].geom2)
            if not (
                (g1 == grasp_geom_id and g2 in finger_geom_ids)
                or (g2 == grasp_geom_id and g1 in finger_geom_ids)
            ):
                continue
            fg = g2 if g1 == grasp_geom_id else g1
            n_manifolds_finger_cyl += 1
            if fg in phalange_geom_ids:
                touching_phalanges.add(fg)
            mujoco.mj_contactForce(model, data, ci, force_buf)
            fn = float(np.linalg.norm(force_buf[:3]))
            force_integral_Ns += fn * dt

        n_ph = len(touching_phalanges)
        max_phalanges_simultaneous = max(max_phalanges_simultaneous, n_ph)
        phalanx_touch_integral += float(n_ph) * dt
        manifold_integral += float(n_manifolds_finger_cyl) * dt

    if CONTACT_OBJECTIVE_MODE == "force":
        contact_primary = force_integral_Ns
    else:
        contact_primary = phalanx_touch_integral

    grasp_ok = grasp_achieved and hold_contact_time_s >= float(HOLD_REQUIRED_S)
    energy_for_objective = energy_J if grasp_ok else float(ENERGY_PENALTY_J)

    diag = {
        "phalanx_touch_integral_geom_s": phalanx_touch_integral,
        "max_phalanges_simultaneous": int(max_phalanges_simultaneous),
        "contact_manifold_integral": manifold_integral,
        "force_integral_Ns": force_integral_Ns,
        "contact_objective_mode": CONTACT_OBJECTIVE_MODE,
        "grasp_achieved": bool(grasp_achieved),
        "grasp_ok": bool(grasp_ok),
        "hold_contact_time_s": float(hold_contact_time_s),
        "hold_required_s": float(HOLD_REQUIRED_S),
        "first_grasp_time_s": first_grasp_time_s,
        "energy_motor_J": float(energy_J),
        "energy_for_objective_J": float(energy_for_objective),
    }
    return contact_primary, energy_for_objective, grasp_ok, diag


def run_nsga2(*, hand, mm: float, pop_size: int, n_gen: int, seed: int, sim_time_s: float):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    xl = np.array(
        [
            BOUND_PART1_LENGTH_Z_MM[0],
            BOUND_PART1_WIDTH_Y_MM[0],
            BOUND_P2_HZ_MM[0],
            BOUND_AZIMUTH_FINGER1_DEG[0],
            BOUND_AZIMUTH_FINGER2_DEG[0],
        ]
    )
    xu = np.array(
        [
            BOUND_PART1_LENGTH_Z_MM[1],
            BOUND_PART1_WIDTH_Y_MM[1],
            BOUND_P2_HZ_MM[1],
            BOUND_AZIMUTH_FINGER1_DEG[1],
            BOUND_AZIMUTH_FINGER2_DEG[1],
        ]
    )

    eval_rows: list[dict] = []

    class GripProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=5, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            xv = np.asarray(x, dtype=float).ravel()
            C, E, ok, diag = simulate_grasp_metrics(
                hand, xv, mm=mm, sim_time_s=sim_time_s
            )
            row = {
                "part1_L_mm": xv[0],
                "part1_Wy_mm": xv[1],
                "p2_hz_mm": xv[2],
                "az1_deg": xv[3],
                "az2_deg": xv[4],
                "az3_deg": 0.0,
                "p2_hy_mm_derived": xv[1] / 2.0,
                "contact_primary": C if diag else None,
                "energy_J": E if diag else None,
                "ok": ok,
                "contact_objective_mode": CONTACT_OBJECTIVE_MODE,
                "phalanx_touch_integral_geom_s": diag.get(
                    "phalanx_touch_integral_geom_s"
                ),
                "max_phalanges_simultaneous": diag.get("max_phalanges_simultaneous"),
                "contact_manifold_integral": diag.get("contact_manifold_integral"),
                "force_integral_Ns": diag.get("force_integral_Ns"),
                "grasp_achieved": diag.get("grasp_achieved"),
                "grasp_ok": diag.get("grasp_ok"),
                "hold_contact_time_s": diag.get("hold_contact_time_s"),
                "hold_required_s": diag.get("hold_required_s"),
                "first_grasp_time_s": diag.get("first_grasp_time_s"),
                "energy_motor_J": diag.get("energy_motor_J"),
            }
            if LOG_ALL_EVALS:
                eval_rows.append(row)

            if not ok:
                out["F"] = np.array([0.0, float(ENERGY_PENALTY_J)])
            else:
                out["F"] = np.array([-C, E])

    problem = GripProblem()
    algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
    )
    return res, xl, xu, eval_rows


def save_csv_pareto(
    res, *, hand, mm: float, sim_time_s: float
):
    X = np.asarray(res.X)
    F = np.asarray(res.F)
    if X.size == 0:
        return
    headers = [
        "part1_L_mm",
        "part1_Wy_mm",
        "p2_hz_mm",
        "az1_deg",
        "az2_deg",
        "az3_deg",
        "p2_hy_mm_derived",
        "f0_pymoo",
        "f1_energy_J",
        "contact_primary",
        "energy_J",
        "ok",
        "contact_objective_mode",
        "phalanx_touch_integral_geom_s",
        "max_phalanges_simultaneous",
        "contact_manifold_integral",
        "force_integral_Ns",
        "grasp_achieved",
        "grasp_ok",
        "hold_contact_time_s",
        "hold_required_s",
        "first_grasp_time_s",
        "energy_motor_J",
    ]
    with open(RESULT_PARETO_CSV, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(headers)
        for i in range(len(X)):
            xv = np.asarray(X[i], dtype=float).ravel()
            C, E, ok, diag = simulate_grasp_metrics(
                hand, xv, mm=mm, sim_time_s=sim_time_s
            )
            row = list(xv) + [0.0, xv[1] / 2.0]
            row += [
                float(F[i, 0]),
                float(F[i, 1]),
                C if diag else None,
                E if diag else None,
                ok,
                CONTACT_OBJECTIVE_MODE,
                diag.get("phalanx_touch_integral_geom_s"),
                diag.get("max_phalanges_simultaneous"),
                diag.get("contact_manifold_integral"),
                diag.get("force_integral_Ns"),
                diag.get("grasp_achieved"),
                diag.get("grasp_ok"),
                diag.get("hold_contact_time_s"),
                diag.get("hold_required_s"),
                diag.get("first_grasp_time_s"),
                diag.get("energy_motor_J"),
            ]
            w.writerow(row)


def save_csv_evaluations(rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(RESULT_EVALS_CSV, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_csv_meta(
    *,
    xl: np.ndarray,
    xu: np.ndarray,
    pop_size: int,
    n_gen: int,
    seed: int,
    sim_time_s: float,
    hand,
):
    rows = [
        ("optimizer", "NSGA-II (pymoo)"),
        ("pop_size", pop_size),
        ("n_gen", n_gen),
        ("seed", seed),
        ("sim_time_s", sim_time_s),
        (
            "objective_0",
            "minimize -contact_primary: phalanx=integral(# phalange geoms touching cyl)*dt [geom·s]; "
            "force=integral||F||dt [N·s] (CONTACT_OBJECTIVE_MODE)",
        ),
        (
            "objective_1",
            f"minimize energy_J (motor ∫τ·qd dt) only if grasp_ok "
            f"(hold_contact_time_s >= {HOLD_REQUIRED_S} after grasp); else {ENERGY_PENALTY_J}",
        ),
        ("HOLD_REQUIRED_S", HOLD_REQUIRED_S),
        ("ENERGY_PENALTY_J", ENERGY_PENALTY_J),
        ("CONTACT_OBJECTIVE_MODE", CONTACT_OBJECTIVE_MODE),
        ("bounds_part1_L_mm", str(BOUND_PART1_LENGTH_Z_MM)),
        ("bounds_part1_Wy_mm", str(BOUND_PART1_WIDTH_Y_MM)),
        ("bounds_p2_hz_mm", str(BOUND_P2_HZ_MM)),
        ("bounds_az1_deg", str(BOUND_AZIMUTH_FINGER1_DEG)),
        ("bounds_az2_deg", str(BOUND_AZIMUTH_FINGER2_DEG)),
        ("az3_deg_fixed", "0"),
        ("p2_hy_rule", "part1_Wy_mm / 2 (half-size in MuJoCo = full PART2 Y / 2)"),
        ("motor_model", getattr(hand, "HAND_OPTIM_PARAMS", {}).get("motor_model", "")),
        ("xl", str(xl.tolist())),
        ("xu", str(xu.tolist())),
    ]
    with open(RESULT_META_CSV, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["key", "value"])
        for k, v in rows:
            w.writerow([k, v])


def main():
    try:
        import pymoo  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Нужен pymoo в том же Python, что и mujoco:\n"
            "  python -m pip install pymoo\n"
            "Проверка: python -c \"import pymoo, mujoco\""
        ) from e

    p = argparse.ArgumentParser(
        description="NSGA-II оптимизация захвата (3_finger_hand)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pop",
        type=int,
        default=DEFAULT_POP_SIZE,
        metavar="N",
        help="Размер популяции (особей за одно поколение), не число поколений",
    )
    p.add_argument(
        "--gen",
        type=int,
        default=DEFAULT_N_GEN,
        metavar="G",
        help="Число поколений NSGA-II",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sim-time",
        type=float,
        default=SIM_TIME_S,
        help="Длительность одного прогона симуляции (с)",
    )
    args = p.parse_args()

    approx_evals = int(args.pop) * int(args.gen)

    hand = _load_hand_module()
    mm = 1e-3

    res, xl, xu, eval_rows = run_nsga2(
        hand=hand,
        mm=mm,
        pop_size=args.pop,
        n_gen=args.gen,
        seed=args.seed,
        sim_time_s=float(args.sim_time),
    )

    save_csv_pareto(res, hand=hand, mm=mm, sim_time_s=float(args.sim_time))
    save_csv_meta(
        xl=xl,
        xu=xu,
        pop_size=args.pop,
        n_gen=args.gen,
        seed=args.seed,
        sim_time_s=float(args.sim_time),
        hand=hand,
    )
    if LOG_ALL_EVALS:
        save_csv_evaluations(eval_rows)

    print("Парето:", RESULT_PARETO_CSV)
    print("Мета:", RESULT_META_CSV)
    if LOG_ALL_EVALS:
        print(RESULT_EVALS_CSV)
    print("n_pareto_solutions:", len(res.X) if res.X is not None else 0)


if __name__ == "__main__":
    main()
