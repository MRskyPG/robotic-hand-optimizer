from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_HERE = Path(__file__).resolve().parent
_HAND_PY = _HERE / "3_finger_hand.py"
DEFAULT_PARETO_CSV = _HERE / "results_GRIP_TASK_pareto.csv"

PARAM_COLS = (
    "part1_L_mm",
    "part1_Wy_mm",
    "p2_hz_mm",
    "az1_deg",
    "az2_deg",
    "az3_deg",
    "p2_hy_mm_derived",
)


def _load_hand_module():
    spec = importlib.util.spec_from_file_location("hand3resopt", _HAND_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить {_HAND_PY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hand3resopt"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def load_pareto_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"CSV пуст: {csv_path}")
    return rows


def _valid_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        if "ok" in r and not _parse_bool(r["ok"]):
            continue
        out.append(r)
    return out if out else rows


def select_best_phalanx_contact(rows: list[dict[str, str]]) -> dict[str, str]:
    data = _valid_rows(rows)
    max_ph = max(float(r["max_phalanges_simultaneous"]) for r in data)
    tied = [r for r in data if float(r["max_phalanges_simultaneous"]) == max_ph]
    best_int = max(float(r["phalanx_touch_integral_geom_s"]) for r in tied)
    best = [r for r in tied if float(r["phalanx_touch_integral_geom_s"]) == best_int]
    return best[0]


def select_min_energy(rows: list[dict[str, str]]) -> dict[str, str]:
    data = _valid_rows(rows)
    if data and "grasp_ok" in data[0]:
        grasped = [r for r in data if _parse_bool(r.get("grasp_ok", "False"))]
        if grasped:
            data = grasped
    elif data and "hold_contact_time_s" in data[0]:
        hold_req = 5.0
        if data[0].get("hold_required_s"):
            hold_req = float(data[0]["hold_required_s"])
        grasped = [
            r
            for r in data
            if float(r.get("hold_contact_time_s", 0)) >= hold_req
            and _parse_bool(r.get("grasp_achieved", "False"))
        ]
        if grasped:
            data = grasped
    min_e = min(float(r["f1_energy_J"]) for r in data)
    tied = [r for r in data if float(r["f1_energy_J"]) == min_e]
    return tied[0]


def row_to_build_kwargs(hand, row: dict[str, str], *, mm: float = 1e-3) -> dict:
    part1_L_mm = float(row["part1_L_mm"])
    part1_Wy_mm = float(row["part1_Wy_mm"])
    p2_hz_mm = float(row["p2_hz_mm"])
    p2_hy_mm = float(row["p2_hy_mm_derived"])
    azimuths = [float(row["az1_deg"]), float(row["az2_deg"]), float(row["az3_deg"])]

    return {
        "part1_length_z_m": part1_L_mm * mm,
        "part1_width_y_m": part1_Wy_mm * mm,
        "p2_hx_m": float(hand.OPT_P2_HX),
        "p2_hy_m": p2_hy_mm * mm,
        "p2_hz_m": p2_hz_mm * mm,
        "finger_azimuths_deg": azimuths,
        "mm": mm,
        "display": {
            "part1_L_mm": part1_L_mm,
            "part1_Wy_mm": part1_Wy_mm,
            "p2_hz_mm": p2_hz_mm,
            "p2_hy_mm_derived": p2_hy_mm,
            "az1_deg": azimuths[0],
            "az2_deg": azimuths[1],
            "az3_deg": azimuths[2],
        },
    }


def build_model_and_data(hand, **kwargs):
    # Сборка сцены как в 3_finger_hand.py
    mm = kwargs["mm"]
    finger_azimuths_deg = kwargs["finger_azimuths_deg"]
    part1_length_z_m = kwargs["part1_length_z_m"]
    part1_width_y_m = kwargs["part1_width_y_m"]
    p2_hx_m = kwargs["p2_hx_m"]
    p2_hy_m = kwargs["p2_hy_m"]
    p2_hz_m = kwargs["p2_hz_m"]

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


def _print_row_summary(title: str, row: dict[str, str], display: dict) -> None:
    print("\n")
    print(title)
    for k in PARAM_COLS:
        print(f"  {k}: {display.get(k, row.get(k))}")
    extras = (
        ("max_phalanges_simultaneous", row.get("max_phalanges_simultaneous")),
        ("phalanx_touch_integral_geom_s", row.get("phalanx_touch_integral_geom_s")),
        ("f1_energy_J", row.get("f1_energy_J")),
        ("energy_J", row.get("energy_J")),
        ("contact_primary", row.get("contact_primary")),
    )
    for k, v in extras:
        if v is not None:
            print(f"  {k}: {v}")
    print("Закройте окно MuJoCo, чтобы перейти к следующей визуализации.\n")


def run_grasp_viewer(hand, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Цикл захвата как в 3_finger_hand.py __main__.
    n_fingers = 3
    motor_act_ids: list[int] = []
    for fi in range(1, n_fingers + 1):
        for part in ("PART1", "PART2"):
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_ID{fi}_{part}"
            )
            if aid < 0:
                raise RuntimeError(f"Актуатор motor_ID{fi}_{part} не найден")
            motor_act_ids.append(aid)

    grasp_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, hand.GRASP_OBJECT_GEOM_NAME
    )
    if grasp_geom_id < 0:
        raise RuntimeError(f"Геометрия «{hand.GRASP_OBJECT_GEOM_NAME}» не найдена")
    finger_geom_ids = hand.finger_geom_id_set(model, n_fingers)

    gravity_nominal = np.asarray(model.opt.gravity, dtype=np.float64).copy()
    model.opt.gravity[:] = 0.0

    cylinder_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, hand.GRASP_OBJECT_BODY_NAME
    )
    if cylinder_bid < 0:
        raise RuntimeError(f"Тело «{hand.GRASP_OBJECT_BODY_NAME}» не найдено")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_forward(model, data)
        grasp_hold = False

        while viewer.is_running():
            step_start = time.time()
            closing_allowed = data.time >= float(hand.GRASP_MOTOR_START_DELAY_S)

            if grasp_hold:
                cmd_tau = 0.0
            elif hand.contact_grasp_object_with_any_finger(
                data, grasp_geom_id, finger_geom_ids
            ):
                grasp_hold = True
                cmd_tau = 0.0
                model.opt.gravity[:] = gravity_nominal
                model.body_gravcomp[cylinder_bid] = 0.0
                mujoco.mj_forward(model, data)
                print(
                    "Захват: контакт палец и объект - гравитация вкл., "
                    "gravcomp цилиндра выкл., моторы стоп."
                )
            elif closing_allowed:
                cmd_tau = float(hand.GRASP_CLOSE_TORQUE_NM) * float(
                    hand.GRASP_CLOSE_TORQUE_SIGN
                )
            else:
                cmd_tau = 0.0

            for aid in motor_act_ids:
                data.ctrl[aid] = cmd_tau

            mujoco.mj_step(model, data)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                    data.time % 2
                )

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def visualize_row(hand, row: dict[str, str], *, title: str) -> None:
    kw = row_to_build_kwargs(hand, row)
    _print_row_summary(title, row, kw["display"])
    build_kw = {k: v for k, v in kw.items() if k != "display"}
    model, data = build_model_and_data(hand, **build_kw)
    run_grasp_viewer(hand, model, data)


def main() -> None:
    p = argparse.ArgumentParser(
        description="MuJoCo: лучшие решения из results_GRIP_TASK_pareto.csv",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_PARETO_CSV,
        help="Путь к results_GRIP_TASK_pareto.csv",
    )
    args = p.parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Файл не найден: {csv_path}")

    rows = load_pareto_rows(csv_path)
    hand = _load_hand_module()

    row_ph = select_best_phalanx_contact(rows)
    visualize_row(
        hand,
        row_ph,
        title="[1/2] Лучший по контакту: max(max_phalanges_simultaneous), "
        "при равенстве max(phalanx_touch_integral_geom_s)",
    )

    row_en = select_min_energy(rows)
    visualize_row(
        hand,
        row_en,
        title="[2/2] Лучший по энергии: min(f1_energy_J) при успешном захвате (grasp_ok)",
    )

    print("Обе визуализации завершены.")


if __name__ == "__main__":
    main()
