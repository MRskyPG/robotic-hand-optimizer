import mujoco
import mujoco.viewer
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

_HAND_MM = 1e-3

# PART1 add_geom: полная длина вдоль локальной Z; полная ширина вдоль Y
OPT_PART1_LENGTH_Z = 60 * _HAND_MM
OPT_PART1_WIDTH_Y = 40 * _HAND_MM

# PART2 add_geom: половины рёбер вдоль X, Y, Z
OPT_P2_HX = 12.5 * _HAND_MM
OPT_P2_HY = 20 * _HAND_MM
OPT_P2_HZ = 36 * _HAND_MM

# Азимуты оснований в плоскости XY
FINGER_AZIMUTHS_DEG = [102.5, -132.5, 0.0]

#  Привод AK10-9 V2.0 KV60 и редуктор
AK10_V2_RATED_TORQUE_SHAFT_NM = 18.0
AK10_V2_RATED_SPEED_RPM = 228.0
AK10_V2_REDUCTION_RATIO = 9.0

# omega на валу мотора
AK10_V2_MOTOR_SHAFT_OMEGA_MAX_RAD_S = AK10_V2_RATED_SPEED_RPM * (2.0 * np.pi / 60.0)

# Лимиты на шарнире пальца после редуктора
AK10_V2_JOINT_MAX_TORQUE_NM = AK10_V2_RATED_TORQUE_SHAFT_NM * AK10_V2_REDUCTION_RATIO
AK10_V2_JOINT_MAX_OMEGA_RAD_S = AK10_V2_MOTOR_SHAFT_OMEGA_MAX_RAD_S / AK10_V2_REDUCTION_RATIO

# Захват
GRASP_OBJECT_GEOM_NAME = "grasp_cylinder_geom"
GRASP_CLOSE_TORQUE_NM = 2.0
GRASP_CLOSE_TORQUE_SIGN = 1.0

GRASP_MOTOR_START_DELAY_S = 1.0
GRASP_OBJECT_BODY_NAME = "grasp_cylinder"

HAND_OPTIM_PARAMS = {
    "part1_length_z": OPT_PART1_LENGTH_Z,
    "part1_width_y": OPT_PART1_WIDTH_Y,
    "p2_hx": OPT_P2_HX,
    "p2_hy": OPT_P2_HY,
    "p2_hz": OPT_P2_HZ,
    "finger_azimuths_deg": list(FINGER_AZIMUTHS_DEG),
    "motor_model": "AK10-9 V2.0 KV60",
    "motor_rated_torque_shaft_nm": AK10_V2_RATED_TORQUE_SHAFT_NM,
    "motor_rated_speed_rpm": AK10_V2_RATED_SPEED_RPM,
    "motor_shaft_omega_max_rad_s": AK10_V2_MOTOR_SHAFT_OMEGA_MAX_RAD_S,
    "motor_reduction_ratio": AK10_V2_REDUCTION_RATIO,
    "joint_max_torque_nm": AK10_V2_JOINT_MAX_TORQUE_NM,
    "joint_max_omega_rad_s": AK10_V2_JOINT_MAX_OMEGA_RAD_S,
    "grasp_object_geom_name": GRASP_OBJECT_GEOM_NAME,
    "grasp_object_body_name": GRASP_OBJECT_BODY_NAME,
    "grasp_close_torque_nm": GRASP_CLOSE_TORQUE_NM,
    "grasp_close_torque_sign": GRASP_CLOSE_TORQUE_SIGN,
    "grasp_motor_start_delay_s": GRASP_MOTOR_START_DELAY_S,
}



def finger_geom_names_for_hand(n_fingers: int) -> list[str]:
    names: list[str] = []
    for i in range(1, int(n_fingers) + 1):
        for suffix in ("ROT", "PART1", "PART2"):
            names.append(f"ID{i}_{suffix}_geom")
    return names


def finger_geom_id_set(model: mujoco.MjModel, n_fingers: int) -> set[int]:
    gids: set[int] = set()
    for name in finger_geom_names_for_hand(n_fingers):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise RuntimeError(f"Геометрия пальца «{name}» не найдена в модели")
        gids.add(gid)
    return gids


def contact_grasp_object_with_any_finger(
    data: mujoco.MjData,
    grasp_geom_id: int,
    finger_geom_ids: set[int],
) -> bool:
    """
    Датчик касания: проверяем, есть ли активный контакт между grasp_geom и любой геометрией пальца
    """
    for i in range(data.ncon):
        g1 = int(data.contact[i].geom1)
        g2 = int(data.contact[i].geom2)
        if g1 == grasp_geom_id and g2 in finger_geom_ids:
            return True
        if g2 == grasp_geom_id and g1 in finger_geom_ids:
            return True
    return False


def add_hand_finger_motor_actuators(
    spec: mujoco.MjSpec,
    max_tau_nm: float,
    n_fingers: int,
) -> None:
    """
    Моторы на шарнирах PART1 / PART2 каждого пальца
    """
    for i in range(1, int(n_fingers) + 1):
        for j_suffix in ("PART1", "PART2"):
            jname = f"ID{i}_{j_suffix}_joint"
            aname = f"motor_ID{i}_{j_suffix}"
            act = spec.add_actuator(name=aname)
            act.set_to_motor()
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.target = jname
            act.ctrlrange = (-float(max_tau_nm), float(max_tau_nm))


def root_quat_finger_on_ring_deg(
    azimuth_deg: float,
    euler_ref_deg: tuple[float, float, float] = (-90.0, 0.0, 90.0),
    azimuth_ref_deg: float = 90.0,
) -> np.ndarray:
    """Ориентация root-пальца"""
    euler_ref = np.deg2rad(np.asarray(euler_ref_deg, dtype=np.float64))
    q_ref = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(q_ref, euler_ref, "xyz")
    delta = np.deg2rad(azimuth_deg - azimuth_ref_deg)
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    q_z = np.zeros(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(q_z, axis, delta)
    q_out = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(q_out, q_z, q_ref)
    return q_out


def add_finger(
    spec,
    root_pos,
    root_quat,
    finger_id,
    part1_length_z,
    part1_width_y,
    p2_hx,
    p2_hy,
    p2_hz,
):
    """
    Добавляет цепочку фаланг пальца (ROT, PART1, PART2) в spec.worldbody.
    Возвращает ссылку на тело ROT
    """

    # Корень вращения (ROT)
    ROT = spec.worldbody.add_body(
        name=f"ID{finger_id}_ROT",
        pos=list(root_pos),
        quat=[float(x) for x in root_quat],
    )
    g_rot = ROT.add_geom(
        name=f"ID{finger_id}_ROT_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[25 * mm / 2, 35 * mm / 2, 30 * mm / 2],
        pos=[0, 0, 0],
        rgba=[0.5, 0.8, 0.5, 0.5],
    )

    # PART1 - первая фаланга
    PART1 = ROT.add_body(
        name=f"ID{finger_id}_PART1",
        pos=[0, 0, 30 * mm / 2],  # Над ROT
        euler=[0, 0, 0]
    )

    # Джоинт для сгибания вправо-влево (вокруг Y) (не учитывая Эйлера)
    PART1.add_joint(
        name=f"ID{finger_id}_PART1_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[0, 0, 0],
        axis=[0, 1, 0],
        range=[0, 90],
        stiffness=0
    )

    # Геометрия PART1, используя половины размеров
    g_p1 = PART1.add_geom(
        name=f"ID{finger_id}_PART1_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[25 * mm / 2, part1_width_y / 2, part1_length_z / 2],
        pos=[0, 0, part1_length_z / 2],
        rgba=[1, 0.5, 0.5, 0.5],
    )

    # PART2 - вторая фаланга (идет вперед по X, без джоинта)
    PART2 = PART1.add_body(
        name=f"ID{finger_id}_PART2",
        pos=[0, 0, part1_length_z],
        euler=[0, 0, 0],
    )

    PART2.add_joint(
        name=f"ID{finger_id}_PART2_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[0, 0, 0],
        axis=[0, 1, 0],
        range=[0, 90],
        stiffness=0  # Высокая жесткость для фиксации = 1000 (если надо будет делать неподвижным)
    )

    # PART2
    g_p2 = PART2.add_geom(
        name=f"ID{finger_id}_PART2_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[p2_hx, p2_hy, p2_hz],
        pos=[p2_hx, 0, p2_hz],
        rgba=[0.5, 0.5, 1, 0.5],
    )

    rot_name = f"ID{finger_id}_ROT"
    p1_name = f"ID{finger_id}_PART1"
    p2_name = f"ID{finger_id}_PART2"
    for b1, b2, tag in (
        (rot_name, p1_name, "rot_p1"),
        (rot_name, p2_name, "rot_p2"),
        (p1_name, p2_name, "p1_p2"),
    ):
        ex = spec.add_exclude()
        ex.bodyname1 = b1
        ex.bodyname2 = b2
        ex.name = f"exc_ID{finger_id}_{tag}"

    return ROT


if __name__ == "__main__":
    mm = 1e-3

    base_xml = str(_HERE / "base_xml.xml")
    mod_xml = str(_HERE / "3_finger_hand.xml")

    spec = mujoco.MjSpec.from_file(base_xml)

    # для столкновений пальцев с цилиндром включаем контакты.
    spec.option.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    # Пальцы на окружности в плоскости XY
    n_fingers = len(FINGER_AZIMUTHS_DEG)
    finger_ring_r = 30 * mm
    euler_ref_deg = [-90.0, 0.0, 90.0]
    azimuth_list = [float(a) for a in FINGER_AZIMUTHS_DEG]

    roots = []
    finger_quats = []
    for azimuth_deg in azimuth_list:
        rad = np.deg2rad(azimuth_deg)
        roots.append(
            np.array(
                [finger_ring_r * np.cos(rad), finger_ring_r * np.sin(rad), 0.0],
                dtype=float,
            )
        )
        finger_quats.append(
            root_quat_finger_on_ring_deg(azimuth_deg, tuple(euler_ref_deg))
        )

    for idx in range(1, n_fingers + 1):
        add_finger(
            spec,
            roots[idx - 1],
            finger_quats[idx - 1],
            str(idx),
            OPT_PART1_LENGTH_Z,
            OPT_PART1_WIDTH_Y,
            OPT_P2_HX,
            OPT_P2_HY,
            OPT_P2_HZ,
        )

    # добавляем для всех пальцев бокс-ладонь
    rot_size = np.array([25 * mm, 35 * mm, 30 * mm]) / 2

    # Находим мин/макс по позициям ROT
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

    # Объект для захвата
    cyl_radius = 27 * mm
    cyl_half_h = 32 * mm
    cyl_center = np.array([0 * mm, 0.0* mm, -50 * mm])
    cyl_euler_deg = [90.0, 0.0, 0.0]

    cyl_body = spec.worldbody.add_body(
        name="grasp_cylinder",
        pos=list(cyl_center),
        euler=list(cyl_euler_deg),
    )
    cyl_body.gravcomp = 1.0
    cyl_body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    cyl_geom = cyl_body.add_geom(
        name="grasp_cylinder_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cyl_radius, cyl_half_h, 0.0],
        rgba=[0.85, 0.65, 0.15, 1.0],
        friction=[1, 1, 1],
        condim=6,
        density=650.0,
    )

    add_hand_finger_motor_actuators(spec, AK10_V2_JOINT_MAX_TORQUE_NM, n_fingers)

    # Промежуточная компиляция
    _ = spec.compile()

    xml = spec.to_xml()

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    with open(mod_xml, "w") as f:
        f.write(xml)

    # Индексы актуаторов-моторов
    motor_act_ids = []
    for fi in range(1, n_fingers + 1):
        for part in ("PART1", "PART2"):
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_ID{fi}_{part}"
            )
            if aid < 0:
                raise RuntimeError(f"Актуатор motor_ID{fi}_{part} не найден")
            motor_act_ids.append(aid)

    grasp_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, GRASP_OBJECT_GEOM_NAME
    )
    if grasp_geom_id < 0:
        raise RuntimeError(f"Геометрия объекта захвата «{GRASP_OBJECT_GEOM_NAME}» не найдена")
    finger_geom_ids = finger_geom_id_set(model, n_fingers)

    gravity_nominal = np.asarray(model.opt.gravity, dtype=np.float64).copy()
    model.opt.gravity[:] = 0.0

    cylinder_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, GRASP_OBJECT_BODY_NAME
    )
    if cylinder_bid < 0:
        raise RuntimeError(f"Тело объекта захвата «{GRASP_OBJECT_BODY_NAME}» не найдено")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_forward(model, data)
        grasp_hold = False  # после касания: выключаем моменты, включаем g на объект

        while viewer.is_running():
            step_start = time.time()

            closing_allowed = data.time >= float(GRASP_MOTOR_START_DELAY_S)

            if grasp_hold:
                cmd_tau = 0.0
            elif contact_grasp_object_with_any_finger(data, grasp_geom_id, finger_geom_ids):
                grasp_hold = True
                cmd_tau = 0.0
                model.opt.gravity[:] = gravity_nominal
                model.body_gravcomp[cylinder_bid] = 0.0
                mujoco.mj_forward(model, data)
                print(
                    "Захват: контакт палец и объект - гравитация включена, gravcomp цилиндра выкл., моторы останавливаем."
                )
            elif closing_allowed:
                cmd_tau = GRASP_CLOSE_TORQUE_NM * GRASP_CLOSE_TORQUE_SIGN
            else:
                cmd_tau = 0.0

            for aid in motor_act_ids:
                data.ctrl[aid] = cmd_tau

            mujoco.mj_step(model, data)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

