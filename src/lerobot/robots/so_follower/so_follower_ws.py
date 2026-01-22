#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from functools import cached_property
from typing import TypeAlias

try:
    import websocket as ws_client
except Exception:  # pragma: no cover - optional dependency
    ws_client = None

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech.tables import MODEL_CONTROL_TABLE, MODEL_RESOLUTION
from lerobot.motors.motors_bus import get_address
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so_follower import SOFollowerWsConfig

logger = logging.getLogger(__name__)


class SOFollowerWs(Robot):
    """
    SO follower implementation using a WebSocket transport for Feetech servo control.
    """

    config_class = SOFollowerWsConfig
    name = "so_follower_ws"

    apply_drive_mode = True

    def __init__(self, config: SOFollowerWsConfig):
        super().__init__(config)
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        self.config = config

        self.motors = {
            "shoulder_pan": Motor(config.joint_ids["shoulder_pan"], config.joint_models["shoulder_pan"], norm_mode_body),
            "shoulder_lift": Motor(config.joint_ids["shoulder_lift"], config.joint_models["shoulder_lift"], norm_mode_body),
            "elbow_flex": Motor(config.joint_ids["elbow_flex"], config.joint_models["elbow_flex"], norm_mode_body),
            "wrist_flex": Motor(config.joint_ids["wrist_flex"], config.joint_models["wrist_flex"], norm_mode_body),
            "wrist_roll": Motor(config.joint_ids["wrist_roll"], config.joint_models["wrist_roll"], norm_mode_body),
            "gripper": Motor(config.joint_ids["gripper"], config.joint_models["gripper"], MotorNormMode.RANGE_0_100),
        }

        self.cameras = make_cameras_from_configs(config.cameras)
        self._id_to_name = {motor.id: name for name, motor in self.motors.items()}
        self._name_to_id = {name: motor.id for name, motor in self.motors.items()}

        self._ws = None
        self._ws_lock = threading.Lock()
        self._ws_thread = None
        self._ws_stop = threading.Event()
        self._ws_ex = None

        self._latest_raw_by_id: dict[int, int] = {}
        self._latest_ts = 0.0

        self._msg_queue: deque[dict] = deque()
        self._msg_cv = threading.Condition()

    def _require_ws_client(self):
        return ws_client

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws_stop.is_set()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        ws = self._require_ws_client()

        self._ws = ws.create_connection(self.config.ws_url, timeout=self.config.socket_timeout_s)
        self._ws.settimeout(self.config.socket_timeout_s)
        self._ws_ex = ws._exceptions
        self._ws_stop.clear()
        self._ws_thread = threading.Thread(target=self._reader_loop, name="so-ws-reader", daemon=True)
        self._ws_thread.start()

        # NOTICE: enable this in caution when deployed to a real robot
        # where torque release can cause damage or injury if the robot used
        # to hold something or perform serious task while the service is disconnected
        # or crashes.
        if self.config.enable_torque_on_connect:
            self._send_ws_json({"type": "servos:torque:enable"})

        for cam in self.cameras.values():
            cam.connect()

        if not self.is_calibrated and calibrate and not self.config.use_raw:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def calibrate(self) -> None:
        if self.config.use_raw:
            raise RuntimeError("Calibration is not available when use_raw=True.")
        if not self.is_connected:
            raise RuntimeError("WebSocket is not connected.")

        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self._write_calibration_to_motors(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self._disable_torque()
        for motor in self.motors:
            self._write_register(motor, "Operating_Mode", 0)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self._set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self._record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self._write_calibration_to_motors(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        pass

    def _reader_loop(self) -> None:
        while not self._ws_stop.is_set():
            try:
                with self._ws_lock:
                    payload = self._ws.recv() if self._ws else None
            except self._ws_ex.WebSocketTimeoutException:
                continue
            except self._ws_ex.WebSocketConnectionClosedException:
                logger.warning("WebSocket connection closed.")
                self._ws_stop.set()
                break
            except Exception:
                logger.exception("WebSocket read failed.")
                continue

            if not payload:
                continue
            try:
                msg = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "telemetry:servos":
                data = msg.get("data", {})
                joints = data.get("joints", [])
                raw_by_id = {}

                for joint in joints:
                    if not isinstance(joint, dict):
                        continue

                    id_ = joint.get("id")
                    pos = joint.get("pos")
                    if isinstance(id_, int) and isinstance(pos, int):
                        raw_by_id[id_] = pos

                if raw_by_id:
                    self._latest_raw_by_id = raw_by_id
                    self._latest_ts = time.monotonic()

                continue

            with self._msg_cv:
                self._msg_queue.append(msg)
                self._msg_cv.notify_all()

    def _wait_for_positions(self) -> dict[int, int]:
        deadline = time.monotonic() + self.config.telemetry_timeout_s
        while time.monotonic() < deadline:
            if self._latest_raw_by_id:
                return self._latest_raw_by_id

            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for telemetry from WebSocket server.")

    def _normalize_by_id(self, raw_by_id: dict[int, int]) -> dict[str, float]:
        if self.config.use_raw:
            normalized = {}
            for id_, val in raw_by_id.items():
                name = self._id_to_name.get(id_)
                if name is None:
                    continue
                normalized[f"{name}.pos"] = float(val)
            return normalized
        if not self.calibration:
            raise RuntimeError("Calibration is required to normalize positions.")

        normalized = {}

        for id_, val in raw_by_id.items():
            name = self._id_to_name.get(id_)
            if name is None or name not in self.calibration:
                continue

            calibration = self.calibration[name]

            min_ = calibration.range_min
            max_ = calibration.range_max
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{name}': min and max are equal.")

            drive_mode = self.apply_drive_mode and calibration.drive_mode
            motor = self.motors[name]
            if motor.norm_mode is MotorNormMode.RANGE_M100_100:
                norm = (((val - min_) / (max_ - min_)) * 200) - 100
                normalized_val = -norm if drive_mode else norm
            elif motor.norm_mode is MotorNormMode.RANGE_0_100:
                norm = ((val - min_) / (max_ - min_)) * 100
                normalized_val = 100 - norm if drive_mode else norm
            elif motor.norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = MODEL_RESOLUTION[motor.model] - 1
                normalized_val = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError(f"Unsupported normalization mode {motor.norm_mode}")

            normalized[f"{name}.pos"] = float(normalized_val)
        return normalized

    def _unnormalize_by_name(self, goal_pos: dict[str, float]) -> dict[int, int]:
        if self.config.use_raw:
            raw_by_id = {}

            for name, val in goal_pos.items():
                id_ = self._name_to_id.get(name)
                if id_ is None:
                    continue

                raw_by_id[id_] = int(val)

            return raw_by_id
        if not self.calibration:
            raise RuntimeError("Calibration is required to unnormalize positions.")

        raw_by_id = {}

        for name, val in goal_pos.items():
            if name not in self.calibration:
                continue

            calibration = self.calibration[name]
            min_ = calibration.range_min
            max_ = calibration.range_max
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{name}': min and max are equal.")

            drive_mode = self.apply_drive_mode and calibration.drive_mode
            motor = self.motors[name]
            if motor.norm_mode is MotorNormMode.RANGE_M100_100:
                adj = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, adj))
                raw_val = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif motor.norm_mode is MotorNormMode.RANGE_0_100:
                adj = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, adj))
                raw_val = int((bounded_val / 100) * (max_ - min_) + min_)
            elif motor.norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = MODEL_RESOLUTION[motor.model] - 1
                raw_val = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError(f"Unsupported normalization mode {motor.norm_mode}")

            raw_by_id[self._name_to_id[name]] = raw_val

        return raw_by_id

    def _send_ws_json(self, payload: dict) -> None:
        message = json.dumps(payload)
        with self._ws_lock:
            if self._ws is None:
                raise RuntimeError("WebSocket is not connected.")
            self._ws.send(message)

    def _read_raw_positions(self, motors: list[str]) -> dict[str, int]:
        raw_by_id = self._wait_for_positions()
        positions = {}
        for motor in motors:
            id_ = self._name_to_id.get(motor)
            if id_ is None or id_ not in raw_by_id:
                raise RuntimeError(f"Missing telemetry for motor '{motor}'")
            positions[motor] = raw_by_id[id_]

        return positions

    def _read_register(self, motor: str, data_name: str) -> int:
        model = self.motors[motor].model
        addr, length = get_address(MODEL_CONTROL_TABLE, model, data_name)
        resp = self._request_ws_json(
            {"type": "servo:read", "data": {"id": self._name_to_id[motor], "addr": addr, "len": length}},
            "servo:read:ack",
        )

        return int(resp["data"]["value"])

    def _write_register(self, motor: str, data_name: str, value: int) -> None:
        model = self.motors[motor].model
        addr, length = get_address(MODEL_CONTROL_TABLE, model, data_name)
        self._request_ws_json(
            {
                "type": "servo:write",
                "data": {"id": self._name_to_id[motor], "addr": addr, "len": length, "value": value},
            },
            "servo:write:ack",
        )

    def _request_ws_json(self, payload: dict, ack_type: str) -> dict:
        self._send_ws_json(payload)
        deadline = time.monotonic() + self.config.telemetry_timeout_s
        with self._msg_cv:
            while True:
                for msg in list(self._msg_queue):
                    if msg.get("type") == ack_type:
                        self._msg_queue.remove(msg)
                        return msg
                    if msg.get("type") == "error":
                        self._msg_queue.remove(msg)
                        raise RuntimeError(msg.get("data", {}).get("message", "unknown error"))

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._msg_cv.wait(timeout=min(0.1, remaining))

        raise TimeoutError(f"Timed out waiting for {ack_type}")

    @contextmanager
    def _eprom_unlocked(self, motors: list[str]):
        for motor in motors:
            self._request_ws_json(
                {"type": "servo:eprom:unlock", "data": {"id": self._name_to_id[motor]}},
                "servo:eprom:unlock:ack",
            )
        try:
            yield
        finally:
            for motor in motors:
                self._request_ws_json(
                    {"type": "servo:eprom:lock", "data": {"id": self._name_to_id[motor]}},
                    "servo:eprom:lock:ack",
                )

    def _disable_torque(self) -> None:
        for motor in self.motors:
            self._write_register(motor, "Torque_Enable", 0)
            self._write_register(motor, "Lock", 0)

    def _set_half_turn_homings(self) -> dict[str, int]:
        motors = list(self.motors)
        positions = self._read_raw_positions(motors)
        homings = {}
        with self._eprom_unlocked(motors):
            for motor in motors:
                max_res = MODEL_RESOLUTION[self.motors[motor].model] - 1
                homing = positions[motor] - int(max_res / 2)
                self._write_register(motor, "Homing_Offset", homing)
                homings[motor] = homing
        return homings

    def _record_ranges_of_motion(
        self, motors: list[str], display_values: bool = True
    ) -> tuple[dict[str, int], dict[str, int]]:
        start_positions = self._read_raw_positions(motors)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self._read_raw_positions(motors)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motors:
                    print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                move_cursor_up(len(motors) + 3)

        same_min_max = [motor for motor in motors if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{same_min_max}")

        return mins, maxes

    def _write_calibration_to_motors(self, calibration_dict: dict[str, MotorCalibration]) -> None:
        motors = list(self.motors)
        with self._eprom_unlocked(motors):
            for motor, calibration in calibration_dict.items():
                self._write_register(motor, "Homing_Offset", calibration.homing_offset)
                self._write_register(motor, "Min_Position_Limit", calibration.range_min)
                self._write_register(motor, "Max_Position_Limit", calibration.range_max)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        raw_by_id = self._wait_for_positions()
        obs_dict = self._normalize_by_id(raw_by_id)

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            raw_by_id = self._wait_for_positions()
            present_pos = self._normalize_by_id(raw_by_id)
            goal_present_pos = {
                key: (goal_pos[key], present_pos[f"{key}.pos"])
                for key in goal_pos
                if f"{key}.pos" in present_pos
            }

            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        raw_by_id = self._unnormalize_by_name(goal_pos)

        for id_, pos in raw_by_id.items():
            self._send_ws_json(
                {
                    "type": "servo:move",
                    "data": {"id": id_, "pos": pos, "speed": self.config.default_speed},
                }
            )

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.config.disable_torque_on_disconnect:
            try:
                self._send_ws_json({"type": "servos:torque:disable"})
            except Exception:
                logger.warning("Failed to send torque disable on disconnect.", exc_info=True)

        self._ws_stop.set()
        if self._ws_thread:
            self._ws_thread.join(timeout=1.0)
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                logger.warning("Failed to close WebSocket connection.", exc_info=True)

        self._ws = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


SO100FollowerWs: TypeAlias = SOFollowerWs
SO101FollowerWs: TypeAlias = SOFollowerWs
