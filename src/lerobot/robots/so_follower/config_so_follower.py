#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import TypeAlias

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    pass


SO100FollowerConfig: TypeAlias = SOFollowerRobotConfig
SO101FollowerConfig: TypeAlias = SOFollowerRobotConfig

@dataclass
class SOFollowerWsBaseConfig:
    """WebSocket configuration for SO Follower robots."""

    ws_url: str

    # enable this in caution when deployed to a real robot
    # where torque release can cause damage or injury if the robot used
    # to hold something or perform serious task while the service is disconnected
    # or crashes.
    disable_torque_on_disconnect: bool = False
    # enable this in caution when deployed to a real robot
    # where torque release can cause damage or injury if the robot used
    # to hold something or perform serious task while the service is disconnected
    # or crashes.
    enable_torque_on_connect: bool = False

    default_speed: int = 1500
    telemetry_timeout_s: float = 1.0
    socket_timeout_s: float = 0.1
    use_raw: bool = False

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    joint_ids: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }
    )
    joint_models: dict[str, str] = field(
        default_factory=lambda: {
            "shoulder_pan": "sts3215",
            "shoulder_lift": "sts3215",
            "elbow_flex": "sts3215",
            "wrist_flex": "sts3215",
            "wrist_roll": "sts3215",
            "gripper": "sts3215",
        }
    )


@RobotConfig.register_subclass("so101_follower_ws")
@RobotConfig.register_subclass("so100_follower_ws")
@dataclass
class SOFollowerWsConfig(RobotConfig, SOFollowerWsBaseConfig):
    pass


SO100FollowerWsConfig: TypeAlias = SOFollowerWsConfig
SO101FollowerWsConfig: TypeAlias = SOFollowerWsConfig
