import os
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class H1RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, xml_path="h1.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # 19个可控关节
        self.joint_names = [
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "torso_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_ankle_joint",
            "right_ankle_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
        ]

        self.actuator_ids = [self.model.actuator(name).id for name in self.joint_names]
        self.num_actions = len(self.actuator_ids)

        # 动作空间
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_actions,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32
        )

        self.max_steps = 1000
        self.current_step = 0

    def _get_obs(self):
        # 关节位置 + 速度 + IMU四元数 + IMU角速度 + IMU加速度
        qpos = self.data.qpos[7:7 + self.num_actions].copy()
        qvel = self.data.qvel[:self.num_actions].copy()
        imu_quat = self.data.sensor("imu_quat").data.copy()
        imu_gyro = self.data.sensor("imu_gyro").data.copy()
        imu_acc = self.data.sensor("imu_acc").data.copy()

        # 拼接后是 19+19+4+3+3 = 48 维
        return np.concatenate([qpos, qvel, imu_quat, imu_gyro, imu_acc]).astype(np.float32)

    def _get_reward(self):
        torso_z = self.data.body("torso_link").xpos[2]
        height_reward = np.clip(torso_z - 0.8, 0, 1)
        return height_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # 动作缩放到XML里的力矩范围
        ctrl = np.clip(action, -1, 1)
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = ctrl[i] * self.model.actuator_ctrlrange[aid, 1]

        mujoco.mj_step(self.model, self.data)

        # 窗口渲染
        if self.viewer is not None:
            self.viewer.sync()

        obs = self._get_obs()
        reward = self._get_reward()
        self.current_step += 1

        torso_z = self.data.body("torso_link").xpos[2]
        terminated = torso_z < 0.6
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 3.5

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


# 特征提取
class H1Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(48, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, features_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# train
if __name__ == "__main__":
    if not os.path.exists("h1.xml"):
        raise Exception("请把h1.xml放在同一文件夹！")

    env = H1RobotEnv()
    env.render()

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=H1Extractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
        learning_rate=3e-4,
        verbose=1
    )

    print("开始训练（窗口已打开）...")
    model.learn(total_timesteps=1250000)
    model.save("h1_final_model")

    print("训练完成")
    env.close()


"""
# test
if __name__ == "__main__":
    env = H1RobotEnv()
    env.render()

    model = PPO.load("h1_final_model")

    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
"""
