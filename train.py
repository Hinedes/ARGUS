"""
train_v3.py — Echos PPO Training with Curriculum Learning
----------------------------------------------------------
Curriculum stages:
  Stage 0: Survivor spawns close (2-4m from UAV). No RTH required. Learn gradient climb.
  Stage 1: Survivor at fixed branch position. RTH required. Learn full mission loop.
  Stage 2: Full randomized maze. Everything active. Generalize.

Promotion criterion: mean episode reward > threshold for N consecutive eval episodes.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import numpy as np
import gymnasium as gym
from sim import DynamicEchosEnv


# ---------------------------------------------------------------------------
# Curriculum-aware environment wrapper
# ---------------------------------------------------------------------------

class CurriculumEchosEnv(DynamicEchosEnv):
    """
    Wraps DynamicEchosEnv with a curriculum stage attribute.
    Stage is set externally by the CurriculumCallback.
    """
    def __init__(self):
        super().__init__()
        self.curriculum_stage = 0  # Start easy

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)

        if self.curriculum_stage == 0:
            # Stage 0: Spawn survivor close to UAV so gradient is immediately detectable.
            # Pick a random angle and distance 2-4m from spawn.
            angle = np.random.uniform(-np.pi / 4, np.pi / 4)
            dist = np.random.uniform(2.0, 4.0)
            self.survivor_pos = np.array([
                self.uav_pos[0] + dist * np.cos(angle),
                self.uav_pos[1] + dist * np.sin(angle)
            ])
            # Keep survivor inside maze bounds roughly
            self.survivor_pos = np.clip(self.survivor_pos, [-1.0, -4.5], [11.5, 4.5])
            # No RTH in stage 0 — terminate on survivor found only
            self._stage0_no_rth = True
        else:
            self._stage0_no_rth = False

        # Recompute SNR after survivor reposition
        self.path_b_snr = self._calculate_path_b_snr()
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Stage 0: terminate on beacon drop, don't require RTH navigation
        if self._stage0_no_rth and self.command_state == "COMMAND C: RTH":
            terminated = True
            reward += 200.0  # Reduced bonus (no RTH leg)

        info["curriculum_stage"] = str(self.curriculum_stage)
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Curriculum promotion callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Monitors mean episode reward across a rolling window.
    Promotes curriculum stage when threshold is met.
    """
    STAGE_THRESHOLDS = {
        0: 80.0,   # Must consistently find close survivor
        1: 150.0,  # Must find + RTH in fixed maze
    }
    WINDOW = 50  # Episodes to average over

    def __init__(self, envs, verbose=1):
        super().__init__(verbose)
        self.envs = envs  # List of CurriculumEchosEnv instances
        self.episode_rewards = []
        self.current_stage = 0

    def _on_step(self) -> bool:
        # Collect episode rewards from infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        if len(self.episode_rewards) >= self.WINDOW:
            mean_reward = np.mean(self.episode_rewards[-self.WINDOW:])

            if self.current_stage in self.STAGE_THRESHOLDS:
                threshold = self.STAGE_THRESHOLDS[self.current_stage]
                if mean_reward >= threshold:
                    self.current_stage += 1
                    self._promote_all_envs()
                    if self.verbose:
                        print(f"\n[Curriculum] Mean reward {mean_reward:.1f} >= {threshold}. "
                              f"Promoted to Stage {self.current_stage}.\n")

        return True

    def _promote_all_envs(self):
        """Push new stage to all parallel envs via set_attr."""
        try:
            self.training_env.set_attr("curriculum_stage", self.current_stage)
        except Exception:
            # Fallback for non-vecenv
            for env in self.envs:
                env.curriculum_stage = self.current_stage


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(rank, seed=0):
    def _init():
        env = CurriculumEchosEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Echos PPO v3 — Curriculum Training")
    print("=" * 60)

    N_ENVS = 8          # Parallel environments — your Core Ultra 7 can handle this
    TOTAL_STEPS = 3_000_000  # 3M steps; curriculum will self-pace

    print(f"\nSpawning {N_ENVS} parallel environments...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    curriculum_cb = CurriculumCallback(envs=[], verbose=1)

    print("Spawning fresh PPO agent (v3)...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      # Small entropy bonus — encourages exploration in tight spaces
        tensorboard_log="./echos_tb_logs/",
    )

    print(f"\nCommencing curriculum training ({TOTAL_STEPS:,} steps)...")
    print("Stage 0: Close survivor, no RTH. Teaches gradient climb.")
    print("Stage 1: Fixed maze, full RTH loop.")
    print("Stage 2: Randomized maze, full mission generalization.\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=curriculum_cb,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    model.save("echos_ppo_v3")
    final_stage = curriculum_cb.current_stage
    print(f"\nTraining complete. Final curriculum stage reached: {final_stage}/2")
    print("Saved as 'echos_ppo_v3.zip'.")
    print("\nTo run the live sim: update sim.py PPO.load() to 'echos_ppo_v3'")

    vec_env.close()


if __name__ == "__main__":
    main()