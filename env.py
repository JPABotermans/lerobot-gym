import gymnasium as gym
from src.base_so101_env import SO101Env

gym.register(
    id="base-sO101-env-v0",
    entry_point=SO101Env,
)


def make_env(
    n_envs: int = 1, use_async_envs: bool = False
) -> gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv:
    """
    Create vectorized environments for your custom task.

    Args:
        n_envs: Number of parallel environments
        use_async_envs: Whether to use AsyncVectorEnv or SyncVectorEnv

    Returns:
        gym.vector.VectorEnv or dict mapping suite names to vectorized envs
    """

    def _make_single_env():
        # Create your custom environment
        return gym.make("base-sO101-env-v0")

    # Choose vector environment type
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    # Create vectorized environment
    vec_env = env_cls([_make_single_env for _ in range(n_envs)])

    return vec_env
