# import conftest
import numpy as np

from envs.exploration_env import ExplorationEnv

def main():
    env = ExplorationEnv(
        n_agents=2,
    )

    obs, shared_obs, info = env.reset(seed=0)

    print("\n===== Scan Pattern =====")

    theta = np.asarray(env.theta)
    phi = np.asarray(env.phi)

    print("theta shape:", theta.shape)
    print("phi shape:", phi.shape)

    print("theta size:", theta.size)
    print("phi size:", phi.size)

    print("theta range:", theta.min(), theta.max())
    print("phi range:", phi.min(), phi.max())

    print("\n===== Environment Observation =====")

    print("obs shape:", obs.shape)
    print("single lidar shape:", obs[0].shape)
    print("single lidar size:", obs[0].size)

    print("\n===== Raw LiDAR =====")

    lidar = env.lidars[0]

    ranges = lidar.trace_rays(
        env.data,
        env.theta,
        env.phi,
    )

    ranges = np.asarray(ranges)

    print("raw ranges shape:", ranges.shape)
    print("raw ranges size:", ranges.size)
    print("raw ranges min:", np.nanmin(ranges))
    print("raw ranges max:", np.nanmax(ranges))

    print("\n===== Hit Points =====")

    points = np.asarray(
        lidar.get_hit_points()
    )

    print("points shape:", points.shape)

    if points.size > 0:
        print("first 5 points:")
        print(points[:5])


if __name__ == "__main__":
    main()

# 64 vertical channels/rings
########################################
# phi range: -0.43458698374658805 0.03490658503988659

# ===== Environment Observation =====
# obs shape: (2, 110016)
# single lidar shape: (110016,)
# single lidar size: 110016

# ===== Raw LiDAR =====
# raw ranges shape: (110016,)
# raw ranges size: 110016
# raw ranges min: 1.0687926627426805
# raw ranges max: 17.55959503548208

# ===== Hit Points =====
# points shape: (110016, 3)
# first 5 points:
# [[ 0.96944199  0.         -0.44999999]
#  [ 0.98866851  0.         -0.44999999]
#  [ 1.00853511  0.         -0.44999999]
#  [ 1.02907662  0.         -0.44999999]
#  [ 1.05033042  0.         -0.44999999]]