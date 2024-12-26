import numpy

numpy.bool8 = numpy.bool_
import sys
import pickle
import gym
import gym_thing
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import time
from dmp import DMP
from policy import BallCatchDMPPolicy
from scipy.interpolate import interp1d

PATH_TO_HOME = str(pathlib.Path(__file__).parent.resolve())
PATH_TO_NLOPTJSON = str(
    (
        pathlib.Path(__file__).parent
        / "ballcatch_env"
        / "gym_thing"
        / "nlopt_optimization"
        / "config"
        / "nlopt_config_stationarybase.json"
    ).resolve()
)

TRAINED_DMP_PATH = "results/trained_dmp.pkl"


def make_env(
    online=True,
    dataset=None,
    n_substeps=1,
    gravity_factor_std=0.0,
    n_substeps_std=0.0,
    mass_std=0.0,
    act_gain_std=0.0,
    joint_noise_std=0.0,
    vicon_noise_std=0.0,
    control_noise_std=0.0,
    random_training_index=False,
):
    # Create thing env
    initial_arm_qpos = np.array([1, -0.3, -1, 0, 1.6, 0])
    initial_ball_qvel = np.array([-1, 1.5, 4.5])
    initial_ball_qpos = np.array([1.22, -1.6, 1.2])
    zero_base_qpos = np.array([0.0, 0.0, 0.0])

    training_data = None
    if not online:
        training_data = np.load(dataset)
        print("OFFLINE")

    env = gym.make(
        "ballcatch-v0",
        model_path="robot_with_cup.xml",
        nlopt_config_path=PATH_TO_NLOPTJSON,
        initial_arm_qpos=initial_arm_qpos,
        initial_ball_qvel=initial_ball_qvel,
        initial_ball_qpos=initial_ball_qpos,
        initial_base_qpos=zero_base_qpos,
        pos_rew_weight=0.1,
        rot_rew_weight=0.01,
        n_substeps=n_substeps,
        online=online,
        training_data=training_data,
        gravity_factor_std=gravity_factor_std,
        n_substeps_std=n_substeps_std,
        mass_std=mass_std,
        act_gain_std=act_gain_std,
        joint_noise_std=joint_noise_std,
        vicon_noise_std=vicon_noise_std,
        control_noise_std=control_noise_std,
        random_training_index=random_training_index,
    )
    env.reset()
    return env


def test_policy(eval_env, policy, eval_episodes=5, render_freq=1, seed=1):
    # Set seeds
    eval_env.seed(seed)
    np.random.seed(seed)

    avg_reward = 0.0
    successes = 0

    for eps in range(eval_episodes):
        print(f"\rEvaluating Episode {eps+1}/{eval_episodes}", end="")
        state, done, truncated = eval_env.reset(), False, False
        policy.set_goal(state=state, goal=eval_env.env.goal)
        while not (done or truncated):
            action = policy.select_action(np.array(state))
            state, reward, done, truncated, info = eval_env.step(action)
            if render_freq != 0 and eps % render_freq == 0:
                eval_env.render()
                # time.sleep(0.01)
            avg_reward += reward
        if truncated and not done:
            actual_done = False
        else:
            actual_done = done

        if actual_done:
            successes += 1

    avg_reward /= eval_episodes
    success_pct = float(successes) / eval_episodes

    print("")
    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}, success%: {:.3f}".format(eval_episodes, avg_reward, success_pct))
    print("---------------------------------------")
    print("")
    return avg_reward, success_pct


def load_dataset(dataset_path="data/demos.pkl"):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    return [traj[:, :6] for traj in dataset["trajectories"]]  # 1st 6 elements are joint angle


def q2_recons():
    # Load the dataset (first 6 elements are joint angles)
    trajectories = load_dataset()

    # Extract the first demonstration (i = 0)
    demo = trajectories[0]
    
    # Set the time step (dt) and compute demo time
    dt = 0.04
    demo_time = np.linspace(0, (demo.shape[0] - 1) * dt, demo.shape[0])

    # Initialize the DMP
    dmp = DMP()

    # Prepare the demonstration in the correct format for learning
    X = demo[None, :, :]  # Shape it as [num_demos, num_timesteps, dofs]
    T = demo_time[None, :]  # Shape it as [num_demos, num_timesteps]

    # Train the DMP on the first demonstration
    dmp.learn(X, T)

    # Set initial position (x0) and goal (g)
    x0 = demo[0]  # Initial position is the first timestep
    g = demo[-1]  # Goal position is the last timestep

    # Perform the rollout (reconstruction)
    rollout = dmp.rollout(dt=dt, tau=demo_time[-1], x0=x0, g=g)
    rollout_time = np.linspace(0, demo_time[-1], rollout.shape[0])

    # Plot the original demonstration and the reconstructed trajectory for each degree of freedom (DoF)
    for k in range(6):  # Assuming 6 DoFs
        plt.figure()
        plt.plot(demo_time, demo[:, k], label='GT')  # Ground truth trajectory
        plt.plot(rollout_time, rollout[:, k], label='DMP')  # Reconstructed trajectory
        plt.legend()
        plt.title(f"Reconstruction for DoF {k+1}")
        plt.savefig(f'results/recons_{k}.png')
        plt.show()


def interpolate_rollout(rollout, target_timesteps):
    """
    Interpolates the rollout to match the number of timesteps in the demo.
    rollout: The reconstructed trajectory from DMP, shape [timesteps, dofs].
    target_timesteps: The desired number of timesteps (from the demo).
    """
    num_rollout_steps = rollout.shape[0]
    rollout_time = np.linspace(0, 1, num_rollout_steps)  # normalized time
    target_time = np.linspace(0, 1, target_timesteps)  # same number of steps as the demo
    interpolator = interp1d(rollout_time, rollout, axis=0, kind='linear', fill_value="extrapolate")
    return interpolator(target_time)

def compute_rmse(demo, rollout):
    """
    Compute the Root Mean Squared Error between the demo and the rollout.
    demo: Ground truth trajectory, shape [timesteps, dofs].
    rollout: Reconstructed trajectory from DMP, shape [timesteps, dofs].
    """
    # If the timesteps of the demo and rollout are not the same, interpolate the rollout
    if demo.shape[0] != rollout.shape[0]:
        rollout = interpolate_rollout(rollout, demo.shape[0])

    return np.sqrt(np.mean((demo - rollout) ** 2))

def q2_tuning():
    # Load the dataset
    trajectories = load_dataset()

    # Set the time step (dt)
    dt = 0.04
    num_demos = len(trajectories)

    # Initialize variables for storing the best configuration and lowest RMSE
    best_nbasis = None
    best_K_vec = None
    best_rmse = float('inf')
    best_dmp = None

    # Experiment with different numbers of basis functions and gain parameters
    nbasis_options = [4]  # Different options for the number of basis functions
    K_vec_options = [np.array([30., 24., 40., 10., 20., 18.])]  # Different gain settings

    for nbasis in nbasis_options:
        for K_vec in K_vec_options:
            print(f"Evaluating DMP with nbasis={nbasis} and K_vec={K_vec[0]}...")

            # Initialize the DMP with the current settings
            dmp = DMP(nbasis=nbasis, K_vec=K_vec)

            # Interpolate the trajectories to have the same number of time steps
            X, T = DMP.interpolate(trajectories, initial_dt=dt)

            # Train the DMP on the full set of trajectories
            dmp.learn(X, T)

            # Compute RMSE across all demonstrations
            total_rmse = 0
            for i in range(num_demos):
                x0 = X[i, 0]  # Initial position of the trajectory
                g = X[i, -1]  # Goal position of the trajectory
                tau = T[i, -1]  # Duration of the trajectory

                # Perform rollout (reconstruct trajectory)
                rollout = dmp.rollout(dt=dt, tau=tau, x0=x0, g=g)

                # Compute RMSE between the demonstration and the rollout
                demo_rmse = compute_rmse(X[i], rollout)
                total_rmse += demo_rmse

            # Compute average RMSE across all demos
            avg_rmse = total_rmse / num_demos
            print(f"Avg RMSE for nbasis={nbasis}, K_vec={K_vec[0]}: {avg_rmse}")

            # If this is the best setting so far, store the DMP and RMSE
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_nbasis = nbasis
                best_K_vec = K_vec
                best_dmp = dmp

    # Save the best DMP to disk
    print(f"Best DMP found with nbasis={best_nbasis} and K_vec={best_K_vec[0]}, RMSE={best_rmse}")
    best_dmp.save(TRAINED_DMP_PATH)


def main():
    env = make_env(n_substeps=1)
    dmp = DMP.load(TRAINED_DMP_PATH)

    policy = BallCatchDMPPolicy(dmp, dt=env.dt)
    test_policy(env, policy, eval_episodes=20, render_freq=1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "recons":
        q2_recons()
    elif sys.argv[1] == "tuning":
        q2_tuning()