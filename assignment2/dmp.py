import numpy as np
import scipy.interpolate
import pickle


class DMP(object):
    """
        Dynamic Movement Primitives wlearned by Locally Weighted Regression (LWR).

    Implementation of P. Pastor, H. Hoffmann, T. Asfour and S. Schaal, "Learning and generalization of
    motor skills by learning from demonstration," 2009 IEEE International Conference on Robotics and
    Automation, 2009, pp. 763-768, doi: 10.1109/ROBOT.2009.5152385.
    """

    def __init__(self, nbasis=30, K_vec=10 * np.ones((6,)), weights=None):
        self.nbasis = nbasis  # Basis function number
        self.K_vec = K_vec

        self.K = np.diag(self.K_vec)  # Spring constant
        self.D = np.diag(2 * np.sqrt(self.K_vec))  # Damping constant, critically damped

        # used to determine the cutoff for s
        self.convergence_rate = 0.01
        self.alpha = -np.log(self.convergence_rate)

        # Creating basis functions and psi_matrix
        # Centers logarithmically distributed between 0.001 and 1
        self.basis_centers = np.logspace(-3, 0, num=self.nbasis)
        self.basis_variances = self.nbasis / (self.basis_centers**2)

        self.weights = weights

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                dict(
                    nbasis=self.nbasis,
                    K_vec=self.K_vec,
                    weights=self.weights,
                ),
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        dmp = cls(nbasis=data["nbasis"], K_vec=data["K_vec"], weights=data["weights"])
        return dmp

    @staticmethod
    def _gradient_nd(x, t):
        g1 = (x[:, 1:, :] - x[:, :-1, :]) / (t[:, 1:, None] - t[:, :-1, None])
        g = np.zeros_like(x)
        g[:, 0, :] = g1[:, 0, :]
        g[:, -1, :] = g1[:, -1, :]
        g[:, 1:-1, :] = (g1[:, 1:, :] + g1[:, :-1, :]) / 2
        return g

    def learn(self, X, T):
        """
        Learn the weights of the DMP using Locally Weighted Regression.

        X: demonstrated trajectories. Has shape [number of demos, number of timesteps,  dofs].
        T: corresponding timings. Has shape [number of demos, number of timesteps].
            It is assumed that trajectories start at t=0
        """
        #
        num_demos = X.shape[0]
        print(f"num demos is {num_demos}")
        num_timesteps = X.shape[1]
        print(f"num_timesteps is {num_timesteps}")

        # Initial position : [num_demos, num_timesteps, num_dofs]
        x0 = np.tile(X[:, 0, :][:, None, :], (1, num_timesteps, 1))
        print(f"Shape of x0 is {x0.shape}")
        # Goal position : [num_demos, num_timesteps, num_dofs]
        g = np.tile(X[:, -1, :][:, None, :], (1, num_timesteps, 1))
        print(f"Shape of g is {g.shape}")
        # Duration of the demonstrations
        tau = T[:, -1]
        print(f"shape of tau is {tau.shape}")

        # TODO: Compute s(t) for each step in the demonstrations
        s = np.exp(-self.alpha * T / tau[:, None])
        print(f"shape of s is {s.shape}")
        print(s)

        # TODO: Compute x_dot and x_ddot using numerical differentiation (np.graident)
        x_dot = self._gradient_nd(X, T)
        print(f"shape of x_dot is {x_dot.shape}")
        x_ddot = self._gradient_nd(x_dot, T)
        print(f"shape of x_ddot is {x_ddot.shape}")

        # TODO: Temporal Scaling by tau.
        v_dot = tau[:, None, None] * x_ddot
        print(f"shape of v_dot is {v_dot.shape}")
        v = tau[:, None, None] * x_dot
        print(f"shape of v is {v.shape}")

        # TODO: Compute f_target(s) based on Equation 8.
        # Step 1: Compute the inverse of self.K (which is 6x6)
        K_inv = np.linalg.inv(self.K)  # Inverse of a 6x6 matrix
        # Step 2: Apply the inverse of K to the last dimension using einsum for matrix multiplication
        f_s_target = (tau[:, None, None] * v_dot + np.einsum('ij,btj->bti', self.D, v))
        # Multiply by the inverse of K along the last axis (dof axis)
        f_s_target = np.einsum('bti,ij->btj', f_s_target, K_inv) - (g - X) + (g - x0) * s[:, :, None] 
        print(f"Shape of f_s_target is {f_s_target.shape}")

        # TODO: Compute psi(s). Hint: shape should be [num_demos, num_timesteps, nbasis]
        psi = np.exp(-self.basis_variances[None, None, :] * (s[:, :, None] - self.basis_centers[None, None, :])**2)
        print(f"Shape of psi is {psi.shape}")

        # TODO: Solve a least squares problem for the weights.
        # Hint: minimize f_target(s) - f_w(s) wrt to w
        # Hint: you can use np.linalg.lstsq
        # Reshape s for broadcasting
        s_repeated = s[:, :, None]

        # Compute the weighted psi(s)
        psi_weighted = psi * s_repeated

        # Least-squares solution to find the weights
        self.weights = np.linalg.lstsq(psi_weighted.reshape(-1, self.nbasis), f_s_target.reshape(-1, f_s_target.shape[-1]), rcond=None)[0]
        print(f"Shape of weights is {self.weights.shape}")

    def execute(self, t, dt, tau, x0, g, x_t, xdot_t):
        """
        Query the DMP at time t, with current position x_t, and velocity xdot_t.
        The parameter tau controls temporal scaling, x0 sets the initial position
        and g sets the goal for the trajectory.

        Returns the next position x_{t + dt} and velocity x_{t + dt}
        """
        if self.weights is None:
            raise ValueError("Cannot execute DMP before parameters are set by DMP.learn()")

        # Calculate s(t) by integrating
        s = np.exp(-self.alpha * t / tau) 

        # TODO: Compute f(s). See equation 3.
        psi = np.exp(-self.basis_variances * (s - self.basis_centers) ** 2)
        f_s = (np.dot(psi, self.weights) * s) / np.sum(psi)

        # Temporal Scaling
        v_t = tau * xdot_t

        # TODO: Calculate acceleration. Equation 6
        v_dot_t = (self.K @ (g - x_t) - self.D @ v_t - self.K @ (g - x0) * s + self.K @ f_s) / tau

        # TODO: Calculate next position and velocity
        xdot_tp1 = xdot_t + v_dot_t * dt
        x_tp1 = x_t + xdot_tp1 * dt

        return x_tp1, xdot_tp1

    def rollout(self, dt, tau, x0, g):
        time = 0
        x = x0
        x_dot = np.zeros_like(x0)
        X = [x0]

        while time <= tau:
            x, x_dot = self.execute(t=time, dt=dt, tau=tau, x0=x0, g=g, x_t=x, xdot_t=x_dot)
            time += dt
            X.append(x)

        return np.stack(X)

    @staticmethod
    def interpolate(trajectories, initial_dt):
        """
        Combine the given variable length trajectories into a fixed length array
        by interpolating shorter arrays to the maximum given sequence length.

        trajectories: A list of N arrays of shape (T_i, num_dofs) where T_i is the number
            of time steps in trajectory i
        initial_dt: A scalar corresponding to the duration of each time step.
        """
        # TODO Your code goes here ...
        max_length = max(len(traj) for traj in trajectories)
        dofs = trajectories[0].shape[1]

        X = np.zeros((len(trajectories), max_length, dofs))  # Interpolated trajectories
        T = np.zeros((len(trajectories), max_length))  # Time array for each trajectory

        for i, traj in enumerate(trajectories):
            num_timesteps = traj.shape[0]  # Original number of timesteps
            max_time = (num_timesteps - 1) * initial_dt  # Max time for this trajectory

            # Create a time array specific to this trajectory
            old_time = np.linspace(0, max_time, num_timesteps)

            # New uniform time array from 0 to the trajectory's max time with max_length timesteps
            new_time = np.linspace(0, max_time, max_length)

            # Use scipy's interp1d to create an interpolation function for each degree of freedom
            interpolator = scipy.interpolate.interp1d(old_time, traj, axis=0, kind='linear', fill_value="extrapolate")
            
            # Generate interpolated trajectory
            X[i] = interpolator(new_time)
            T[i] = new_time

        return X, T
