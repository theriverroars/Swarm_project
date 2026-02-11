import os
from dataclasses import dataclass
from typing import Optional, Union, List, Dict


import numpy as np
import casadi as cs
from scipy.interpolate import interp1d
from copy import copy
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import shutil


def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print("Error while removing directory: {0}".format(directory))


def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return cs.mtimes(rot_mat, v)


def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ]
        )

    else:
        rot_mat = cs.vertcat(
            cs.horzcat(
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ),
            cs.horzcat(
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qw * qx),
            ),
            cs.horzcat(
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2),
            ),
        )

    return rot_mat


def q_dot_q(q, r):
    """
    Applies the rotation of quaternion r to quaternion q. In order words, rotates quaternion q by r. Quaternion format:
    wxyz.

    :param q: 4-length numpy array or CasADi MX. Initial rotation
    :param r: 4-length numpy array or CasADi MX. Applied rotation
    :return: The quaternion q rotated by r, with the same format as in the input.
    """

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    t0 = rw * qw - rx * qx - ry * qy - rz * qz
    t1 = rw * qx + rx * qw - ry * qz + rz * qy
    t2 = rw * qy + rx * qz + ry * qw - rz * qx
    t3 = rw * qz - rx * qy + ry * qx + rz * qw

    return np.array([t0, t1, t2, t3])


def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)

    :param v: 3D numpy vector or CasADi MX
    :return: the corresponding skew-symmetric matrix of v with the same data type as v
    """

    if isinstance(v, np.ndarray):
        return np.array(
            [
                [0, -v[0], -v[1], -v[2]],
                [v[0], 0, v[2], -v[1]],
                [v[1], -v[2], 0, v[0]],
                [v[2], v[1], -v[0], 0],
            ]
        )

    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0),
    )


def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)


def separate_variables(traj):
    """
    Reshapes a trajectory into expected format.

    :param traj: N x 13 array representing the reference trajectory
    :return: A list with the components: Nx3 position trajectory array, Nx4 quaternion trajectory array, Nx3 velocity
    trajectory array, Nx3 body rate trajectory array
    """

    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]
    return [p_traj, a_traj, v_traj, r_traj]


def get_reference_chunk(
    reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling
):
    """
    Extracts the reference states and controls for the current MPC optimization given the over-sampled counterparts.

    :param reference_traj: The reference trajectory, which has been finely over-sampled by a factor of
    reference_over_sampling. It should be a vector of shape (Nx13), where N is the length of the trajectory in samples.
    :param reference_u: The reference controls, following the same requirements as reference_traj. Should be a vector
    of shape (Nx4).
    :param current_idx: Current index of the trajectory tracking. Should be an integer number between 0 and N-1.
    :param n_mpc_nodes: Number of MPC nodes considered in the optimization.
    :param reference_over_sampling: The over-sampling factor of the reference trajectories. Should be a positive
    integer.
    :return: Returns the chunks of reference selected for the current MPC iteration. Two numpy arrays will be returned:
        - An ((N+1)x13) array, corresponding to the reference trajectory. The first row is the state of current_idx.
        - An (Nx4) array, corresponding to the reference controls.
    """

    # Dense references
    ref_traj_chunk = reference_traj[
        current_idx : current_idx + (n_mpc_nodes + 1) * reference_over_sampling, :
    ]
    ref_u_chunk = reference_u[
        current_idx : current_idx + n_mpc_nodes * reference_over_sampling, :
    ]

    # Indices for down-sampling the reference to number of MPC nodes
    downsample_ref_ind = np.arange(
        0,
        min(reference_over_sampling * (n_mpc_nodes + 1), ref_traj_chunk.shape[0]),
        reference_over_sampling,
        dtype=int,
    )

    ref_traj_chunk = ref_traj_chunk[downsample_ref_ind, :]
    ref_u_chunk = ref_u_chunk[
        downsample_ref_ind[: max(len(downsample_ref_ind) - 1, 1)], :
    ]

    return ref_traj_chunk, ref_u_chunk


def get_reference_trajectory(eval_npz, idx, control_dt):
    """
    Gives the position and velocity reference as accurate.
    Orientation and Angular velocity are set to zero.
    Control reference is also set to 0.
    """

    eval_traj = eval_npz[idx]
    rows_not_nan = sum(~np.isnan(eval_traj[:, 1]))
    eval_traj = eval_traj[:rows_not_nan]

    trajectory_dt = eval_traj[1, 0] - eval_traj[0, 0]
    T = eval_traj[-1, 0]

    resampled_dt = control_dt
    resampled_t = np.arange(0, T, resampled_dt)

    interp = interp1d(eval_traj[:, 0], eval_traj[:, 1:], axis=0)
    resampled_traj = interp(resampled_t)
    resampled_traj = np.hstack((resampled_t.reshape(-1, 1), resampled_traj))

    assert resampled_traj.shape[1] == eval_traj.shape[1]
    # print("Resampled trajectory shape:", resampled_traj.shape)

    # Position, Quaternion, Velocity, Angular Velocity
    reference_trajectory = np.zeros((resampled_traj.shape[0], 13))
    reference_trajectory[:] = np.nan

    reference_inputs = np.zeros((resampled_traj.shape[0], 4))

    # position
    reference_trajectory[:, 0:3] = resampled_traj[:, 1:4]
    reference_trajectory[:, 3:7] = np.array([1, 0, 0, 0])
    reference_trajectory[:, 7:10] = resampled_traj[:, 8:11]
    reference_trajectory[:, 10:] = np.array([0, 0, 0])

    reference_timestamps = resampled_t

    return reference_trajectory, reference_inputs, reference_timestamps


def minimum_snap_trajectory_generator(traj_derivatives, yaw_derivatives, t_ref):
    """
    Follows the Minimum Snap Trajectory paper to generate a full trajectory given the position reference and its
    derivatives, and the yaw trajectory and its derivatives.

    :param traj_derivatives: np.array of shape 4x3xN. N corresponds to the length in samples of the trajectory, and:
        - The 4 components of the first dimension correspond to position, velocity, acceleration and jerk.
        - The 3 components of the second dimension correspond to x, y, z.
    :param yaw_derivatives: np.array of shape 2xN. N corresponds to the length in samples of the trajectory. The first
    row is the yaw trajectory, and the second row is the yaw time-derivative trajectory.
    :param t_ref: vector of length N, containing the reference times (starting from 0) for the trajectory.
    :param quad: Quadrotor3D object, corresponding to the quadrotor model that will track the generated reference.
    :type quad: Quadrotor3D
    :param map_limits: dictionary of map limits if available, None otherwise.
    :param plot: True if show a plot of the generated trajectory.
    :return: tuple of 3 arrays:
        - Nx13 array of generated reference trajectory. The 13 dimension contains the components: position_xyz,
        attitude_quaternion_wxyz, velocity_xyz, body_rate_xyz.
        - N array of reference timestamps. The same as in the input
        - Nx4 array of reference controls, corresponding to the four motors of the quadrotor.
    """

    discretization_dt = t_ref[1] - t_ref[0]
    len_traj = traj_derivatives.shape[2]

    # Add gravity to accelerations
    gravity = 9.81
    thrust = (
        traj_derivatives[2, :, :].T
        + np.tile(np.array([[0, 0, 1]]), (len_traj, 1)) * gravity
    )
    # Compute body axes
    z_b = thrust / np.sqrt(np.sum(thrust**2, 1))[:, np.newaxis]

    yawing = np.any(yaw_derivatives[0, :] != 0)

    rate = np.zeros((len_traj, 3))
    f_t = np.zeros((len_traj, 1))

    # new way to compute attitude:
    # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
    e_z = np.array([[0.0, 0.0, 1.0]])
    q_w = 1.0 + np.sum(e_z * z_b, axis=1)
    q_xyz = np.cross(e_z, z_b)
    q = 0.5 * np.concatenate([np.expand_dims(q_w, axis=1), q_xyz], axis=1)
    q = q / np.sqrt(np.sum(q**2, 1))[:, np.newaxis]

    # Use numerical differentiation of quaternions
    q_dot = np.gradient(q, axis=0) / discretization_dt
    w_int = np.zeros((len_traj, 3))
    for i in range(len_traj):
        w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]
    rate[:, 0] = w_int[:, 0]
    rate[:, 1] = w_int[:, 1]
    rate[:, 2] = w_int[:, 2]

    full_pos = traj_derivatives[0, :, :].T
    full_vel = traj_derivatives[1, :, :].T
    full_acc = traj_derivatives[2, :, :].T

    reference_traj = np.concatenate((full_pos, q, full_vel, rate), 1)

    reference_inputs = np.zeros((len_traj, 4))

    return reference_traj, reference_inputs, t_ref


def lemniscate_trajectory(discretization_dt, radius, z, lin_acc, v_max):
    """

    :param quad:
    :param discretization_dt:
    :param radius:
    :param z:
    :param lin_acc:
    :param clockwise:
    :param yawing:
    :param v_max:
    :param map_name:
    :param plot:
    :return:
    """

    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = (
        transition_t_vec[-1]
        + np.arange(0, coasting_duration, discretization_dt)
        + discretization_dt
    )
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = (
        down_coasting_t_vec[-1]
        + np.arange(0, ramp_up_t, discretization_dt)
        + discretization_dt
    )
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate(
        (
            ramp_t_vec,
            coasting_t_vec,
            transition_t_vec,
            down_coasting_t_vec,
            ramp_up_t_vec,
        )
    )
    alpha_vec = np.concatenate(
        (
            ramp_up_alpha,
            coasting_alpha,
            transition_alpha,
            down_coasting_alpha,
            ramp_up_alpha_end,
        )
    )

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Adaption: we achieve the highest spikes in the bodyrates when passing through the 'center' part of the figure-8
    # This leads to negative reference thrusts.
    # Let's see if we can alleviate this by adapting the z-reference in these parts to add some acceleration in the
    # z-component
    z_dim = 0.0

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = (
        radius * (np.sin(angle_vec) * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    )
    pos_traj_z = -z_dim * np.cos(4.0 * angle_vec)[np.newaxis, np.newaxis, :] + z

    vel_traj_x = -radius * (w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = (
        radius
        * (w_vec * np.cos(angle_vec) ** 2 - w_vec * np.sin(angle_vec) ** 2)[
            np.newaxis, np.newaxis, :
        ]
    )
    vel_traj_z = (
        4.0 * z_dim * w_vec * np.sin(4.0 * angle_vec)[np.newaxis, np.newaxis, :]
    )

    x_ref = pos_traj_x.reshape(-1)
    y_ref = pos_traj_y.reshape(-1)
    z_ref = pos_traj_z.reshape(-1)

    vx_ref = vel_traj_x.reshape(-1)
    vy_ref = vel_traj_y.reshape(-1)
    vz_ref = vel_traj_z.reshape(-1)

    position_ref = np.vstack((x_ref, y_ref, z_ref)).T
    velocity_ref = np.vstack((vx_ref, vy_ref, vz_ref)).T
    acceleration_ref = np.gradient(velocity_ref, axis=0) / discretization_dt
    jerk_ref = np.gradient(acceleration_ref, axis=0) / discretization_dt

    traj_derivatives = np.stack(
        (position_ref, velocity_ref, acceleration_ref, jerk_ref), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(
        traj_derivatives, np.zeros((2, len(t_ref))), t_ref
    )


def satellite_orbit(radius=1.0, loops=3, total_time=10.0, dt=0.01):
    """
    Generates a 3D parametric 'satellite orbit' trajectory with a smooth speed
    profile: speed ramps up from 0, reaches a max, then ramps down to 0.

    Parameters
    ----------
    radius : float
        Radius of the orbit.
    loops : int
        How many times the drone will loop around in the path.
    total_time : float
        Total duration (in seconds) for completing the entire trajectory.
    n_points : int
        Number of points to sample along the trajectory.

    Returns
    -------
    t : 1D numpy array
        Time array, from 0 to total_time.
    x, y, z : 1D numpy arrays
        Coordinates of the trajectory in 3D.
    """
    # Create a time array from 0 to total_time
    n_points = int(total_time / dt)
    t = np.linspace(0, total_time, n_points)

    # Dimensionless progress parameter s(t) in [0,1],
    # with a smooth 'cosine ramp' from 0 to 1
    # s(0) = 0 and s(T) = 1
    s = 0.5 * (1 - np.cos(np.pi * t / total_time))

    # Orbit angles:
    #   theta(t) = loops * 2π * s(t)
    #   phi(t)   = 2π * s(t)
    theta = loops * 2 * np.pi * s
    phi = 2 * np.pi * s

    # Parametric equations for the rotated circle:
    x = -radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta)
    z = radius * np.cos(theta) * np.cos(phi)

    z += 1.0

    position = np.vstack((x, y, z)).T
    velocity = np.gradient(position, axis=0) / dt
    acceleration = np.gradient(velocity, axis=0) / dt
    jerk = np.gradient(acceleration, axis=0) / dt

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(traj_derivatives, np.zeros((2, len(t))), t)


def octahedron_trajectory(scale=0.5, total_time=30, dt=0.01, hover_time=2.5):
    """
    Generates a continuous 3D trajectory that traces an Eulerian circuit
    along the edges of a regular octahedron, using a fixed time step dt.

    The six vertices of the octahedron (scaled by `scale`) are at:
      0: ( 1,  0,  0)
      1: (-1,  0,  0)
      2: ( 0,  1,  0)
      3: ( 0, -1,  0)
      4: ( 0,  0,  1)
      5: ( 0,  0, -1)

    An Eulerian circuit is computed (since every vertex has even degree), and then the
    path is reparameterized with constant speed. Positions are computed at time intervals
    of dt (here, 0.01 s).

    Parameters
    ----------
    scale : float
        Scale factor for the size of the octahedron.
    total_time : float
        Total flight time (seconds).
    dt : float
        Fixed time step (seconds).

    Returns
    -------
    t_total : 1D numpy array
        Array of time stamps from 0 to total_time in steps of dt.
    x, y, z : 1D numpy arrays
        Coordinates of the trajectory.
    waypoints : 2D numpy array
        The key vertices (in the order visited in the Eulerian circuit).
    """
    # Define the six vertices of a regular octahedron.
    vertices = {
        0: np.array([1, 0, 0]),
        1: np.array([-1, 0, 0]),
        2: np.array([0, 1, 0]),
        3: np.array([0, -1, 0]),
        4: np.array([0, 0, 1]),
        5: np.array([0, 0, -1]),
    }
    # Scale the vertices.
    for k in vertices:
        vertices[k] = vertices[k] * scale

    # Define the connectivity of the octahedron.
    graph = {
        0: [2, 3, 4, 5],
        1: [2, 3, 4, 5],
        2: [0, 1, 4, 5],
        3: [0, 1, 4, 5],
        4: [0, 1, 2, 3],
        5: [0, 1, 2, 3],
    }

    # Compute an Eulerian circuit starting from vertex 0.
    circuit = find_eulerian_circuit(graph, start=0)
    # Convert vertex indices to coordinates.
    waypoints = np.array([vertices[v] for v in circuit])

    # Compute segment lengths and the cumulative arc-length along the path.
    segment_lengths = []
    cum_length = [0]  # cumulative distance starts at 0
    for i in range(len(waypoints) - 1):
        seg_len = np.linalg.norm(waypoints[i + 1] - waypoints[i])
        segment_lengths.append(seg_len)
        cum_length.append(cum_length[-1] + seg_len)
    cum_length = np.array(cum_length)
    total_length = cum_length[-1]

    # Determine constant speed needed to traverse the full length in total_time.
    speed = total_length / total_time

    # Create the time array with fixed dt.
    t_total = np.arange(0, total_time + dt, dt)

    # For each time, determine the arc-length traveled.
    s_values = speed * t_total

    # For each s, find the corresponding segment and interpolate the position.
    traj_points = []
    seg_index = 0
    for s in s_values:
        # If s equals total_length (or very close), use the last waypoint.
        if s >= total_length:
            traj_points.append(waypoints[-1])
            continue
        # Find the segment in which s falls.
        while seg_index < len(cum_length) - 1 and s > cum_length[seg_index + 1]:
            seg_index += 1
        # Interpolate between waypoints[seg_index] and waypoints[seg_index+1].
        s0 = cum_length[seg_index]
        s1 = cum_length[seg_index + 1]
        f = (s - s0) / (s1 - s0)
        point = waypoints[seg_index] + f * (
            waypoints[seg_index + 1] - waypoints[seg_index]
        )
        traj_points.append(point)

        # Now add the hover phase at the final endpoint.
    n_extra = int(hover_time / dt)
    # Create extra time stamps starting from the end of the trajectory.
    t_hover = t_total[-1] + np.arange(dt, hover_time + dt, dt)
    # For these extra time steps, the position remains constant at the final point.
    hover_points = np.tile(waypoints[-1], (len(t_hover), 1))

    # Concatenate the trajectory with the hover phase.
    t_total = np.concatenate([t_total, t_hover])
    traj_points = np.concatenate([traj_points, hover_points], axis=0)

    x = traj_points[:, 0]
    y = traj_points[:, 1]
    z = traj_points[:, 2]

    position = np.vstack((x, y, z)).T
    z += 1.0

    print("Zminmax", z.max(), z.min())

    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    vz = np.gradient(z) / dt

    ax = np.zeros_like(vx)
    ay = np.zeros_like(vy)
    az = np.zeros_like(vz)

    position = np.vstack((x, y, z)).T
    velocity = np.vstack((vx, vy, vz)).T
    acceleration = np.vstack((ax, ay, az)).T
    jerk = np.zeros_like(acceleration)

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(
        traj_derivatives, np.zeros((2, len(t_total))), t_total
    )


def find_eulerian_circuit(graph, start):
    """
    Compute an Eulerian circuit in an undirected graph using Hierholzer's algorithm.

    Parameters
    ----------
    graph : dict
        Dictionary mapping each vertex to a list of neighboring vertices.
    start : hashable
        The starting vertex.

    Returns
    -------
    circuit : list
        A list of vertices representing the Eulerian circuit.
    """
    # Make a copy so we can modify the graph.
    graph_copy = {u: list(neighbors) for u, neighbors in graph.items()}
    circuit = []
    stack = [start]

    while stack:
        v = stack[-1]
        if graph_copy[v]:
            # Choose an arbitrary neighbor, remove the edge, and traverse.
            w = graph_copy[v].pop()
            graph_copy[w].remove(v)
            stack.append(w)
        else:
            circuit.append(stack.pop())

    return circuit


def random_trajectory(seed, total_time=16, dt=0.01):

    rng = np.random.default_rng(seed=seed)
    num_waypoints = 5
    t_waypoints = np.linspace(0, total_time, num_waypoints)

    spline_durations = np.diff(t_waypoints)
    num_splines = len(spline_durations)

    waypoints_xy = rng.uniform(-1.5, 1.5, (num_waypoints, 2))
    waypoints_z = rng.uniform(0.5, 1.0, num_waypoints)

    waypoints_xy[-1] = waypoints_xy[0]
    waypoints_z[-1] = waypoints_z[0]

    waypoints = np.hstack((waypoints_xy, waypoints_z.reshape(-1, 1)))

    velocity_constraints = np.zeros((num_waypoints, 3))
    velocity_constraints[1:-1, :2] = rng.uniform(-1.0, 1.0, (num_waypoints - 2, 2))
    velocity_constraints[1:-1, 2] = rng.uniform(-0.75, 0.75, (num_waypoints - 2))

    acceleration_constraints = np.zeros((num_waypoints, 3))
    # acceleration_constraints[1:-1] = rng.uniform(0.0, 0.1, (num_waypoints - 2, 3))

    boundary_conditions = np.zeros((num_splines * 6, 3))
    coeffMatrix = np.zeros((num_splines * 6, num_splines * 6))

    for i in range(num_splines):
        T = spline_durations[i]
        idx = i * 6

        boundary_conditions[idx : idx + 6] = np.array(
            [
                waypoints[i],
                velocity_constraints[i],
                acceleration_constraints[i],
                waypoints[i + 1],
                velocity_constraints[i + 1],
                acceleration_constraints[i + 1],
            ]
        )
        coeffMatrix[idx : idx + 6, idx : idx + 6] = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [T**5, T**4, T**3, T**2, T, 1],
                [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
            ]
        )

    xTrajCoeff, yTrajCoeff, zTrajCoeff = np.linalg.solve(
        coeffMatrix, boundary_conditions
    ).T

    xVelCoeff, yVelCoeff, zVelCoeff = [], [], []

    for i in range(num_splines):
        idx = i * 6
        xVelCoeff.append(np.polyder(xTrajCoeff[idx : idx + 6]))
        yVelCoeff.append(np.polyder(yTrajCoeff[idx : idx + 6]))
        zVelCoeff.append(np.polyder(zTrajCoeff[idx : idx + 6]))

    xVelCoeff = np.array(xVelCoeff)
    yVelCoeff = np.array(yVelCoeff)
    zVelCoeff = np.array(zVelCoeff)

    t = np.linspace(0, total_time, int(total_time / dt))
    reference_trajectory = np.zeros((len(t), 6))

    for i in range(num_splines):
        T = spline_durations[i]
        t_idx = np.logical_and(t >= t_waypoints[i], t <= t_waypoints[i + 1])
        t_rel = t[t_idx] - t_waypoints[i]

        reference_trajectory[t_idx, 0:3] = np.array(
            [
                np.polyval(xTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(yTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(zTrajCoeff[i * 6 : i * 6 + 6], t_rel),
            ]
        ).T

        reference_trajectory[t_idx, 3:6] = np.array(
            [
                np.polyval(xVelCoeff[i], t_rel),
                np.polyval(yVelCoeff[i], t_rel),
                np.polyval(zVelCoeff[i], t_rel),
            ]
        ).T

    position = reference_trajectory[:, :3]
    position[:, 2] += 0.15
    velocity = reference_trajectory[:, 3:]
    speed_velocity = np.linalg.norm(velocity, axis=1)

    print(f"Max speed: {np.max(speed_velocity)}")

    acceleration = np.gradient(velocity, axis=0) / dt
    jerk = np.gradient(acceleration, axis=0) / dt

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(traj_derivatives, np.zeros((2, len(t))), t)


def random_looped_trajectory(seed, total_time=16, dt=0.01):

    rng = np.random.default_rng(seed=seed)
    num_waypoints_per_loop = 5
    num_loops = 3
    num_waypoints = num_waypoints_per_loop * num_loops + 2
    # plus 2 for the start and end points
    t_waypoints = np.linspace(0, total_time, num_waypoints)
    spline_durations = np.diff(t_waypoints)
    num_splines = len(spline_durations)

    _start_point_xy = rng.uniform(-1.75, 1.75, 2)
    _start_point_z = rng.uniform(0.5, 1.25)
    _start_point = np.hstack((_start_point_xy, _start_point_z))

    _end_point = _start_point.copy()

    _loop_waypoints_xy = rng.uniform(-1.75, 1.75, (num_waypoints_per_loop, 2))
    _loop_waypoints_z = rng.uniform(0.5, 1.0, num_waypoints_per_loop)
    _loop_waypoints = np.hstack((_loop_waypoints_xy, _loop_waypoints_z.reshape(-1, 1)))

    _all_loop_waypoints = np.tile(_loop_waypoints, (num_loops, 1))
    # print(_all_loop_waypoints.shape)

    waypoints = np.vstack((_start_point, _all_loop_waypoints, _end_point))
    # print(waypoints.shape)

    _loop_velocity_constraints = np.zeros((num_waypoints_per_loop, 3))
    _loop_velocity_constraints[:, :2] = rng.uniform(
        -1.25, 1.25, (num_waypoints_per_loop, 2)
    )
    _loop_velocity_constraints[:, 2] = rng.uniform(-0.75, 0.75, num_waypoints_per_loop)
    _start_velocity_constraints = np.zeros(3)
    _end_velocity_constraints = np.zeros(3)

    velocity_constraints = np.vstack(
        (
            _start_velocity_constraints,
            np.tile(_loop_velocity_constraints, (num_loops, 1)),
            _end_velocity_constraints,
        )
    )

    acceleration_constraints = np.zeros_like(velocity_constraints)

    boundary_conditions = np.zeros((num_splines * 6, 3))
    coeffMatrix = np.zeros((num_splines * 6, num_splines * 6))

    for i in range(num_splines):
        T = spline_durations[i]
        idx = i * 6

        boundary_conditions[idx : idx + 6] = np.array(
            [
                waypoints[i],
                velocity_constraints[i],
                acceleration_constraints[i],
                waypoints[i + 1],
                velocity_constraints[i + 1],
                acceleration_constraints[i + 1],
            ]
        )
        coeffMatrix[idx : idx + 6, idx : idx + 6] = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [T**5, T**4, T**3, T**2, T, 1],
                [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
            ]
        )

    xTrajCoeff, yTrajCoeff, zTrajCoeff = np.linalg.solve(
        coeffMatrix, boundary_conditions
    ).T

    xVelCoeff, yVelCoeff, zVelCoeff = [], [], []

    for i in range(num_splines):
        idx = i * 6
        xVelCoeff.append(np.polyder(xTrajCoeff[idx : idx + 6]))
        yVelCoeff.append(np.polyder(yTrajCoeff[idx : idx + 6]))
        zVelCoeff.append(np.polyder(zTrajCoeff[idx : idx + 6]))

    xVelCoeff = np.array(xVelCoeff)
    yVelCoeff = np.array(yVelCoeff)
    zVelCoeff = np.array(zVelCoeff)

    t = np.linspace(0, total_time, int(total_time / dt))
    reference_trajectory = np.zeros((len(t), 6))

    for i in range(num_splines):
        T = spline_durations[i]
        t_idx = np.logical_and(t >= t_waypoints[i], t <= t_waypoints[i + 1])
        t_rel = t[t_idx] - t_waypoints[i]

        reference_trajectory[t_idx, 0:3] = np.array(
            [
                np.polyval(xTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(yTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(zTrajCoeff[i * 6 : i * 6 + 6], t_rel),
            ]
        ).T

        reference_trajectory[t_idx, 3:6] = np.array(
            [
                np.polyval(xVelCoeff[i], t_rel),
                np.polyval(yVelCoeff[i], t_rel),
                np.polyval(zVelCoeff[i], t_rel),
            ]
        ).T

    position = reference_trajectory[:, :3]
    position[:, 2] += 0.15
    velocity = reference_trajectory[:, 3:]
    speed_velocity = np.linalg.norm(velocity, axis=1)

    print(f"Max speed: {np.max(speed_velocity)}")

    acceleration = np.gradient(velocity, axis=0) / dt
    jerk = np.gradient(acceleration, axis=0) / dt

    # return t, reference_trajectory[:, :3], reference_trajectory[:, 3:]

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(traj_derivatives, np.zeros((2, len(t))), t)


@dataclass
class State:
    """State of the quadrotor in 3D space"""

    pos: np.ndarray
    vel: np.ndarray
    # quaternion format [w, x, y, z]
    quat: np.ndarray
    a_rate: np.ndarray


class Quadrotor3D:

    def __init__(self, emulate_state: List[str]):

        # state space
        self.state: State = State(
            pos=np.zeros(3),
            vel=np.zeros(3),
            quat=np.array([1, 0, 0, 0]),  # quaternion format [w, x, y, z]
            a_rate=np.zeros(3),
        )

        # which state to emulate
        # NOTE: the state is a list of 4 elements: [pos, quat, vel, a_rate]
        emulate_list = ["pos", "quat", "vel", "a_rate"]
        assert all(
            [state in emulate_list for state in emulate_state]
        ), f"emulate_state should be in {emulate_list}"
        self.emulate_state = emulate_state

        # input
        self.max_input_value = np.array([1.0, 1.0, 1.0, 1.0])  # max thrust and torques
        self.min_input_value = np.array(
            [0.0, -1.0, -1.0, -1.0]
        )  # min thrust and torques

        # gravity
        self.g = np.array([0, 0, -9.81])

        # actuation
        self.u = np.zeros(4)  # N.
        # NOTE: the vector u is the vector containing the thrust and torques that are produced by the rotors
        # thrust goes from 0 to 1.0 , with 1.0 being the maximum thrust and 0.0 being the minimum thrust
        # torques goes from -1.0 to 1.0, with 1.0 being the max torque in +ve direction
        # and -1.0 being the max torque in -ve direction, with 0.0 being no torque

        # dynamics
        self.mass = 0.033  # kg
        self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])
        self.arm_length = 0.046  # m
        self.prop_const = 0.006
        self.thrust2weight = 2.0

        self.HOVER_THRUST_NEWTONS = self.mass * -self.g[2]  # N
        self.HOVER_THRUST = self.HOVER_THRUST_NEWTONS / self.thrust2weight  # unitless
        print(
            f"Hover thrust: {self.HOVER_THRUST_NEWTONS} N, {self.HOVER_THRUST} unitless"
        )
        self.max_thrust = self.thrust2weight * self.HOVER_THRUST_NEWTONS  # N
        self.max_torque = np.array(
            [
                self.max_thrust / 4 * self.arm_length,
                self.max_thrust / 4 * self.arm_length,
                2 * self.max_thrust / 4 * self.prop_const,
            ]
        )  # N.m

        self.invJ = 1 / self.J
        self.step_counter = 0
        self.reference_trajectory = None

    def post_process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Post process the action to be in the range of the quadrotor
        :param action: action to be post processed
        :return: post processed action
        """
        action = np.clip(action, -1.0, 1.0)

        force_torque = np.zeros(4)
        force_torque[0] = action[0] * self.max_thrust
        force_torque[1:] = action[1:] * self.max_torque

        return force_torque

    def get_state(self, as_numpy: bool = False):
        """
        Get the state of the quadrotor
        :return: state of the quadrotor
        """
        if not as_numpy:
            return [
                self.state.pos,
                self.state.quat,
                self.state.vel,
                self.state.a_rate,
            ]
        else:

            return [
                self.state.pos[0],
                self.state.pos[1],
                self.state.pos[2],
                self.state.quat[0],
                self.state.quat[1],
                self.state.quat[2],
                self.state.quat[3],
                self.state.vel[0],
                self.state.vel[1],
                self.state.vel[2],
                self.state.a_rate[0],
                self.state.a_rate[1],
                self.state.a_rate[2],
            ]

    def set_state(self, state: Union[State, List]) -> None:
        """
        Set the state of the quadrotor
        :param state: state to be set
        """
        if isinstance(state, State):
            self.state = state
        else:
            self.state.pos = state[0]
            self.state.quat = state[1]
            self.state.vel = state[2]
            self.state.a_rate = state[3]

    def step(self, u, dt):
        """
        Step the quadrotor dynamics
        :param u: input to the quadrotor
        :param dt: time step
        """
        u = self.post_process_action(u)
        self.u = u

        x = self.get_state(as_numpy=False)

        # RK4 integration
        k1 = [
            self.f_pos(x),
            self.f_att(x),
            self.f_vel(x, self.u),
            self.f_rate(x, self.u),
        ]
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(4)]
        k2 = [
            self.f_pos(x_aux),
            self.f_att(x_aux),
            self.f_vel(x_aux, self.u),
            self.f_rate(x_aux, self.u),
        ]
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(4)]
        k3 = [
            self.f_pos(x_aux),
            self.f_att(x_aux),
            self.f_vel(x_aux, self.u),
            self.f_rate(x_aux, self.u),
        ]
        x_aux = [x[i] + dt * k3[i] for i in range(4)]
        k4 = [
            self.f_pos(x_aux),
            self.f_att(x_aux),
            self.f_vel(x_aux, self.u),
            self.f_rate(x_aux, self.u),
        ]
        x = [x[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)]
        x[1] = x[1] / np.linalg.norm(x[1])  # normalize quaternion

        # & STATE UPDATE
        pos, quat, vel, rate = x

        ## * CALLBACK LISTS
        # position_callback_list = ["vel", "quat", "a_rate"]
        # velocity_callback_list = ["quat", "a_rate"]
        # attitude_callback_list = ["a_rate"]
        # a_rate_callback_list = []

        # Only update states that need to be emulated
        state_map = {"pos": pos, "quat": quat, "vel": vel, "a_rate": rate}

        # For each state type that should be emulated, update it
        for state_type in self.emulate_state:
            if state_type == "pos":
                self.state.pos = state_map["pos"]
            elif state_type == "quat":
                self.state.quat = state_map["quat"]
            elif state_type == "vel":
                self.state.vel = state_map["vel"]
            elif state_type == "a_rate":
                self.state.a_rate = state_map["a_rate"]

        self.step_counter += 1

        return pos, vel, self.f_vel(x, u), quat, rate

    def f_pos(self, x):
        """
        Time-derivative of the position vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: position differential increment (vector): d[pos_x; pos_y]/dt
        """

        vel = x[2]
        return vel

    def f_att(self, x):
        """
        Time-derivative of the attitude in quaternion form
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
        """

        rate = x[3]
        angle_quaternion = x[1]

        return 1 / 2 * skew_symmetric(rate).dot(angle_quaternion)

    def f_vel(self, x, u):
        """
        Time-derivative of the velocity vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [ Thrust, Torque_x, Torque_y, Torque_z ]
        :param f_d: disturbance force vector (3-dimensional)
        :return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
        """

        a_thrust = u[0] / self.mass
        a_thrust = np.array([0, 0, a_thrust])

        angle_quaternion = x[1]

        return self.g + v_dot_q(a_thrust, angle_quaternion)

    def f_rate(self, x, u):
        """
        Time-derivative of the angular rate
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [ Thrust, Torque_x, Torque_y, Torque_z ]
        :return: angular rate differential increment (scalar): dr/dt
        """

        rate = x[3]
        return np.array(
            [
                1 / self.J[0] * (u[1] + (self.J[1] - self.J[2]) * rate[1] * rate[2]),
                1 / self.J[1] * (u[2] + (self.J[2] - self.J[0]) * rate[2] * rate[0]),
                1 / self.J[2] * (u[3] + (self.J[0] - self.J[1]) * rate[0] * rate[1]),
            ]
        )


class Quadrotor3DMPC:
    def __init__(
        self,
        quadrotor: Quadrotor3D,
        simulation_dt: float,
        n_mpc_node: int,
        t_horizon: float,
        q_cost: np.ndarray,
        r_cost: np.ndarray,
        q_mask: np.ndarray,
    ):

        self.quadrotor = quadrotor
        self.simulation_dt = simulation_dt

        self.quadrotor_opt = Quadrotor3DOptimizer(
            quadrotor, t_horizon, n_mpc_node, q_cost, r_cost, q_mask
        )

    def clear(self):
        self.quadrotor_opt.clear_acados_models()

    def get_state(self):
        """
        Returns the state of the drone, with the angle described as a wxyz quaternion
        :return: 13x1 array with the drone state: [p_xyz, a_wxyz, v_xyz, r_xyz]
        """
        x = np.expand_dims(self.quadrotor.get_state(as_numpy=True), 1)
        return x

    def set_reference(self, x_reference, u_reference=None):
        """
        Sets a target state for the MPC optimizer
        :param x_reference: list with 4 sub-components (position, angle quaternion, velocity, body rate). If these four
        are lists, then this means a single target point is used. If they are Nx3 and Nx4 (for quaternion) numpy arrays,
        then they are interpreted as a sequence of N tracking points.
        :param u_reference: Optional target for the optimized control inputs
        """

        if isinstance(x_reference[0], list):
            # Target state is just a point
            return self.quadrotor_opt.set_reference_state(x_reference, u_reference)
        else:
            # Target state is a sequence
            return self.quadrotor_opt.set_reference_trajectory(x_reference, u_reference)

    def optimize(self, use_model=0, return_x=False):
        """
        Runs MPC optimization to reach the pre-set target.
        :param use_model: Integer. Select which dynamics model to use from the available options.
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.

        :return: 4*m vector of optimized control inputs with the format: [u_1(0), u_2(0), u_3(0), u_4(0), u_1(1), ...,
        u_3(m-1), u_4(m-1)]. If return_x is True, will also return a vector of shape N+1 x 13 containing the optimized
        state prediction.
        """

        quad_current_state = self.quadrotor.get_state(as_numpy=True)

        # Remove rate state for simplified model NLP
        out_out = self.quadrotor_opt.run_optimization(
            quad_current_state,
            use_model=use_model,
            return_x=return_x,
        )
        return out_out

    def simulate(self, ref_u):
        """
        Runs the simulation step for the dynamics model of the quadrotor 3D.

        :param ref_u: 4-length reference vector of control inputs
        """

        # Simulate step
        return self.quadrotor.step(ref_u, self.simulation_dt)


class Quadrotor3DOptimizer:
    def __init__(
        self,
        quadrotor: Quadrotor3D,
        t_horizon: float,
        n_mpc_node: int,
        q_cost: np.ndarray,
        r_cost: np.ndarray,
        q_mask: np.ndarray,
        model_name: str = "Quadrotor3D",
        solver_options: Union[None, Dict] = None,
        acados_path_postfix: Optional[str] = None,
    ):

        self.quadrotor = quadrotor
        self.T = t_horizon
        self.N = n_mpc_node

        self.max_u = quadrotor.max_input_value
        self.min_u = quadrotor.min_input_value

        # declare model variables
        self.p = cs.MX.sym("p", 3)  # position
        self.q = cs.MX.sym("q", 4)  # quaternion format [w, x, y, z]
        self.v = cs.MX.sym("v", 3)  # velocity
        self.r = cs.MX.sym("r", 3)  # angular rate

        # full state vector
        self.x = cs.vertcat(self.p, self.q, self.v, self.r)
        self.state_dim = 13

        # control input vector
        self.forces = cs.MX.sym("forces")
        self.tau1 = cs.MX.sym("tau1")
        self.tau2 = cs.MX.sym("tau2")
        self.tau3 = cs.MX.sym("tau3")

        self.u = cs.vertcat(self.forces, self.tau1, self.tau2, self.tau3)

        # Nominal model equations
        self.quad_x_dot_nominal = self.quad_dynamics()

        # Initialize objective function, 0 target state and integration equations
        self.L = None
        self.target = None

        # Build full model. Will have 13 variables.
        acados_model, dynamics_equations = self.acados_setup_model(
            self.quad_x_dot_nominal(x=self.x, u=self.u)["x_dot"], model_name
        )

        # Convert dynamics variables to functions of the state and input vectors
        self.quad_xdot = {}

        for dyn_model_idx in dynamics_equations.keys():
            dyn = dynamics_equations[dyn_model_idx]
            self.quad_xdot[dyn_model_idx] = cs.Function(
                "x_dot", [self.x, self.u], [dyn], ["x", "u"], ["x_dot"]
            )

        # ### Setup and compile Acados OCP solvers ### #
        self.acados_ocp_solver = {}

        # Add one more weight to the rotation (use quaternion norm weighting in acados)
        q_diagonal = np.concatenate(
            (q_cost[:3], np.mean(q_cost[3:6])[np.newaxis], q_cost[3:])
        )
        if q_mask is not None:
            q_mask = np.concatenate((q_mask[:3], np.zeros(1), q_mask[3:]))
            q_diagonal *= q_mask

        self.acados_models_dir = "acados_models"
        if acados_path_postfix is not None:
            self.acados_models_dir = self.acados_models_dir + "/" + acados_path_postfix
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))

        for key, key_model in zip(acados_model.keys(), acados_model.values()):
            nx = key_model.x.size()[0]
            nu = key_model.u.size()[0]
            ny = nx + nu
            n_param = key_model.p.size()[0] if isinstance(key_model.p, cs.MX) else 0

            acados_source_path = os.environ["ACADOS_SOURCE_DIR"]

            # Create OCP object to formulate the optimization

            ocp = AcadosOcp()
            ocp.acados_include_path = acados_source_path + "/include"
            ocp.acados_lib_path = acados_source_path + "/lib"
            ocp.code_export_directory = (
                self.acados_models_dir + "/" + "c_generated_code"
            )
            ocp.model = key_model
            ocp.solver_options.N_horizon = self.N

            ocp.solver_options.tf = t_horizon

            # Initialaise partamteres
            ocp.dims.np = n_param
            ocp.parameter_values = np.zeros(n_param)

            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"

            ocp.cost.W = np.diag(np.concatenate((q_diagonal, r_cost)))
            ocp.cost.W_e = np.diag(q_diagonal)
            terminal_cost = (
                0
                if solver_options is None or not solver_options["terminal_cost"]
                else 1
            )
            ocp.cost.W_e *= terminal_cost

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:nx, :nx] = np.eye(nx)
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vu[-4:, -4:] = np.eye(nu)

            ocp.cost.Vx_e = np.eye(nx)

            # Initial reference trajectory (will be overwritten)
            x_ref = np.zeros(nx)
            ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
            ocp.cost.yref_e = x_ref

            # Initial state (will be overwritten)
            ocp.constraints.x0 = x_ref
            # Set constraints
            ocp.constraints.lbu = quadrotor.min_input_value
            ocp.constraints.ubu = quadrotor.max_input_value
            ocp.constraints.idxbu = np.array([0, 1, 2, 3])

            # Solver options
            ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
            ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
            ocp.solver_options.integrator_type = "ERK"
            ocp.solver_options.print_level = 0
            ocp.solver_options.nlp_solver_type = (
                "SQP_RTI" if solver_options is None else solver_options["solver_type"]
            )

            # Compile acados OCP solver if necessary
            json_file = os.path.join(
                self.acados_models_dir, key_model.name + "_acados_ocp.json"
            )

            self.json_file = json_file
            self.acados_ocp_solver[key] = AcadosOcpSolver(
                ocp, json_file=json_file, verbose=False
            )

    def acados_setup_model(self, nominal, model_name):
        """
        Builds an Acados symbolic models using CasADi expressions.
        :param model_name: name for the acados model. Must be different from previously used names or there may be
        problems loading the right model.
        :param nominal: CasADi symbolic nominal model of the quadrotor: f(self.x, self.u) = x_dot, dimensions 13x1.
        :return: Returns a total of three outputs, where m is the number of GP's in the GP ensemble, or 1 if no GP:
            - A dictionary of m AcadosModel of the GP-augmented quadrotor
            - A dictionary of m CasADi symbolic nominal dynamics equations with GP mean value augmentations (if with GP)
        :rtype: dict, dict, cs.MX
        """

        def fill_in_acados_model(x, u, p, dynamics, name):

            x_dot = cs.MX.sym("x_dot", dynamics.shape)
            f_impl = x_dot - dynamics

            # Dynamics model
            model = AcadosModel()
            model.f_expl_expr = dynamics
            model.f_impl_expr = f_impl
            model.x = x
            model.xdot = x_dot
            model.u = u
            model.p = p
            model.name = name

            return model

        acados_models = {}
        dynamics_equations = {}

        dynamics_equations[0] = nominal
        x_ = self.x
        dynamics_ = nominal

        acados_models[0] = fill_in_acados_model(
            x=x_, u=self.u, p=[], dynamics=dynamics_, name=model_name
        )

        return acados_models, dynamics_equations

    def clear_acados_models(self):
        """
        Removes previous stored acados models to avoid conflicts with new models.
        """
        json_file = self.json_file

        if os.path.exists(json_file):
            os.remove(json_file)
        c_code_dir = self.acados_models_dir + "/" + "c_generated_code"

        if os.path.exists(c_code_dir):
            shutil.rmtree(c_code_dir)

    def set_reference_state(self, x_target=None, u_target=None):
        """
        Sets the target state and pre-computes the integration dynamics with cost equations
        :param x_target: 13-dimensional target state (p_xyz, a_wxyz, v_xyz, r_xyz)
        :param u_target: 4-dimensional target control input vector (u_1, u_2, u_3, u_4)
        """

        if x_target is None:
            x_target = [[0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]]
        if u_target is None:
            u_target = [0, 0, 0, 0]
        # Set new target state
        self.target = copy(x_target)

        ref = np.concatenate([x_target[i] for i in range(4)])
        #  Transform velocity to body frame
        v_b = v_dot_q(ref[7:10], quaternion_inverse(ref[3:7]))
        ref = np.concatenate((ref[:7], v_b, ref[10:]))

        idx = 0
        ref = np.concatenate((ref, u_target))

        for j in range(self.N):
            self.acados_ocp_solver[idx].set(j, "yref", ref)
            self.acados_ocp_solver[idx].set(self.N, "yref", ref[:-4])

        return idx

    def set_reference_trajectory(self, x_target, u_target):
        """
        Sets the reference trajectory and pre-computes the cost equations for each point in the reference sequence.
        :param x_target: Nx13-dimensional reference trajectory (p_xyz, angle_wxyz, v_xyz, rate_xyz). It is passed in the
        form of a 4-length list, where the first element is a Nx3 numpy array referring to the position targets, the
        second is a Nx4 array referring to the quaternion, two more Nx3 arrays for the velocity and body rate targets.
        :param u_target: Nx4-dimensional target control input vector (u1, u2, u3, u4)
        """

        if u_target is not None:
            assert (
                x_target[0].shape[0] == (u_target.shape[0] + 1)
                or x_target[0].shape[0] == u_target.shape[0]
            )

        # If not enough states in target sequence, append last state until required length is met
        while x_target[0].shape[0] < self.N + 1:
            x_target = [
                np.concatenate((x, np.expand_dims(x[-1, :], 0)), 0) for x in x_target
            ]
            if u_target is not None:
                u_target = np.concatenate(
                    (u_target, np.expand_dims(u_target[-1, :], 0)), 0
                )

        stacked_x_target = np.concatenate([x for x in x_target], 1)

        #  Transform velocity to body frame
        x_mean = stacked_x_target[int(self.N / 2)]
        v_b = v_dot_q(x_mean[7:10], quaternion_inverse(x_mean[3:7]))
        x_target_mean = np.concatenate((x_mean[:7], v_b, x_mean[10:]))

        idx = 0

        self.target = copy(x_target)

        for j in range(self.N):
            ref = stacked_x_target[j, :]
            ref = np.concatenate((ref, u_target[j, :]))
            self.acados_ocp_solver[idx].set(j, "yref", ref)
        # the last MPC node has only a state reference but no input reference
        self.acados_ocp_solver[idx].set(self.N, "yref", stacked_x_target[self.N, :])

        return idx

    def quad_dynamics(self):
        """
        Symbolic dynamics of the 3D quadrotor model. The state consists on: [p_xyz, a_wxyz, v_xyz, r_xyz]^T, where p
        stands for position, a for angle (in quaternion form), v for velocity and r for body rate. The input of the
        system is: [u_1, u_2, u_3, u_4], i.e. the activation of the four thrusters.


        :return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        Inputs: 'x' state of quadrotor (6x1) and 'u' control input (2x1). Output: differential state vector 'x_dot'
        (6x1)
        """
        xdot = cs.vertcat(
            self.p_dynamics(),
            self.q_dyanamics(),
            self.v_dynamics(),
            self.r_dynamics(),
        )
        return cs.Function(
            "x_dot", [self.x[:13], self.u], [xdot], ["x", "u"], ["x_dot"]
        )

    def run_optimization(self, initial_state=None, use_model=0, return_x=False):
        """
        Optimizes a trajectory to reach the pre-set target state, starting from the input initial state, that minimizes
        the quadratic cost function and respects the constraints of the system

        :param initial_state: 13-element list of the initial state. If None, 0 state will be used
        :param use_model: integer, select which model to use from the available options.
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.
        :return: optimized control input sequence (flattened)
        """

        if initial_state is None:
            initial_state = [0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0]

        # Set initial state.
        x_init = initial_state
        x_init = np.stack(x_init)

        # Set initial condition, equality constraint
        self.acados_ocp_solver[use_model].set(0, "lbx", x_init)
        self.acados_ocp_solver[use_model].set(0, "ubx", x_init)

        # Solve OCP
        self.acados_ocp_solver[use_model].solve()

        # Get u
        w_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver[use_model].get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver[use_model].get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_ocp_solver[use_model].get(i + 1, "x")

        w_opt_acados = np.reshape(w_opt_acados, (-1))
        return w_opt_acados if not return_x else (w_opt_acados, x_opt_acados)

    def p_dynamics(self):
        return self.v

    def q_dyanamics(self):
        return 1 / 2 * cs.mtimes(skew_symmetric(self.r), self.q)

    def v_dynamics(self):
        # self.u is between [0.0, 1.0] for thrust and [-1.0, 1.0] for torques
        forces = self.u[0] * self.quadrotor.max_thrust
        a_thrust = forces / self.quadrotor.mass
        a_thrust = cs.vertcat(0, 0, a_thrust)
        g = cs.vertcat(0, 0, self.quadrotor.g[2])

        angle_quaternion = self.q

        v_dynamics = g + v_dot_q(a_thrust, angle_quaternion)
        return v_dynamics

    def r_dynamics(self):
        forces = self.u[0] * self.quadrotor.max_thrust
        torques = self.u[1:] * self.quadrotor.max_torque

        rate = self.r
        J = self.quadrotor.J

        r_dynamics = cs.vertcat(
            1 / J[0] * (torques[0] + (J[1] - J[2]) * rate[1] * rate[2]),
            1 / J[1] * (torques[1] + (J[2] - J[0]) * rate[2] * rate[0]),
            1 / J[2] * (torques[2] + (J[0] - J[1]) * rate[0] * rate[1]),
        )

        return r_dynamics
