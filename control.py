import numpy as np


# inital_conditions
inertia_matrix = np.array([[0.0820, 0.0845, 0.1377]])
m = 4.34
d = 0.315 # this is the distance between the rotor and COM
thrust_to_torque_rate = 8.004 **-4
g = 9.81
# controller parameters (assigning the weights)
k_x = 16 # for adjusting the translation
k_v = 5.6 # for adjusting the velocity of the COM
k_R = 8.81 # roation (attitude) error
k_omega = 2.54 # angular velocity error weight

# this is how we gonna compute the error
def desired_trajectory(t):
    x_d = np.array([[0.4 * t, 0.4 * np.sin(np.pi * t), 0.6 * np.cos(np.pi * t)]])
    b_d = np.array([[np.cos(np.pi * t), np.sin(np.pi * t), 0]])
    return x_d, b_d

def derivative_desired_traj(t):
    der_x_d = np.array([[0.4, 0.4 * np.pi * np.cos(np.pi * t), -0.6 * np.pi * np.sin(np.pi * t)]])
    der_b_d = np.array([[-np.pi * np.sin(np.pi * t), np.pi * np.cos(np.pi * t), 0]])
    return der_x_d, der_b_d

def double_der_desired_traj(t):
    acc_x = np.array([[0, -0.4 * (np.pi**2) * np.sin(np.pi * t), -0.6 * (np.pi**2) * np.cos(np.pi * t)]])
    acc_b = np.array([[-(np.pi**2) * np.cos(np.pi * t), -(np.pi**2) * np.sin(np.pi * t), 0]])
    return acc_x, acc_b

e_3 = np.array([0, 0, 1])

def vee(matrix):
    """
    Converts a skew-symmetric matrix to a 3D vector.
    :param matrix: A 3x3 skew-symmetric matrix
    :return: A 3D vector as a NumPy array
    """
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])

def controller(t, x, v, R, omega):
    """
    calculation for torque and thrust needed
    """
    
    ### desired thrust
    x_d, b1_d = desired_trajectory(t)
    x_v_d, _ = derivative_desired_traj(t)
    x_a_d, _ = double_der_desired_traj(t)
    error_x = x - x_d
    error_v = v - x_v_d
    f_des = -(-k_x * error_x - k_v * error_v - m * g * e_3 + m * x_a_d)
    
    ### desired attitude
    # need the desired attitude
    b3_d = f_des
    f_norm = np.linalg.norm(b3_d)
    b3_d = b3_d / f_norm
    
    b2_d = np.cross(b3_d, b1_d)
    norm = np.linalg.norm(b2_d)
    b2_d = b2_d / norm
    
    R_d = np.array([[np.cross(b2_d, b3_d), b2_d, b3_d]])
    
    #calculate error
    attitude_error = 0.5 * vee(R_d.T @ R - R @ R_d.T)
    omega_desired = np.array([0, 0, 0]) # we want 0 torque, so the quad wont twirl like a pony go around
    omega_error = omega - R.T @ R_d @ omega_desired
    
    m = -k_R * attitude_error - k_omega * omega_error + np.cross(omega, inertia_matrix @ omega)
    
    f = f_norm * (R[:, 2] @ np.array([0, 0, 1]))
    
    return f, m

time = np.linspace(0, 0.1, 10) # this is time
# treat as a step size
x_0 = np.array([[0, 0, 0]])
v_0 = np.array([[0, 0, 0]])
attitude_0 = np.eye(3)
angular_vel_0 = np.array([[0, 0, 0]])

def skew_symmetric(vec):
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

for t in time:
    x = x_0
    v = v_0
    attitude = attitude_0
    angular_vel = angular_vel_0
    F, M = controller(t=t, x=x, v=v, R=attitude, omega=angular_vel)
    
    # find a, 
    a  = F / m
    v += a * t
    x += v * t
    
    angular_acc = np.linalg.inv(inertia_matrix) @ (M - np.cross(angular_vel, inertia_matrix @ angular_vel))
    angular_vel += angular_acc * t
    
    R_dot = attitude @ skew_symmetric(angular_vel.flatten())
    attitude = attitude + t * R_dot
    
    print(f"Time: {t:.2f}")
    print(f"Position: {x}")
    print(f"Velocity: {v}")
    print(f"Attitude:\n{attitude}")
    print(f"Angular Velocity: {angular_vel}\n")