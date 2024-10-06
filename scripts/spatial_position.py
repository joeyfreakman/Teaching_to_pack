import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from src.aloha.aloha_scripts.visualize_episodes import load_hdf5
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import tikzplotlib

def skew_symmetric(v):
    """
    Create a skew-symmetric matrix from a 3D vector.
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def screw_matrix(S):
    """
    Convert a screw vector to its matrix representation.
    """
    omega = S[:3]
    v = S[3:]
    return np.block([
        [skew_symmetric(omega), v.reshape(3, 1)],
        [np.zeros((1, 3)), 0]
    ])

def forward_kinematics(M, Slist, theta):
    """
    Calculate the forward kinematics.
    """
    T = M
    for i in range(len(theta)):
        T = T @ expm(screw_matrix(Slist[:, i]) * theta[i])
    return T

# Given matrices
M = np.array([
    [1.0, 0.0, 0.0, 0.458325],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.36065],
    [0.0, 0.0, 0.0, 1.0]
])
# M1 = np.array([
#     [1.0, 0.0, 0.0, 0.408575],
#     [0.0, 1.0, 0.0, 0.0],
#     [0.0, 0.0, 1.0, 0.31065],
#     [0.0, 0.0, 0.0, 1.0]
# ])

Slist = np.array([
    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -0.11065, -0.36065, 0.0, -0.36065, 0.0],
    [0.0, 0.0, 0.0, 0.36065, 0.0, 0.36065],
    [0.0, 0.0, 0.04975, 0.0, 0.29975, 0.0]
]) # Transpose to match the expected shape
# Slist1 = np.array([
#     [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
#     [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
#     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, -0.11065, -0.31065, 0.0, -0.42705, 0.0],
#     [0.0, 0.0, 0.0, 0.31065, 0.0, 0.42705],
#     [0.0, 0.0, 0.05, 0.0, 0.35955, 0.0]
# ])

def calculate_spatial_coordinates(qpos):
    """
    Calculate spatial coordinates for both left and right arms.
    """
    left_arm_qpos = qpos[:6]
    right_arm_qpos = qpos[7:13]
    
    # Calculate forward kinematics for left arm
    T_left = forward_kinematics(M, Slist, left_arm_qpos)
    left_coordinates = T_left[:3, 3]
    
    # Calculate forward kinematics for right arm
    T_right = forward_kinematics(M, Slist, right_arm_qpos)
    right_coordinates = T_right[:3, 3]
    
    return left_coordinates, right_coordinates

def calculate_end_effector_position(qpos):
    left_arm_qpos = qpos[:6]
    right_arm_qpos = qpos[7:13]
    
    T_left = forward_kinematics(M, Slist, left_arm_qpos)
    T_right = forward_kinematics(M, Slist, right_arm_qpos)
    
    return T_left[:3, 3], T_right[:3, 3]

def world_to_pixel(world_coord, reference_world, reference_pixel, scale_factor):
    """
    Convert world coordinates to pixel coordinates.
    
    :param world_coord: (x, y) world coordinate to convert
    :param reference_world: (x, y) world coordinate of the reference point
    :param reference_pixel: (x, y) pixel coordinate of the reference point
    :param scale_factor: (x_scale, y_scale) scaling factors for x and y axes
    :return: (x, y) pixel coordinate
    """
    dx = (world_coord[0] - reference_world[0]) * scale_factor[0]
    dy = (world_coord[1] - reference_world[1]) * scale_factor[1]
    
    # Note the negation and swap of dx, dy due to the coordinate system difference
    pixel_x = reference_pixel[0] - dx
    pixel_y = reference_pixel[1] + dy
    
    return (int(pixel_x), int(pixel_y))

def visualize_trajectory(position_data):
    """
    Visualize the end-effector trajectory given position data with color gradient.
    
    Args:
    position_data (np.array): Array of shape (t, 14) containing position data for t timesteps.
    """
    t, _ = position_data.shape
    left_positions = []
    right_positions = []
    
    for i in range(t):
        left_pos, right_pos = calculate_end_effector_position(position_data[i])
        left_positions.append(left_pos)
        right_positions.append(right_pos)
        # if i <= 9:
        #     print(left_pos, right_pos)
    left_positions = np.array(left_positions)
    right_positions = np.array(right_positions)
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create custom colormap
    colors = ['yellow', 'red']
    n_bins = 100  # Number of color bins
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Plot left arm trajectory
    for i in range(t-1):
        ax.plot(left_positions[i:i+2, 0], left_positions[i:i+2, 1], left_positions[i:i+2, 2], 
                color=cmap(i/t), linewidth=2)
    
    # Plot right arm trajectory
    for i in range(t-1):
        ax.plot(right_positions[i:i+2, 0], right_positions[i:i+2, 1], right_positions[i:i+2, 2], 
                color=cmap(i/t), linewidth=2)
    
    # Add start and end markers
    ax.scatter(left_positions[0, 0], left_positions[0, 1], left_positions[0, 2], 
               color='black', s=100, label='Start (Left)')
    ax.scatter(left_positions[-1, 0], left_positions[-1, 1], left_positions[-1, 2], 
               color='red', s=100, label='End (Left)')
    ax.scatter(right_positions[0, 0], right_positions[0, 1], right_positions[0, 2], 
               color='purple', s=100, label='Start (Right)')
    ax.scatter(right_positions[-1, 0], right_positions[-1, 1], right_positions[-1, 2], 
               color='brown', s=100, label='End (Right)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot End-Effector Trajectory with Time Gradient')
    # ax.legend()
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=t))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Time Steps', pad=0.1)
    plt.show()
    # plt.savefig("trajectory.tex", figure=fig, axis_width='\\figwidth', axis_height='\\figheight', strict=True)
    
# Example usage
if __name__ == "__main__":
    dataset_name = 'episode_0'
    dataset_dir = '/mnt/d/kit/ALR/dataset/ttp_full'
    qpos, _, action, _ = load_hdf5(dataset_dir, dataset_name)
    visualize_trajectory(action)


