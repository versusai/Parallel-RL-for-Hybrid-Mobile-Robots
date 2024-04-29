import json
import os
from scipy.linalg import norm
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def obstacle_2(x, y, z, roll, pitch, yaw, ax):
    # Define the rectangle parameters
    width = 1.0
    height = 5.0
    depth = 0.5

    rotation_angles = [
        roll,
        pitch,
        yaw,
    ]  # Rotation angles around X, Y, and Z axes in degrees

    # Define the vertices of the rectangle
    vertices = np.array(
        [
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0],
            [0, 0, depth],
            [width, 0, depth],
            [width, height, depth],
            [0, height, depth],
        ]
    )

    # Apply translation
    translated_vertices = vertices + np.array([x, y, z])

    # Apply rotation
    rotation_matrix = np.eye(3)
    for angle, axis in zip(rotation_angles, ["x", "y", "z"]):
        # rotation = np.deg2rad(angle)
        rotation = angle
        if axis == "x":
            rotation_matrix = np.dot(
                rotation_matrix,
                np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(rotation), -np.sin(rotation)],
                        [0, np.sin(rotation), np.cos(rotation)],
                    ]
                ),
            )
        elif axis == "y":
            rotation_matrix = np.dot(
                rotation_matrix,
                np.array(
                    [
                        [np.cos(rotation), 0, np.sin(rotation)],
                        [0, 1, 0],
                        [-np.sin(rotation), 0, np.cos(rotation)],
                    ]
                ),
            )
        elif axis == "z":
            rotation_matrix = np.dot(
                rotation_matrix,
                np.array(
                    [
                        [np.cos(rotation), -np.sin(rotation), 0],
                        [np.sin(rotation), np.cos(rotation), 0],
                        [0, 0, 1],
                    ]
                ),
            )

    rotated_vertices = np.dot(translated_vertices, rotation_matrix)

    # Define the faces of the rectangle
    faces = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [4, 5, 6, 7],
        ]
    )

    # Create a Poly3DCollection object and add it to the plot
    rectangle = Poly3DCollection(rotated_vertices[faces], alpha=0.5)
    rectangle.set_facecolor("k")
    ax.add_collection3d(rectangle)


def plot_linear_cube(x, y, z, dx, dy, dz, ax, color="k"):
    xx = [x, x, x + dx, x + dx, x]
    yy = [y, y + dy, y + dy, y, y]
    kwargs = {"alpha": 0.5, "color": color}
    ax.plot3D(xx, yy, [z] * 5, **kwargs)
    ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
    ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)


def draw_cylinder(
    radius: float = 1.0,
    height: float = 2.0,
    translation: np.array = np.array([0, 0, 0]),
    rotation_matrix: np.array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
):
    # Generate the cylinder surface points
    theta = np.linspace(0, 2 * np.pi, 100)
    u = np.linspace(0, 1, 10)

    # Create a meshgrid for theta and u
    theta, u = np.meshgrid(theta, u)

    # Compute the x, y, z coordinates of the cylinder points
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = height * u

    # Apply translation
    x += translation[0]
    y += translation[1]
    z += translation[2]

    xyz = np.stack((x, y, z), axis=0)  # Shape: (3, 10, 100)
    xyz_rotated = np.matmul(
        rotation_matrix, xyz.reshape(3, -1)
    )  # Reshape and apply rotation

    # Extract the rotated coordinates
    x_rotated, y_rotated, z_rotated = xyz_rotated.reshape(3, 10, 100)

    return x_rotated, y_rotated, z_rotated


path = os.path.dirname(os.path.abspath(__file__))
list_dir = os.listdir(path + "/results_1/")

# stage = [
#    mpimg.imread(path + '/media/stage_{}.png'.format(i)) for i in range(1, 4)
# ]
color = {
    "SAC": "dodgerblue",
    "SAC-P": "springgreen",
    "PDSAC": "indigo",
    "PDSAC-P": "deeppink",
}
sel = {"S1": 0, "S2": 1, "S3": 2}

splitted_dir = list()
for dir in list_dir:
    if dir != "data" and dir.split("_")[0] != "BUG2":
        splitted_dir.append(dir.split("_"))
sorted_dir = sorted(
    splitted_dir, key=lambda row: row[1] if row[0] == "BUG2" else row[3]
)
print("Dir:", sorted_dir)

sorted_dir = np.array(sorted_dir)
sorted_dir = sorted_dir[4:8]
for c, directory in tqdm(enumerate(sorted_dir), total=len(sorted_dir)):
    with open(path + "/results_1/" + "_".join(directory) + "/writer_data.json") as f:
        data = json.load(f)

    key_list = list(data.keys())
    new_key_list = ["/".join(key.split("/")[-2:]) for key in key_list]

    for i, key in enumerate(key_list):
        data[new_key_list[i]] = data.pop(key)

    df = pd.DataFrame(data, dtype=np.float32)
    reward = df.iloc[:, df.columns == new_key_list[0]].to_numpy()
    new_reward = list()
    for i, t in enumerate(reward):
        new_reward.append(t[0][-1])

    timing = df.iloc[:, df.columns == new_key_list[1]].to_numpy()
    new_timing = list()
    for i, t in enumerate(timing):
        new_timing.append(t[0][-1])

    episode = df.iloc[:, df.columns == new_key_list[2]].to_numpy()
    new_episode = list()
    for i, t in enumerate(episode):
        new_episode.append(t[0][-1])

    df = pd.DataFrame(
        {
            new_key_list[0]: list(new_reward),
            new_key_list[1]: list(new_timing),
            new_key_list[2]: list(new_episode),
        },
        dtype=np.float32,
    )
    df = df.sort_values(
        [new_key_list[2], new_key_list[0], new_key_list[1]],
        ascending=[True, False, False],
    )
    df = df.groupby(new_key_list[2]).first().reset_index()[1:]

    if directory[0] != "BUG2":
        a = "PDSAC" if directory[0] == "PDSRL" else "SAC"
        if directory[-1] != "N":
            name = "-".join([a, directory[-1]])
        else:
            name = a
        c = directory[-2]
    else:
        name = a
        c = directory[1]

    # name = f"Delay update: {directory[-1]}"
    print(name)
    #    name = 'D4PG' if name=='PDDRL' else 'DSAC'
    sucess_list = list()
    for value in df[new_key_list[0]]:
        if value == 200:
            sucess_list.append(1)
        else:
            sucess_list.append(0)

    sucess_rate = (sum(sucess_list) / len(sucess_list)) * 100
    print("Data for", name + directory[3], "test simulations:")
    print("Sucess rate:", sucess_rate, "%")
    print("Episode reward mean:", df[new_key_list[0]].mean())
    print("Episode reward std:", df[new_key_list[0]].std())
    print("Episode timing mean:", df[new_key_list[1]].mean())
    print("Episode timing std:", df[new_key_list[1]].std())

    x = pd.DataFrame(data["agent_0/x"]).iloc[:, 2].to_numpy().tolist()
    y = pd.DataFrame(data["agent_0/y"]).iloc[:, 2].to_numpy().tolist()
    z = pd.DataFrame(data["agent_0/z"]).iloc[:, 2].to_numpy().tolist()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # origin = np.array([0, 0, 0])
    # plt.imshow(stage[sel[c]], extent=[-10, 10, -10, 10])
    #'''
    new_x = list()
    new_y = list()
    new_z = list()
    last = 0
    for i in range(len(x) - 1):
        if abs(x[i]) > 0.5 >= abs(x[i + 1]):
            new_x.append(x[last + 1 : i - 1])
            new_y.append(y[last + 1 : i - 1])
            new_z.append(z[last + 1 : i - 1])
            last = i + 1

    for x, y, z in zip(new_x, new_y, new_z):
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        ax.plot(x, y, z, color="blue")
    # ax.plot(x, y, z, color="blue", label='Trajectory Performed')

    plt.plot([0.0], [0.0], [2.5], marker="o", markersize=20, color="green")
    # plt.plot([3.0], [0.0], [2.5], marker='o', markersize=5, color="green", label='Start')
    if directory[3] == "S1":
        plt.plot([-3.0], [-3.0], [1.5], marker="o", markersize=20, color="red")
        # plt.plot([-3.0], [-3.0], [1.5], marker='o', markersize=5, color="red", label='Goal')
        plt.plot([-3.0], [3.0], [1.5], marker="o", markersize=20, color="red")
        plt.plot([3.0], [-3.0], [1.5], marker="o", markersize=20, color="red")
        plt.plot([3.0], [3.0], [1.5], marker="o", markersize=20, color="red")
    elif directory[3] == "S2":
        x1, y1, z1 = draw_cylinder(radius=0.5, translation=np.array([2.0, 2.0, 0.0]))
        ax.plot_surface(x1, y1, z1, color="k", alpha=0.3)
        x2, y2, z2 = draw_cylinder(radius=0.5, translation=np.array([-2.0, 2.0, 0.0]))
        ax.plot_surface(x2, y2, z2, color="k", alpha=0.3)
        x3, y3, z3 = draw_cylinder(radius=0.5, translation=np.array([2.0, -2.0, 0.0]))
        ax.plot_surface(x3, y3, z3, color="k", alpha=0.3)
        x4, y4, z4 = draw_cylinder(radius=0.5, translation=np.array([-2.0, -2.0, 0.0]))
        ax.plot_surface(x4, y4, z4, color="k", alpha=0.3)
        plt.plot([-3.75], [-3.75], [2.5], marker="o", markersize=20, color="red")
        # plt.plot([-3.75], [-3.75], [2.5], marker='o', markersize=5, color="red", label='Goal')
        plt.plot([-3.75], [3.75], [2.5], marker="o", markersize=20, color="red")
        plt.plot([3.75], [-3.75], [2.5], marker="o", markersize=20, color="red")
        plt.plot([3.75], [3.75], [2.5], marker="o", markersize=20, color="red")
    elif directory[3] == "S3":
        plt.plot([-3.5], [0.0], [1.5], marker="o", markersize=20, color="red")
        plt.plot(
            [-3.5], [0.0], [1.5], marker="o", markersize=5, color="red", label="Goal"
        )
        plt.plot([0.0], [3.5], [1.5], marker="o", markersize=20, color="red")
        plt.plot([0.0], [-3.5], [1.5], marker="o", markersize=20, color="red")
        plt.plot([3.5], [3.0], [1.5], marker="o", markersize=20, color="red")
        obstacle_2(-2.0, -4.4, 2.5, 0.0, 0, -1.57, ax)  # 1
        obstacle_2(-2.0, -3.4, 2.5, 0.0, 0, -1.57, ax)  # 2
        obstacle_2(4.4, -1.5, 2.5, 0.0, 0, 0, ax)  # 3
        obstacle_2(3.4, -1.5, 2.5, 0.0, 0, 0, ax)  # 4
        obstacle_2(0.0, -1.5, 2.5, 0.0, 0, 0, ax)  # 5
        obstacle_2(2.0, 2.0, 2.5, 0.0, 0, -0.75, ax)  # 6
        obstacle_2(0.95, 2.5, 2.5, 0.0, 0, 0.0, ax)  # 7
        obstacle_2(-3.5, 3.0, 2.5, 0.0, 0, -0.75, ax)  # obs3 8
        obstacle_2(-2.0, 2.8, 2.1, 0.9, 0.3, -0.1, ax)  # obs4 9
        obstacle_2(-2.0, -2.35, 2.2, -1.55, 1.5, 0, ax)  # 10
        obstacle_2(2.25, 2.4, 3.2, 0, 1.55, 1.55, ax)  # 11
        obstacle_2(-0.11, -2.54, 0.35, 1.53, 0.0, 0.0, ax)  # obs4 12
        obstacle_2(-0.11, -2.54, 4.15, 1.53, 0.0, 0.0, ax)  # obs4 13
        obstacle_2(2.25, 2.4, 4.2, 0, 1.55, 1.55, ax)  # 14
        obstacle_2(0.0, 2.5, 2.4, -3.1, 1.55, 0.0, ax)  # 15

    # plt.title('Trajectory ' + name, size=20)
    plt.xlabel("Metros")
    plt.ylabel("Metros")
    ax.set_zlabel("Metros")
    # plt.legend()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(0, 4)
    plot_linear_cube(-4.0, -4.0, 0.0, 8.0, 8.0, 4.0, ax)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # plt.show()
    plt.savefig(
        "{}.pdf".format(directory[3] + "_" + name),
        format="pdf",
        bbox_inches=extent.expanded(1.0, 1.1),
    )
    # plt.show()
