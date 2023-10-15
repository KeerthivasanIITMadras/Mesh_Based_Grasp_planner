import open3d as o3d
import numpy as np
import pandas as pd
from itertools import combinations


class KdTree:
    def __init__(self, pcd):
        self.pcd = pcd
        self.kd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.radius_sweep = 20
        self.selected_points = []
        self.checked_points = []

    def get_points(self, center_point):

        [k, idx, _] = self.kd_tree.search_radius_vector_3d(
            center_point, self.radius_sweep)
        return idx

    def search(self):
        while len(self.checked_points) <= len(self.pcd.points):
            sample_point = np.random.randint(0, len(self.pcd.points))
            if sample_point in self.checked_points:
                continue
            idx = self.get_points(self.pcd.points[sample_point])
            self.checked_points.append(sample_point)
            for i in idx:
                self.checked_points.append(i)
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(
                np.array([np.asarray(self.pcd.points[i]) for i in idx]))
            new_pcd.normals = o3d.utility.Vector3dVector(
                np.array([np.asarray(self.pcd.normals[i]) for i in idx]))
            force_optimization = Optimization(new_pcd)
            force_optimization.transformation()
            break
            # visualize(new_pcd)


class Optimization:
    def __init__(self, pcd):
        self.pcd = pcd
        self.max_force = 10
        self.f_ext = np.asarray([0, 0, -10, 0, 0, 0])
        # For point contact with friction
        self.Bci = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [
            0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def choose(self):
        unique_combinations = np.asarray(
            list(combinations(zip(self.pcd.points, self.pcd.normals), 3)))
        return unique_combinations

    def transformation(self):
        unique_combinations = self.choose()
        for i, combination in enumerate(unique_combinations, start=1):
            print("New combination")
            G = None
            for point, normal in combination:
                # This gives us orientation of normal vector with x,y and z axis
                x_axis_angle = np.arctan2(np.linalg.norm(np.cross(
                    normal, np.asarray([1, 0, 0]))), np.dot(normal, np.asarray([1, 0, 0])))
                y_axis_angle = np.arctan2(np.linalg.norm(np.cross(
                    normal, np.asarray([1, 0, 0]))), np.dot(normal, np.asarray([0, 1, 0])))
                z_axis_angle = np.arctan2(np.linalg.norm(np.cross(
                    normal, np.asarray([1, 0, 0]))), np.dot(normal, np.asarray([0, 0, 1])))

                R_alpha = np.array([[np.cos(z_axis_angle), -np.sin(z_axis_angle), 0],
                                    [np.sin(z_axis_angle), np.cos(
                                        z_axis_angle), 0],
                                    [0, 0, 1]])

                R_beta = np.array([[np.cos(y_axis_angle), 0, np.sin(y_axis_angle)],
                                   [0, 1, 0],
                                   [-np.sin(y_axis_angle), 0, np.cos(y_axis_angle)]])

                R_gamma = np.array([[1, 0, 0],
                                    [0, np.cos(x_axis_angle), -
                                     np.sin(x_axis_angle)],
                                    [0, np.sin(x_axis_angle), np.cos(x_axis_angle)]])
                R = np.dot(R_alpha, np.dot(R_beta, R_gamma))

                H = np.matrix(
                    [[0, -point[2], point[1]], [point[2], 0, -point[0]], [-point[1], point[0], 0]])

                cross_G = np.dot(H, R)
                zeros_matrix = np.zeros((3, 3))
                G_i = np.vstack((np.hstack((R, zeros_matrix)),
                                np.hstack((np.cross(cross_G, R), R))))
                F_oi = np.dot(G_i, self.Bci)
                if G is None:
                    G = F_oi
                else:
                    G = np.hstack((G, F_oi))
            print(f"{G.shape}")


def visualize(mesh):
    # Creating a mesh of the XYZ axes Cartesian coordinates frame.
    # This mesh will show the directions in which the X, Y & Z-axes point,
    # and can be overlaid on the 3D mesh to visualize its orientation in
    # the Euclidean space.
    # X-axis : Red arrow
    # Y-axis : Green arrow
    # Z-axis : Blue arrow
    points = np.asarray(mesh.points)
    x_centroid = np.mean(points[:, 0])
    y_centroid = np.mean(points[:, 1])
    z_centroid = np.mean(points[:, 2])
    mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5, origin=[x_centroid, y_centroid, z_centroid])
    o3d.visualization.draw_geometries(
        [mesh_coord_frame, mesh], point_show_normal=True)


def mesh2PointCloud(mesh):
    n_pts = 100
    pcd = mesh.sample_points_uniformly(n_pts)
    return pcd


def main():
    mesh_path = "cuboid.stl"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh2PointCloud(mesh)
    pcd_df = pd.DataFrame(np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals)), axis=1),
                          columns=["x", "y", "z", "norm-x", "norm-y", "norm-z"]
                          )
    # pcd_df.to_excel('object.xlsx', sheet_name='Sheet_name_1')
    # To visualize normal,press n
    obj = KdTree(pcd)
    obj.search()
    # visualize(pcd)


if __name__ == "__main__":
    main()
