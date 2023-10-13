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
            force_optimization.recursiveLeastSquares()
            break
            # visualize(new_pcd)


class Optimization:
    def __init__(self, pcd):
        self.pcd = pcd
        self.max_force = 10
        self.f_ext = np.asarray([0, 0, -10, 0, 0, 0])

    def choose(self):
        unique_combinations = np.asarray(
            list(combinations(zip(self.pcd.points, self.pcd.normals), 3)))
        return unique_combinations

    def recursiveLeastSquares(self):
        unique_combinations = self.choose()
        for i, combination in enumerate(unique_combinations, start=1):
            for point, normal in combination:
                normal = -normal
                pass


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
