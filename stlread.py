import open3d as o3d
import numpy as np
import pandas as pd


class KdTree:
    def __init__(self, pcd):
        self.pcd = np.asarray(pcd.points)
        self.radius_sweep = 20
        self.c = 0

    def calc_distance(self, point1, point2):
        return np.linalg.norm(point1-point2)

    def get_remaining_points(self, point_cloud, center_point, radius):
        remaining_points = []
        current_points = list(center_point)
        for point in point_cloud:
            distance = self.calc_distance(center_point, point)
            if distance > self.radius_sweep:
                remaining_points.append(point)
            else:
                current_points.append(point)
        return np.asarray(remaining_points), np.asarray(current_points)

    def search(self):
        selected_points = []
        while len(self.pcd) > 0:
            starting_point = self.pcd[[np.random.randint(0, len(self.pcd))]
                                      ]
            selected_points.append(starting_point)
            self.pcd, current_points = self.get_remaining_points(
                self.pcd, starting_point, self.radius_sweep)
            print(len(current_points))


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
    o3d.visualization.draw_geometries([mesh_coord_frame, mesh])


def mesh2PointCloud(mesh):
    n_pts = 1000
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
