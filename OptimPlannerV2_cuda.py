import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import cupy as cp
from scipy.optimize import minimize

class logger():
    def __init__(self):
        self.candidate1 = []
        self.candidate2 = []
        self.candidate3 = []
        self.err = []
        self.handle_force = []
        self.sec_force = []
        self.ter_force = []
        self.temp = 10
        self.min = []

    def log(self, sol, id, err, pcd):
        ns = cp.array([pcd.normals[id[0]], pcd.normals[id[1]], pcd.normals[id[2]])
        rank = cp.linalg.matrix_rank(ns)

        if err < self.temp:
            self.temp = err
            self.min = id
        self.candidate1.append(id[0])
        self.candidate2.append(id[1])
        self.candidate3.append(id[2])
        self.handle_force.append([sol[2]])
        self.sec_force.append([sol[5])
        self.ter_force.append([sol[8]])
        self.err.append(err)

    def save_file(self):
        df = pd.DataFrame({"Pt1": self.candidate1, "Pt2": self.candidate2, "Pt3": self.candidate3, "F1": self.handle_force,
                           "F2": self.sec_force, "F3": self.ter_force, "Error": self.err})
        df.to_csv("realtime_diagnostics.csv", index=False)
        return self.min

    def cost_visualizer(self):
        k = []
        for i in range(len(self.err)):
            k.append(i)
        plt.plot(k, self.err, label='Cost comparison')
        plt.xlabel('Candidate Number')
        plt.ylabel('Final cost')
        plt.title('cost analysis')
        plt.show()

class KdTree:
    def __init__(self, pcd):
        self.pcd = pcd
        self.kd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.radius_sweep = 80
        self.selected_points = []
        self.checked_points = []

    def get_points(self, center_point):
        [k, idx, _] = self.kd_tree.search_radius_vector_3d(
            center_point, self.radius_sweep)
        return idx

    def search(self):
        for k in range(len(self.pcd.points)):
            min = 12
            handle_id = k
            query_point = self.pcd.points[handle_id]
            idx = self.get_points(query_point)
            idx = list(idx)
            idx.remove(handle_id)
            force_optimization = Optimization(handle_id, idx, self.pcd)
            force_optimization.transformation()

class Optimization:
    def __init__(self, handle_id, idx, pcd):
        self.handle_id = handle_id
        self.idx = idx
        self.pcd = pcd
        self.max_force = 10
        self.f_ext = cp.array([0, 0, 10, 0, 0, 0])
        # For point contact with friction
        self.Bci = cp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.G = None
        self.mew = 0.7
        self.fc = cp.array([0, 0, 2, 0, 0, 2, 0, 0, 2])
        self.min = 15  # Taking 15 since max error will be 10
        self.solved_combination = None
        self.solution = None
        self.length = self.boundingbox_len()

    def choose(self):
        unique_combinations = cp.asarray(list(combinations(self.idx, 2))
        return unique_combinations

    def transformation(self):
        unique_combinations = list(self.choose())
        new_combinations = [[item[0], item[1], self.handle_id] for item in unique_combinations]

        for i in range(len(new_combinations)):
            self.G = None
            self.idt = []
            for j in range(3):
                id = new_combinations[i][j]
                self.idt.append(id)
                normal = self.pcd.normals[id]
                point = self.pcd.points[id]
                # This gives us orientation of normal vector with x, y, and z axis
                normal = -normal
                x_axis_angle = cp.arctan2(cp.linalg.norm(cp.cross(
                    normal, cp.array([1, 0, 0])), cp.dot(normal, cp.array([1, 0, 0])))
                y_axis_angle = cp.arctan2(cp.linalg.norm(cp.cross(
                    normal, cp.array([0, 1, 0])), cp.dot(normal, cp.array([0, 1, 0])))
                z_axis_angle = cp.arctan2(cp.linalg.norm(cp.cross(
                    normal, cp.array([0, 0, 1])), cp.dot(normal, cp.array([0, 0, 1])))
                R_alpha = cp.array([[cp.cos(z_axis_angle), -cp.sin(z_axis_angle), 0],
                                    [cp.sin(z_axis_angle), cp.cos(z_axis_angle), 0],
                                    [0, 0, 1]])
                R_beta = cp.array([[cp.cos(y_axis_angle), 0, cp.sin(y_axis_angle)],
                                   [0, 1, 0],
                                   [-cp.sin(y_axis_angle), 0, cp.cos(y_axis_angle)])
                R_gamma = cp.array([[1, 0, 0],
                                    [0, cp.cos(x_axis_angle), -cp.sin(x_axis_angle)],
                                    [0, cp.sin(x_axis_angle), cp.cos(x_axis_angle)])
                R = cp.dot(R_alpha, cp.dot(R_beta, R_gamma))
                H = cp.matrix(
                    [[0, -point[2], point[1]], [point[2], 0, -point[0]], [-point[1], point[0], 0]])
                cross_G = cp.dot(H, R)
                zeros_matrix = cp.zeros((3, 3))
                G_i = cp.vstack((cp.hstack((R, zeros_matrix)),
                                cp.hstack((cp.cross(cross_G, R), R)))
                F_oi = cp.dot(G_i, self.Bci)
                if self.G is None:
                    self.G = F_oi
                else:
                    self.G = cp.hstack((self.G, F_oi))
            self.solve()

    def objective_function(self, fc):
        return cp.linalg.norm(cp.dot(self.G, fc) + self.f_ext)

    def constraint_4(self, fc):
        return self.mew * fc[2] - cp.sqrt(fc[0]**2 + fc[1]**2)

    def constraint_5(self, fc):
        return self.mew * fc[5] - cp.sqrt(fc[3]**2 + fc[4]**2)

    def constraint_6(self, fc):
        return self.mew * fc[8] - cp.sqrt(fc[6]**2 + fc[7]**2)

    def centroid_sep(self):
        c = cp.array([])
        center_point = cp.mean(cp.asarray(self.pcd.points), axis=0)
        point1 = self.pcd.points[self.idt[0]]
        point2 = self.pcd.points[self.idt[0]]
        point3 = self.pcd.points[self.idt[0]]
        centroid = (point1 + point2 + point3) / 3.0
        distance = cp.linalg.norm(centroid - center_point)
        return distance

    def boundingbox_len(self):
        min_bound = cp.min(cp.asarray(self.pcd.points), axis=0)
        max_bound = cp.max(cp.asarray(self.pcd.points), axis=0)
        length = max_bound[0] - min_bound[0]
        width = max_bound[1] - min_bound[1]
        height = max_bound[2] - min_bound[2]
        k = cp.max([length, width, height])
        return k

    def solve(self):
        con4 = {'type': 'ineq', 'fun': self.constraint_4}
        con5 = {'type': 'ineq', 'fun': self.constraint_5}
        con6 = {'type': 'ineq', 'fun': self.constraint_6}
        b = (0, 10)
        bnds = [b, b, b, b, b, b, b, b, b]
        cons = [con4, con5, con6]
        sol = minimize(self.objective_function, self.fc,
                       method='SLSQP', bounds=bnds, constraints=cons)
        err = self.objective_function(sol.x)
        solution = list(sol.x)
        if self.objective_function(sol.x) < 10 and self.centroid_sep() < self.length / 7:
            log1.log(solution, self.idt, err, self.pcd)

def mesh2PointCloud(mesh):
    n_pts = 50
    pcd = mesh.sample_points_uniformly(n_pts)
    return pcd

def visualize(mesh):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(mesh)
    visualizer.run()
    visualizer.destroy_window()

def main():
    mesh_path = "mesh.stl"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh2PointCloud(mesh)
    obj = KdTree(pcd)
    pcd_df = pd.DataFrame(cp.concatenate((cp.asarray(pcd.points), cp.asarray(pcd.normals)), axis=1),
                          columns=["x", "y", "z", "norm-x", "norm-y", "norm-z"])
    reqd_combination = obj.search()
    log1.save_file()
    log1.cost_visualizer()
    print("Optimum force at fingers is ", pcd.normals[log1.min[0]], pcd.normals[log1.min[1]], pcd.normals[log1.min[2]])
    ns = cp.asarray([pcd.normals[log1.min[0]], pcd.normals[log1.min[1]], pcd.normals[log1.min[2]])
    pts = cp.asarray([pcd.points[log1.min[0]], pcd.points[log1.min[1]], pcd.points[log1.min[2]])
    force_visualizer(mesh, pts, ns)
    visualize(mesh)

log1 = logger()

if __name__ == "__main__":
    main()
