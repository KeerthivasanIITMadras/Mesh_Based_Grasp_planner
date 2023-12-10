import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
import time
import Isochoose


class logger():
    def __init__(self,):
      
        self.candidate1 = []
        self.candidate2 = []
        self.candidate3 = []
        self.err = []
        self.handle_force = []
        self.sec_force = []
        self.ter_force = []
        self.temp = 70
        self.min = []
        self.eefa = []
        self.min_eef = []
    
    

    def log(self,sol,id,err,pcd,eefs):

        ns = np.asarray([pcd.normals[id[0]],pcd.normals[id[1]],pcd.normals[id[2]]])

        if err < self.temp:
            self.temp = err
            self.min = id
            self.min_eef= eefs
        self.candidate1.append(id[0])
        self.candidate2.append(id[1])
        self.candidate3.append(id[2])
        self.handle_force.append([sol[2]])
        self.sec_force.append([sol[5]]) 
        self.ter_force.append([sol[8]])
        self.err.append(err)
        self.eefa.append(eefs)
        
          

    def save_file(self,filename):
       df = pd.DataFrame({"Pt1" : self.candidate1, "Pt2" : self.candidate2,"Pt3" : self.candidate3, "F1" : self.handle_force, "F2": self.sec_force, "F3": self.ter_force, "Error" :self.err, "EEF":self.eefa})
       df.to_csv(f"{filename}.csv", index = False)
       print(self.min_eef)
       return [self.min, self.min_eef]
       

    def cost_visualizer(self):
        k = []
        for i in range(len(self.err)):
            k.append(i)
        plt.plot( k ,self.err,label=' Cost comparison')
        plt.xlabel('Candidate Number')
        plt.ylabel('Final cost')
        plt.title('cost analysis')
        plt.show()
    

class Optimization():

    def __init__(self, pcd):
        self.isochoose = Isochoose.Isotrichoose()
        self.pcd = pcd
        self.max_force = 70
        self.f_ext_1 = np.array([0, 0, 20, 0, 0, 0])
        # For point contact with friction
        self.Bci = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [
            0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.G = None
        self.mew = 0.7  
        self.fc = np.array([0, 0, 70, 0, 0, 70, 0, 0, 70])
        self.min = 70  # Taking 15 since max error will be 10
        self.solved_combination = None
        self.solution = None
        self.idt = 0
        self.valid_points = 0

   
    def select(self):
        points = np.array(self.pcd.points)
        normals_all = np.array(self.pcd.normals)
        point_indices = list(range(len(points)))
        point_combinations = combinations(point_indices, 3)

        for comb in point_combinations:
            indices = list(comb)
            self.idt = indices
            triplets = np.array(points)[indices]
            normals = normals_all[indices]
            if self.isochoose.choose(triplets,normals):
                self.transformation(triplets,normals)
                self.valid_points+=1
                
            else:
                continue
        print(self.valid_points)
            
    def transformation(self,triplets,normals):
        self.G = None
        for i in range(len(triplets)):
            normal = normals[i]
            point = triplets[i]
            # This gives us orientation of normal vector with x,y and z axis
            normal = -normal
            x_axis_angle = np.arctan2(np.linalg.norm(np.cross(
                normal, np.asarray([1, 0, 0]))), np.dot(normal, np.asarray([1, 0, 0])))
            y_axis_angle = np.arctan2(np.linalg.norm(np.cross(
                normal, np.asarray([0, 1, 0]))), np.dot(normal, np.asarray([0, 1, 0])))
            z_axis_angle = np.arctan2(np.linalg.norm(np.cross(
                normal, np.asarray([0, 0, 1]))), np.dot(normal, np.asarray([0, 0, 1])))
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
            if self.G is None:
                self.G = F_oi
            else:
                self.G = np.hstack((self.G, F_oi))     
        self.solve()



    def objective_function(self, fc):
        return np.linalg.norm(np.dot(self.G, fc)+self.f_ext_1)
    
    def constraint_1(self, fc):
        return self.mew*fc[2]-np.sqrt(fc[0]**2+fc[1]**2)

    def constraint_2(self, fc):
        return self.mew*fc[5]-np.sqrt(fc[3]**2+fc[4]**2)

    def constraint_3(self, fc):
        return self.mew*fc[8]-np.sqrt(fc[6]**2+fc[7]**2)
        
    # def constraint_4(self,fc):
    #     return fc[2]-fc[5]
    
    # def constraint_5(self,fc):
    #     return fc[5]-fc[8]

    def solve(self):
        con1 = {'type': 'ineq', 'fun': self.constraint_1}
        con2 = {'type': 'ineq', 'fun': self.constraint_2}
        con3 = {'type': 'ineq', 'fun': self.constraint_3}
        # con4 = {'type': 'eq', 'fun': self.constraint_4}
        # con5 = {'type': 'eq', 'fun': self.constraint_5}
        b = (0, 70)
        bnds = [b, b, b, b, b, b, b, b, b]

        cons = [con1, con2, con3]
        sol = minimize(self.objective_function, self.fc,
                       method='SLSQP', bounds=bnds, constraints=cons)
        err = self.objective_function(sol.x)
        if err < 10:
            print(
                f"Normal forces to be applied at the contacts {sol.x[2]} {sol.x[5]} {sol.x[8]} and corresponding error = {self.objective_function(sol.x)}")
            print(
                f"Friction forces in these points are {sol.x[0]} {sol.x[1]} {sol.x[3]} {sol.x[4]} {sol.x[6]} {sol.x[7]}")
        solution = list(sol.x)
        eef = np.asarray(self.isochoose.EEf())
        log1.log(solution,self.idt,err,self.pcd,eef)

       

def mesh2PointCloud(mesh):
    n_pts = 200 
    pcd = mesh.sample_points_uniformly(n_pts)
    return pcd


def force_visualizer(mesh, points, normals,center_point,eef):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(mesh)

    points1 = np.array([[center_point[0],center_point[1],-75], [center_point[0],center_point[1],75]])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points1)

    print(eef)
    point3 = np.array([eef])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point3)
    visualizer.add_geometry(pcd1)

 
    points5 = np.array([eef,points[0]])
    pcd5 = o3d.geometry.PointCloud()
    pcd5.points = o3d.utility.Vector3dVector(points5)
    lines5 = [[0, 1]]
    colors5 = [[0, 0, 1]]  # Red color
    line_set5 = o3d.geometry.LineSet()
    line_set5.points = o3d.utility.Vector3dVector(points5)
    line_set5.lines = o3d.utility.Vector2iVector(lines5)
    line_set5.colors = o3d.utility.Vector3dVector(colors5)
    visualizer.add_geometry(line_set5)

    points6 = np.array([eef,points[1]])
    pcd6 = o3d.geometry.PointCloud()
    pcd6.points = o3d.utility.Vector3dVector(points6)
    lines6 = [[0, 1]]
    line_set6 = o3d.geometry.LineSet()
    line_set6.points = o3d.utility.Vector3dVector(points6)
    line_set6.lines = o3d.utility.Vector2iVector(lines6)
    line_set6.colors = o3d.utility.Vector3dVector(colors5)
    visualizer.add_geometry(line_set6)

    points7 = np.array([eef,points[2]])
    pcd7 = o3d.geometry.PointCloud()
    pcd7.points = o3d.utility.Vector3dVector(points7)
    lines7 = [[0, 1]]
    line_set7 = o3d.geometry.LineSet()
    line_set7.points = o3d.utility.Vector3dVector(points7)
    line_set7.lines = o3d.utility.Vector2iVector(lines7)
    line_set7.colors = o3d.utility.Vector3dVector(colors5)
    visualizer.add_geometry(line_set7)
# Visualize the point cloud
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    visualizer.add_geometry(pcd)

    scene = o3d.geometry.PointCloud()
    points2 = np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]])
    scene.points = o3d.utility.Vector3dVector(points2)
    axes_line_set = o3d.geometry.LineSet()
    axes_line_set.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes_line_set.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    colors1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axes_line_set.colors = o3d.utility.Vector3dVector(colors1)
    visualizer.add_geometry(axes_line_set)


    lines = [[0, 1]]
    colors = [[1, 0, 0]]  # Red color
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points1)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    visualizer.add_geometry(line_set)
    visualizer.run()
    visualizer.destroy_window()


def main():
    eefy=[]
    start = time.time()
    name = "mesh"
    mesh_path = f"{name}.stl"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh2PointCloud(mesh)
    scaling_factor = 0.05

    pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)*scaling_factor)
    center_point = np.mean(np.asarray(pcd.points), axis=0)
    optimser = Optimization(pcd)
    optimser.select()
    [min, eefy] = log1.save_file(name)
    log1.cost_visualizer()
    # print( "Optimum force at fingers is ",pcd.normals[min[0]],pcd.normals[min[1]],pcd.normals[min[2]])
    ns = np.asarray([pcd.normals[min[0]],pcd.normals[min[1]],pcd.normals[min[2]]])
    
    pts = np.asarray([pcd.points[min[0]],pcd.points[min[1]],pcd.points[min[2]]])
    end = time.time()
    print(f"Total time to run={-start+end}")
    force_visualizer(mesh,pts/scaling_factor,ns,center_point/scaling_factor,eefy/scaling_factor)

log1 = logger()

if __name__ == "__main__":
    main()
