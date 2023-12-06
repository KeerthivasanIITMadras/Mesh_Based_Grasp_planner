import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
import time

import csv

file_path = '/home/keerthivasan/keerthi/nkp/results/CubeSquare.csv'

with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader, None)
    
    min_value = float('inf') 
    min = None  # Initialize variable to store the row with the minimum value
    
    for row in csv_reader:
        last_column_value = float(row[-1])
        
        # Check if the current value is smaller than the current minimum
        if last_column_value < min_value:
            min_value = last_column_value
            min = row[:]  # Save a copy of the current row
            
print(f"The minimum value from the last column is: {min_value}")

# if min:
#     print("Corresponding values in the first three columns:")
#     print(min[:3])  # Display values from the first three columns of the row with the minimum value

def force_visualizer(mesh, points, normals,center_point):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(mesh)
    print(normals)
    points1 = np.array([[center_point[0],center_point[1],-75], [center_point[0],center_point[1],75]])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points1)
 
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


def mesh2PointCloud(mesh):
    n_pts = 200
    pcd = mesh.sample_points_uniformly(n_pts,seed=32)
    return pcd

mesh_path = "/home/keerthivasan/keerthi/nkp/cad_files/CubeSquare.stl"
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
pcd = mesh2PointCloud(mesh)

min[0] = int(min[0])
min[1] = int(min[1])
min[2] = int(min[2])

ns = np.asarray([pcd.normals[min[0]],pcd.normals[min[1]],pcd.normals[min[2]]])
    
pts = np.asarray([pcd.points[min[0]],pcd.points[min[1]],pcd.points[min[2]]])
center_point = np.mean(np.asarray(pcd.points), axis=0)

force_visualizer(mesh,pts,ns,center_point)