import open3d as o3d
import numpy as np
import math

class Isotrichoose:
    def __init__(self):
        self.t1 =0
        self.t2 =0
        self.t3 =0
        self.dist =0
        self.n1 =0
        self.n2 =0
        

    def isclose(self,a,b):
        if abs(a-b)<0.1:
            return True
        else:
            return False

    def are_parallel(self,vec1, vec2):
        theta = np.arctan2(np.linalg.norm(np.cross(
                vec1, vec2)), np.dot(vec1, vec2))
        if theta*180/np.pi<=2 and theta*180/np.pi>=0:
            return True
        else:
            return False

    def is_antiparallel(self,vec1, vec2):
        theta = np.arctan2(np.linalg.norm(np.cross(
               vec1, vec2)), np.dot(vec1, vec2))
        if theta*180/np.pi<=180 and theta*180/np.pi>=178:
            return True
        else:
            return False
    
    def EEFpose(self,s1,s2):
        altitude = (s1**2 - (s2**2)/4)**0.5
        base = altitude/2 - 4.4704
        hpt = 9.4488
        palm_dist = 9.2075 + (hpt**2 - base**2)**0.5
        return palm_dist

    def EEf(self):
        centr = (self.t1 +(self.t2 + self.t3)/2)/2
        normal = np.cross((self.t1 -self.t2),(self.t1 -self.t3))
        pose = centr + self.dist*normal/np.linalg.norm(normal)
        return np.array(pose)

    def choose(self,triplet,normals):
        side1 = np.linalg.norm(triplet[0]-triplet[1])
        side2 = np.linalg.norm(triplet[1]-triplet[2])
        side3 = np.linalg.norm(triplet[2]-triplet[0])
        # the first condition has many cases but checking only this
        if self.isclose(side1,side2) and self.isclose(side1,side3):
            if self.isclose(side1,7.302):
                if self.are_parallel(normals[0],normals[1]) and self.is_antiparallel(normals[0],normals[2]):
                    self.dist = self.EEFpose(side1,side3)
                    self.n1 = normals[1]
                    self.n2 = normals[2]
                    self.t1 = triplet[1]
                    self.t2 = triplet[2]
                    self.t3 = triplet[0]
                    return True
        elif self.isclose(side1,side2):
            print(f"side1={side1}")
            if side1<18.8695779 and side1>0.1 and self.isclose(side3,7.302):
               if self.are_parallel(normals[0],normals[2]) and self.is_antiparallel(normals[0],normals[1]):
                    self.dist = self.EEFpose(side1,side3)
                    self.n1 = normals[1]
                    self.n2 = normals[2]
                    self.t1 = triplet[1]
                    self.t2 = triplet[2]
                    self.t3 = triplet[0]
                    return True
        elif self.isclose(side1,side3):
            print(f"side3={side3}")
            if side1<18.8695779 and side1>0.1 and self.isclose(side2,7.302):
                if self.are_parallel(normals[2],normals[1]) and self.is_antiparallel(normals[0],normals[2]):
                    self.dist = self.EEFpose(side1,side2)
                    self.n1 = normals[0]
                    self.n2 = normals[2]
                    self.t1 = triplet[0]
                    self.t2 = triplet[1]
                    self.t3 = triplet[2]
                    return True
        elif self.isclose(side3,side2):
            print(f"side2={side2}")
            if side2<18.8695779 and side2>0.1 and self.isclose(side1,7.302):
                if self.are_parallel(normals[0],normals[1]) and self.is_antiparallel(normals[0],normals[2]):
                    self.dist = self.EEFpose(side2,side1)
                    self.n1 = normals[2]
                    self.n2 = normals[1]
                    self.t1 = triplet[2]
                    self.t2 = triplet[1]
                    self.t3 = triplet[0]
                    return True

        return False
