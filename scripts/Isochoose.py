import open3d as o3d
import numpy as np

class Isotrichoose:
    def __init__(self):
        pass

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
    
    def choose(self,triplet,normals):
        side1 = np.linalg.norm(triplet[0]-triplet[1])
        side2 = np.linalg.norm(triplet[1]-triplet[2])
        side3 = np.linalg.norm(triplet[2]-triplet[0])
        # the first condition has many cases but checking only this
        if self.isclose(side1,side2) and self.isclose(side1,side3):
            if self.isclose(side1,7.302):
                if self.are_parallel(normals[0],normals[1]) and self.is_antiparallel(normals[0],normals[2]):
                    return True
        elif self.isclose(side1,side2):
            print(f"side1={side1}")
            if side1<18.8695779 and side1>0.1 and self.isclose(side3,7.302):
               if self.are_parallel(normals[0],normals[2]) and self.is_antiparallel(normals[0],normals[1]):
                    return True
        elif self.isclose(side1,side3):
            print(f"side3={side3}")
            if side1<18.8695779 and side1>0.1 and self.isclose(side2,7.302):
                if self.are_parallel(normals[2],normals[1]) and self.is_antiparallel(normals[0],normals[2]):
                    
                    return True
        elif self.isclose(side3,side2):
            print(f"side2={side2}")
            if side2<18.8695779 and side1>0.1 and self.isclose(side1,7.302):
                if self.are_parallel(normals[0],normals[1]) and self.is_antiparallel(normals[0],normals[2]):
                    
                    return True

        return False
