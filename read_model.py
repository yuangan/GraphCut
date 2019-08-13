import os
import numpy as np
import time

class ModelReader(object):
    def __init__(self, obj_path):
        self.f = obj_path
        self.vertices = []
        self.normals = []
        
        self.face_normals = []
        self.face_points = []
        self.faces = []
        self.OFF = False
        self.area_sum = []
        if obj_path[-3:] == 'off' or obj_path[-3:] == 'OFF':
            self.OFF = True
        self.read(obj_path)       

    def norm(self, a):
        bias = np.array([1e-20, 1e-20, 1e-20])
        a = a + bias
        a = a / np.sqrt(np.dot(a, a))
        return a
        
    def read(self, swapyz=False):
        #read obj or off
		print('read obj', self.f)
        if self.OFF:
            r = open(self.f, "r")
            r.readline()
            r.readline()
            all_lines = r.readlines()
            for line in all_lines:
                values = line.split()
                if len(values)==3:
                    #point
                    v = list(map(float, values[0:3]))
                    self.vertices.append(v)
                else:
                    #face
                    face = []
                    for i in values[1:]:
                        face.append(int(i)+1)
                    self.faces.append((face, []))

        else:           
            for line in open(self.f, "r"):
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    v = list(map(float, values[1:4]))
                    if swapyz:
                       v = v[0], v[2], v[1]
                    self.vertices.append(v)
                elif values[0] == 'vn':
                    v = list(map(float, values[1:4]))
                    if swapyz:
                        v = v[0], v[2], v[1]
                    self.normals.append(v)
                elif values[0] == 'f':
                    face = []
                    norms = []
                    for v in values[1:]:
                        w = v.split('/')
                        face.append(int(w[0]))
                        if len(w) >= 3 and len(w[2]) > 0:
                            norms.append(int(w[2]))
                        else:
                            norms.append(0)
                    self.faces.append((face, norms))

        #compute face normals
        self.face_center_points = []
        bias = np.array([1e-20, 1e-20, 1e-20])
        for face in self.faces:
            vertices, _ = face
            normals = []
            polygon = []
            
            now_p1 = np.float32(self.vertices[vertices[0]-1]) - np.float32(self.vertices[vertices[1]-1])
            now_p2 = np.float32(self.vertices[vertices[1]-1]) - np.float32(self.vertices[vertices[2]-1])
            now_face_normal = np.cross(now_p1, now_p2)
            now_face_normal = self.norm(now_face_normal)
            
            self.face_normals.append(now_face_normal)
            for p in vertices:
                polygon.append(self.vertices[p-1])
            self.face_points.append(polygon)
            mid_p = np.mean(polygon, axis=0)
            self.face_center_points.append(mid_p)
            
    def compute_dist(self, a, b):
        return np.sqrt(np.sum((a-b)**2))
    
    def compute_pairwise_features(self):
        start_time = time.time()
        print('......compute pairwise features......', end='')
        points_adj_faces = [[] for i in range(len(self.vertices))]
        
        for face_idx, face in enumerate(self.faces):
            vertices, _ = face
            for p in vertices:
                if not face_idx in points_adj_faces[p-1]:
                    points_adj_faces[p-1].append(face_idx)
                    
        bias = np.array([1e-20, 1e-20, 1e-20])
        self.faces_adj_faces = [[] for i in range(len(self.faces))]
        self.faces_adj_dist = [[] for i in range(len(self.faces))]
        self.faces_adj_da = [[] for i in range(len(self.faces))]
        
        for face_idx, face in enumerate(self.faces):
            vertices, _  = face
            now_face_normal = self.face_normals[face_idx]
            now_face_center_point = self.face_center_points[face_idx]
            for i, p in enumerate(vertices):
                from_p = p
                to_p = vertices[(i+1)%(len(vertices))]
                
                from_xyz_np_array = np.array(self.vertices[from_p-1])
                to_xyz_np_array = np.array(self.vertices[to_p-1])
                
                mid_point = 0.5*(from_xyz_np_array + to_xyz_np_array)
                
                for p_adj_face in points_adj_faces[from_p-1]:
                    if p_adj_face == face_idx:
                        continue
                    if p_adj_face in self.faces_adj_faces[face_idx] or face_idx in self.faces_adj_faces[p_adj_face]:
                        continue
                        
                    adj_face_center_point = self.face_center_points[p_adj_face]
                    temp_vertices, _ = self.faces[p_adj_face]
                    adj_face_normal = self.face_normals[p_adj_face]
                    if to_p in temp_vertices:
                        
                        edge_dist = self.compute_dist(now_face_center_point, mid_point) + self.compute_dist(adj_face_center_point, mid_point)
                        
                        cos_theta = np.dot(now_face_normal, adj_face_normal)
                        sin_theta = np.cross(now_face_normal, adj_face_normal)
						
                        sin_theta_len = np.sqrt(np.sum(sin_theta**2))
                        da = np.arctan2(sin_theta_len, cos_theta)
                        edge_ = self.norm(to_xyz_np_array - from_xyz_np_array)
                        
                        da = np.pi + da * np.sign(np.dot(sin_theta, edge_))

                        da = da / np.pi
                        
                        if not p_adj_face in self.faces_adj_faces[face_idx]:
                            self.faces_adj_faces[face_idx].append(p_adj_face)
                            self.faces_adj_dist[face_idx].append(edge_dist)
                            self.faces_adj_da[face_idx].append(da)

                        if not face_idx in self.faces_adj_faces[p_adj_face]:
                            self.faces_adj_faces[p_adj_face].append(face_idx)
                            self.faces_adj_dist[p_adj_face].append(edge_dist)
                            self.faces_adj_da[p_adj_face].append(da)                            
        print('......time passed %f......'%(time.time()-start_time))
        
    def write_obj_with_seg_group(self, name, dir, face_seg_list):
        output_path = os.path.join(dir, name)
        output_file = open(output_path, 'w')
        for point_idx in range(len(self.vertices)):
            output_file.write('v %f %f %f\n'%(self.vertices[point_idx][0], self.vertices[point_idx][2], self.vertices[point_idx][1]))
            
        group_name_list = list(set(face_seg_list))
        num_group = len(group_name_list)
        
        write_group_text = {}
        for group_name in group_name_list:
            write_group_text[group_name] = '#\n#object %s#\ng %s\n'%(str(group_name), str(group_name))
            
        for face_idx, face_label in enumerate(face_seg_list):
            v, _ = self.faces[face_idx]
            write_group_text[face_label] += 'f '
            for i in range(len(v)):
                write_group_text[face_label] +='%d '%(v[i])
            write_group_text[face_label]+='\n'
            
        for group_name in write_group_text:
            output_file.write('\n')
            output_file.write(write_group_text[group_name])
        output_file.close()
        print(name, 'write obj!')
        return True
        

def read_seg_file(file):
    return np.int32(open(file,'r').read().split())-1

if __name__ == '__main__': 
    model = ModelReader('201.off')
    seg = read_seg_file('201_not_good.seg')
    model.compute_pairwise_features()
    model.write_obj_with_seg_group('201_seg.obj','./',seg)

