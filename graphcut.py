import maxflow
import numpy as np
import os

class GC(object):
	def __init__(self):
		self.lamda_1 = 1.
		self.lamda_2 = 30.0
		self.g = None
		
	def reset_graph(self, node_num):
		if self.g:
			del self.g
		self.g = maxflow.Graph[int](node_num, 2)
		self.nodes = self.g.add_nodes(node_num)
		
	def set_unary_item(self, pred_seg, need_label):
		wrong = -np.log(0.1)
		right = -np.log(0.9)
		for idx, face_label in enumerate(pred_seg):
			if face_label == need_label:
				self.g.add_tedge(self.nodes[idx], self.lamda_1*right, self.lamda_1*wrong)
			else:
				self.g.add_tedge(self.nodes[idx], self.lamda_1*wrong, self.lamda_1*right)
				
	def set_pairwise_item(self, obj, pred_seg, need_label):
		for face_from, face_to_list in enumerate(obj.faces_adj_faces):
			for idx, face_to in enumerate(face_to_list):
				da = obj.faces_adj_da[face_from][idx]
				da = min(1.-min(da,1.)+1e-20, 1.)
				dst = obj.faces_adj_dist[face_from][idx]
				
				edge_feature = -np.log(da)*dst
				self.g.add_edge(self.nodes[face_from], self.nodes[face_to], self.lamda_2*edge_feature, self.lamda_2*edge_feature)
				
	def refine(self):
		flow = self.g.maxflow()
		self.sgm = self.g.get_grid_segments(self.nodes)
		
	def output_labels(self):
		return np.int32(self.sgm)
		
	