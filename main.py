import os
import numpy as np
import time
from graphcut import *
from read_model import *

if __name__ == '__main__': 
    model = ModelReader('201.off')
    seg = read_seg_file('201_not_good.seg')
    model.compute_pairwise_features()
    model.write_obj_with_seg_group('201_seg_before.obj','./',seg)
    
    #mutli-label graphcut, only voting
    label_list = list(set(seg))
    gc_solver = GC()
    mutil_label_output = np.zeros((len(label_list), len(seg)))
    for idx, l in enumerate(label_list):
        gc_solver.reset_graph(len(seg))
        gc_solver.set_unary_item(seg, l)
        gc_solver.set_pairwise_item(model, seg, l)
        gc_solver.refine()
        mutil_label_output[idx,:] = gc_solver.output_labels()
    out_idx = np.argmax(mutil_label_output, axis=0)
    
    seg_after = [label_list[i] for i in out_idx]
    model.write_obj_with_seg_group('201_seg_after.obj','./',seg_after)

