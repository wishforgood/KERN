import itertools

import torch
from torch.autograd import Variable
import os
import numpy as np

# num_object = 4
# num_rel = 16
# edge_pairs = list(itertools.product(list(range(num_object)), repeat=2))
# source_nodes = Variable(torch.LongTensor(list(zip(*edge_pairs))[0]), requires_grad=False).cuda()
# matrix_e2n_s_np = np.zeros((num_object, num_rel)).astype(np.float32)
# matrix_e2n_s = Variable(torch.from_numpy(matrix_e2n_s_np), requires_grad=False).cuda()
# print(source_nodes)
# print(matrix_e2n_s)
# print(source_nodes.unsqueeze(0))
# print(matrix_e2n_s.scatter_(0, source_nodes.unsqueeze(0), 1))
https://epubs.siam.org/doi/abs/10.1137/040605503?journalCode=sjope8
