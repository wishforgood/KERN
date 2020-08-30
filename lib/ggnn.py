# Well, this file contains modules of GGNN_obj and GGNN_rel
import itertools
import os, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def torch_tile(target_tensor, dim, n):
    """Tile n times along the dim axis"""
    if dim == 0:
        return target_tensor.unsqueeze(0).transpose(0,1).repeat(1,n,1).view(-1,target_tensor.shape[1])
    else:
        return target_tensor.unsqueeze(0).transpose(0,1).repeat(1,1,n).view(target_tensor.shape[0], -1)

class GSNN(nn.Module):
    def __init__(self, num_obj_cls=151, num_rel_cls=51, time_step_num=3, hidden_dim=512, output_dim=512):
        super(GSNN, self).__init__()
        self.num_rel_cls = num_rel_cls
        self.num_obj_cls = num_obj_cls
        self.time_step_num = time_step_num
        self.output_dim = output_dim

        # if you want to use multi gpu to run this model, then you need to use the following line code to replace the last line code.
        # And if you use this line code, the model will save prior matrix as parameters in saved models.
        # self.matrix = nn.Parameter(torch.from_numpy(matrix_np), requires_grad=False)

        # here we follow the paper "Gated graph sequence neural networks" to implement GGNN, so eq3 means equation 3 in this paper.
        self.fc_node_trans_a = nn.Linear(hidden_dim, hidden_dim)
        self.fc_node_trans_b = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq6_w = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq6_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq7_w = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq7_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq8_w = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq8_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq9_w = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq9_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq10_w = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq10_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq11_w = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq11_u = nn.Linear(hidden_dim, hidden_dim)

        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output_obj = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_obj_cls = nn.Linear(output_dim, self.num_obj_cls)
        self.pair_hidden_trans = nn.Linear(2 * hidden_dim, hidden_dim)
        self.subject_trans = nn.Linear(hidden_dim, hidden_dim)
        self.object_trans = nn.Linear(hidden_dim, hidden_dim)
        self.edge_att_trans = nn.Linear(hidden_dim, 1)

    def forward(self, input_ggnn, node_confidence, pair_features):

        pair_features = pair_features.detach()
        # propogation process
        num_object = input_ggnn.size()[0]

        hidden = input_ggnn

        # matrix_np = np.ones((num_object, num_object)).astype(np.float32) / num_object

        # self.matrix = Variable(torch.from_numpy(matrix_np), requires_grad=False).cuda()

        global_feature = self.global_proj(torch.mean(hidden, 0))

        for t in range(self.time_step_num):

            # for i in range(num_object):
            #     for j in range(num_object):
            #         if i != j:
            #             self.matrix[i, j] = self.edge_att_trans(
            #                 self.subject_trans(hidden[i]) * self.object_trans(hidden[j]) * self.pair_hidden_trans(
            #                     torch.cat([hidden[i], hidden[j]], 0)))
            ae = self.pair_hidden_trans(torch.cat((torch_tile(self.subject_trans(hidden), 0, num_object), self.object_trans(hidden).repeat(num_object, 1)), 1))

            ze = torch.sigmoid(self.fc_eq9_w(ae) + self.fc_eq9_u(pair_features))

            re = torch.sigmoid(self.fc_eq10_w(ae) + self.fc_eq10_u(pair_features))

            he = torch.tanh(self.fc_eq11_w(ae) + self.fc_eq11_u(re * pair_features))

            pair_features = (1 - ze) * pair_features + ze * he

            # self.matrix = self.edge_att_trans(torch_tile(self.subject_trans(hidden), 0, num_object) * self.object_trans(hidden).repeat(num_object, 1))
            self.matrix = self.edge_att_trans(pair_features * torch_tile(self.subject_trans(hidden), 0, num_object) * self.object_trans(hidden).repeat(num_object, 1)).view(-1, num_object)

            av = torch.cat([torch.cat([self.matrix @ hidden], 0), global_feature.repeat(self.matrix.size(0), 1)], 1)

            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(hidden))

            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq4_u(hidden))

            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * hidden))

            hidden = (1 - zv) * hidden + zv * hv

            v_bar = node_confidence @ hidden

            zu = torch.sigmoid(self.fc_eq6_w(v_bar) + self.fc_eq6_u(global_feature))

            ru = torch.sigmoid(self.fc_eq7_w(v_bar) + self.fc_eq7_u(global_feature))

            hu = torch.tanh(self.fc_eq8_w(v_bar) + self.fc_eq8_u(ru * global_feature))

            global_feature = (1 - zu) * global_feature + zu * hu

        output_obj = torch.cat((hidden, input_ggnn), 1)
        output_obj = self.fc_output_obj(output_obj)
        output_obj = self.ReLU(output_obj)
        obj_dists = self.fc_obj_cls(output_obj.view(-1, self.output_dim))
        return obj_dists, output_obj, global_feature

class GOGNN(nn.Module):
    def __init__(self, num_obj_cls=151, num_rel_cls=51, time_step_num=3, hidden_dim=512, output_dim=512):
        super(GOGNN, self).__init__()
        self.num_rel_cls = num_rel_cls
        self.num_obj_cls = num_obj_cls
        self.time_step_num = time_step_num
        self.output_dim = output_dim

        # if you want to use multi gpu to run this model, then you need to use the following line code to replace the last line code.
        # And if you use this line code, the model will save prior matrix as parameters in saved models.
        # self.matrix = nn.Parameter(torch.from_numpy(matrix_np), requires_grad=False)

        # here we follow the paper "Gated graph sequence neural networks" to implement GGNN, so eq3 means equation 3 in this paper.
        self.fc_node_trans_a = nn.Linear(hidden_dim, hidden_dim)
        self.fc_node_trans_b = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq6_w = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq6_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq7_w = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq7_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq8_w = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq8_u = nn.Linear(hidden_dim, hidden_dim)

        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output_obj = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_obj_cls = nn.Linear(output_dim, self.num_obj_cls)

    def forward(self, input_ggnn):
        # print(input_ggnn)
        # print(pair_features)
        # propogation process
        num_object = input_ggnn.size()[0]

        hidden = input_ggnn

        matrix_np = np.ones((num_object, num_object)).astype(np.float32) / num_object

        self.matrix = Variable(torch.from_numpy(matrix_np), requires_grad=False).cuda()

        global_feature = self.global_proj(torch.mean(hidden, 0))

        source_hidden = hidden
        for t in range(self.time_step_num):

            # eq(2)
            # here we use some matrix operation skills
            av = torch.cat([torch.cat([self.matrix @ hidden], 0), global_feature.repeat(self.matrix.size(0), 1)], 1)

            # eq(3)
            # hidden = hidden.view(num_object*self.num_obj_cls, -1)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(hidden))

            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(hidden))

            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * hidden))

            hidden = (1 - zv) * hidden + zv * hv
            # hidden = hidden.view(num_object, self.num_obj_cls, -1)

            v_bar = torch.mean(hidden, 0)

            zu = torch.sigmoid(self.fc_eq6_w(v_bar) + self.fc_eq6_u(global_feature))

            ru = torch.sigmoid(self.fc_eq7_w(v_bar) + self.fc_eq6_u(global_feature))

            hu = torch.tanh(self.fc_eq8_w(v_bar) + self.fc_eq8_u(ru * global_feature))

            global_feature = (1 - zu) * global_feature + zu * hu

        output_obj = torch.cat((hidden, input_ggnn), 1)
        output_obj = self.fc_output_obj(output_obj)
        output_obj = self.ReLU(output_obj)
        obj_dists = self.fc_obj_cls(output_obj.view(-1, self.output_dim))
        return obj_dists, output_obj, global_feature


class GNN(nn.Module):
    def __init__(self, num_obj_cls=151, num_rel_cls=51, time_step_num=3, hidden_dim=512, output_dim=512,
                 use_knowledge=True, prior_matrix=''):
        super(GNN, self).__init__()
        self.num_rel_cls = num_rel_cls
        self.num_obj_cls = num_obj_cls
        self.time_step_num = time_step_num
        self.output_dim = output_dim

        # if you want to use multi gpu to run this model, then you need to use the following line code to replace the last line code.
        # And if you use this line code, the model will save prior matrix as parameters in saved models.
        # self.matrix = nn.Parameter(torch.from_numpy(matrix_np), requires_grad=False)

        # here we follow the paper "Gated graph sequence neural networks" to implement GGNN, so eq3 means equation 3 in this paper.
        self.fc_node_trans_a = nn.Linear(hidden_dim, hidden_dim)
        self.fc_node_trans_b = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_w = nn.Linear(4 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(4 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(4 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq6_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq6_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq7_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq7_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq8_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq8_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_edge_trans_a = nn.Linear(hidden_dim, hidden_dim)
        self.fc_edge_trans_b = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq9_w = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc_eq9_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq10_w = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc_eq10_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq11_w = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc_eq11_u = nn.Linear(hidden_dim, hidden_dim)

        self.edge_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.global_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.fc_output_obj = nn.Linear(2 * hidden_dim, output_dim)
        self.fc_output_rel = nn.Linear(3 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_obj_cls = nn.Linear(output_dim, self.num_obj_cls)
        self.fc_rel_cls = nn.Linear(output_dim, self.num_rel_cls)

    def forward(self, input_ggnn, rel_inds, obj_num_before):
        # propogation process
        num_object = input_ggnn.size()[0]

        num_rel = rel_inds.size(0)

        source_nodes = Variable(torch.LongTensor([(rel_ind[1] - obj_num_before.item()) for index, rel_ind in enumerate(rel_inds)]), requires_grad=False).cuda()

        target_nodes = Variable(torch.LongTensor([(rel_ind[2] - obj_num_before.item()) for index, rel_ind in enumerate(rel_inds)]), requires_grad=False).cuda()

        hidden = input_ggnn

        matrix_np = np.ones((num_object, num_object)).astype(np.float32) / num_object

        matrix_e2n_s_np = np.zeros((num_object, num_rel)).astype(np.float32)

        matrix_e2n_t_np = np.zeros((num_object, num_rel)).astype(np.float32)

        self.matrix = Variable(torch.from_numpy(matrix_np), requires_grad=False).cuda()

        self.matrix_e2n_s = Variable(torch.from_numpy(matrix_e2n_s_np), requires_grad=False).cuda()

        self.matrix_e2n_t = Variable(torch.from_numpy(matrix_e2n_t_np), requires_grad=False).cuda()

        self.matrix_e2n_s = self.matrix_e2n_s.scatter_(0, source_nodes.unsqueeze(0), 1)

        self.matrix_e2n_s = self.matrix_e2n_s / self.matrix_e2n_s.sum(1).repeat(num_rel, 1).transpose(0, 1)

        self.matrix_e2n_s[self.matrix_e2n_s != self.matrix_e2n_s] = 0

        self.matrix_e2n_t = self.matrix_e2n_t.scatter_(0, target_nodes.unsqueeze(0), 1)

        self.matrix_e2n_t = self.matrix_e2n_t / self.matrix_e2n_t.sum(1).repeat(num_rel, 1).transpose(0, 1)

        self.matrix_e2n_t[self.matrix_e2n_t != self.matrix_e2n_t] = 0

        edge_feature = self.edge_proj(torch.cat([hidden.index_select(0, source_nodes), hidden.index_select(0, target_nodes)], 1))

        global_feature = self.global_proj(torch.cat([torch.mean(edge_feature, 0), torch.mean(hidden, 0)]))

        for t in range(self.time_step_num):
            ae = torch.cat([self.fc_edge_trans_a(edge_feature),
                            self.fc_edge_trans_b(torch.cat([hidden.index_select(0, source_nodes),
                                                            hidden.index_select(0, target_nodes)], 1)),
                            global_feature.repeat(num_rel, 1)], 1)

            ze = torch.sigmoid(self.fc_eq9_w(ae) + self.fc_eq9_u(edge_feature))

            re = torch.sigmoid(self.fc_eq10_w(ae) + self.fc_eq10_u(edge_feature))

            he = torch.tanh(self.fc_eq11_w(ae) + self.fc_eq11_u(re * edge_feature))

            edge_feature = (1 - ze) * edge_feature + ze * he

            # eq(2)
            # here we use some matrix operation skills
            av = torch.cat([self.fc_node_trans_a(self.matrix_e2n_s @ edge_feature),
                            self.fc_node_trans_b(self.matrix_e2n_t @ edge_feature),
                            torch.cat([self.matrix @ hidden], 0), global_feature.repeat(self.matrix.size(0), 1)], 1)

            # eq(3)
            # hidden = hidden.view(num_object*self.num_obj_cls, -1)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(hidden))

            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(hidden))

            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * hidden))

            hidden = (1 - zv) * hidden + zv * hv
            # hidden = hidden.view(num_object, self.num_obj_cls, -1)

            e_bar = torch.mean(edge_feature, 0)

            v_bar = torch.mean(hidden, 0)

            e_v = torch.cat([e_bar, v_bar], 0)

            zu = torch.sigmoid(self.fc_eq6_w(e_v) + self.fc_eq6_u(global_feature))

            ru = torch.sigmoid(self.fc_eq7_w(e_v) + self.fc_eq6_u(global_feature))

            hu = torch.tanh(self.fc_eq8_w(e_v) + self.fc_eq8_u(ru * global_feature))

            global_feature = (1 - zu) * global_feature + zu * hu

        output_obj = torch.cat((hidden, input_ggnn), 1)
        output_obj = self.fc_output_obj(output_obj)
        output_obj = self.ReLU(output_obj)
        obj_dists = self.fc_obj_cls(output_obj.view(-1, self.output_dim))
        output_rel = torch.cat(
            [hidden.index_select(0, source_nodes), hidden.index_select(0, target_nodes), edge_feature], 1)
        output_rel = self.fc_output_rel(output_rel)
        rel_dists = self.fc_rel_cls(output_rel.view(-1, self.output_dim))
        return [obj_dists, rel_dists]


class GGNNRel(nn.Module):
    def __init__(self, num_rel_cls=51, time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=False,
                 prior_matrix=''):
        super(GGNNRel, self).__init__()
        self.num_rel_cls = num_rel_cls
        self.time_step_num = time_step_num
        if use_knowledge:
            self.matrix = np.load(prior_matrix).astype(np.float32)
        self.use_knowledge = use_knowledge

        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)


        self.fc_output = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_rel_cls = nn.Linear((self.num_rel_cls + 2) * output_dim, self.num_rel_cls)

    def forward(self, rel_inds, sub_obj_preds, input_ggnn):
        (input_rel_num, node_num, _) = input_ggnn.size()
        assert input_rel_num == len(rel_inds)
        batch_in_matrix_sub = np.zeros((input_rel_num, 2, self.num_rel_cls), dtype=np.float32)

        if self.use_knowledge:  # construct adjacency matrix depending on the predicted labels of subject and object.
            for index, rel in enumerate(rel_inds):
                batch_in_matrix_sub[index][0] = \
                    self.matrix[sub_obj_preds[index, 0].cpu().data, sub_obj_preds[index, 1].cpu().data]
                batch_in_matrix_sub[index][1] = batch_in_matrix_sub[index][0]
        else:
            for index, rel in enumerate(rel_inds):
                batch_in_matrix_sub[index][0] = 1.0 / float(self.num_rel_cls)
                batch_in_matrix_sub[index][1] = batch_in_matrix_sub[index][0]
        batch_in_matrix_sub_gpu = Variable(torch.from_numpy(batch_in_matrix_sub), requires_grad=False).cuda()
        del batch_in_matrix_sub
        hidden = input_ggnn
        for t in range(self.time_step_num):
            # eq(2)
            # becase in this case, A^(out) == A^(in), so we use function "repeat"
            # What is A^(out) and A^(in)? Please refer to paper "Gated graph sequence neural networks"
            av = torch.cat((torch.bmm(batch_in_matrix_sub_gpu, hidden[:, 2:]),
                            torch.bmm(batch_in_matrix_sub_gpu.transpose(1, 2), hidden[:, :2])), 1).repeat(1, 1, 2)
            av = av.view(input_rel_num * node_num, -1)
            flatten_hidden = hidden.view(input_rel_num * node_num, -1)
            # eq(3)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_hidden))
            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq4_u(flatten_hidden))
            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_hidden))
            flatten_hidden = (1 - zv) * flatten_hidden + zv * hv
            hidden = flatten_hidden.view(input_rel_num, node_num, -1)

        output = torch.cat((flatten_hidden, input_ggnn.view(input_rel_num * node_num, -1)), 1)
        output = self.fc_output(output)
        output = self.ReLU(output)

        rel_dists = self.fc_rel_cls(output.view(input_rel_num, -1))
        return rel_dists


class GGNNObj(nn.Module):
    def __init__(self, num_obj_cls=151, time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True,
                 prior_matrix=''):
        super(GGNNObj, self).__init__()
        self.num_obj_cls = num_obj_cls
        self.time_step_num = time_step_num
        self.output_dim = output_dim

        if use_knowledge:
            matrix_np = np.load(prior_matrix).astype(np.float32)
        else:
            matrix_np = np.ones((num_obj_cls, num_obj_cls)).astype(np.float32) / num_obj_cls

        self.matrix = Variable(torch.from_numpy(matrix_np), requires_grad=False).cuda()
        # if you want to use multi gpu to run this model, then you need to use the following line code to replace the last line code.
        # And if you use this line code, the model will save prior matrix as parameters in saved models.
        # self.matrix = nn.Parameter(torch.from_numpy(matrix_np), requires_grad=False)

        # here we follow the paper "Gated graph sequence neural networks" to implement GGNN, so eq3 means equation 3 in this paper.
        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_obj_cls = nn.Linear(self.num_obj_cls * output_dim, self.num_obj_cls)

    def forward(self, input_ggnn):
        # propogation process
        num_object = input_ggnn.size()[0]
        hidden = input_ggnn.repeat(1, self.num_obj_cls).view(num_object, self.num_obj_cls, -1)
        for t in range(self.time_step_num):
            # eq(2)
            # here we use some matrix operation skills
            hidden_sum = torch.sum(hidden, 0)
            av = torch.cat(
                [torch.cat([self.matrix.transpose(0, 1) @ (hidden_sum - hidden_i) for hidden_i in hidden], 0),
                 torch.cat([self.matrix @ (hidden_sum - hidden_i) for hidden_i in hidden], 0)], 1)

            # eq(3)
            hidden = hidden.view(num_object * self.num_obj_cls, -1)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(hidden))

            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(hidden))

            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * hidden))

            hidden = (1 - zv) * hidden + zv * hv
            hidden = hidden.view(num_object, self.num_obj_cls, -1)

        output = torch.cat((hidden.view(num_object * self.num_obj_cls, -1),
                            input_ggnn.repeat(1, self.num_obj_cls).view(num_object * self.num_obj_cls, -1)), 1)
        output = self.fc_output(output)
        output = self.ReLU(output)
        obj_dists = self.fc_obj_cls(output.view(-1, self.num_obj_cls * self.output_dim))
        return obj_dists

