"""Training and testing the dual learning algorithm for unbiased learning to rank.

See the following paper for more information on the dual learning algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
from ultra.learning_algorithm.base_propensity_model import BasePropensityModel
import ultra.utils
import torch.autograd
import numpy as np


# def sigmoid_prob(logits):
#     return torch.sigmoid(logits - torch.mean(logits, -1, keepdim=True))
#
# class DenoisingNet(nn.Module):
#     def __init__(self, input_vec_size):
#         super(DenoisingNet, self).__init__()
#         self.linear_layer = nn.Linear(input_vec_size, 1)
#         self.elu_layer = nn.ELU()
#         self.propensity_net = nn.Sequential(self.linear_layer, self.elu_layer)
#         self.list_size = input_vec_size
#
#     def forward(self, input_list):
#         output_propensity_list = []
#         for i in range(self.list_size):
#             # Add position information (one-hot vector)
#             click_feature = [
#                 torch.unsqueeze(
#                     torch.zeros_like(
#                         input_list[i]), -1) for _ in range(self.list_size)]
#             click_feature[i] = torch.unsqueeze(
#                 torch.ones_like(input_list[i]), -1)
#             # Predict propensity with a simple network
#             # print(torch.cat(click_feature, 1).shape) 256*10
#             output_propensity_list.append(
#                 self.propensity_net(
#                     torch.cat(
#                         click_feature, 1)))
#         return torch.cat(output_propensity_list, 1) #256*10，有负数
#
# class PropensityModel_logit(nn.Module):
#     def __init__(self, list_size):
#         super(PropensityModel_logit, self).__init__()
#         self.IOBM_model = nn.Parameter(torch.ones(1, list_size))
#
#     def forward(self):
#         return self.IOBM_model  # (1, T)

class IOBM(nn.Module):

    def __init__(self, feature_size, rank_list_size, **kwargs):
        super(IOBM, self).__init__()
        print("Propensity: use IOBM")

        self.hparams = {
            "activation": "tanh",
            "units": 8,
            "embedding_size": 4,
            "position_embedding_size": 4,
            "bidirection": True
        }

        print("IOBM params: " + str(self.hparams))

        self.position_embedding = nn.Embedding(rank_list_size, self.hparams["position_embedding_size"])
        self.click_label_embedding = nn.Embedding(2, self.hparams["embedding_size"])
        self.lstm = nn.LSTM(input_size=self.hparams["position_embedding_size"] + self.hparams["embedding_size"],
                            hidden_size=self.hparams["units"],
                            batch_first=True)
        self.dense = nn.Linear(feature_size + self.hparams["position_embedding_size"] + self.hparams["embedding_size"], self.hparams["position_embedding_size"] + self.hparams["embedding_size"])
        self.dense_1 = nn.Linear(feature_size + self.hparams["units"] * 2, self.hparams["units"] * 2)
        self.dense_2 = nn.Linear(self.hparams["units"] * 2, 1)

    def lstm_layer(self, inpt):
        x, _ = self.lstm(inpt)
        return x

    def additive_attention(self, context, sequence):
        x = torch.cat([context, sequence], dim=-1)  # (B, T, C1 + C2)
        C2 = sequence.size(-1)
        if C2 == self.hparams["position_embedding_size"] + self.hparams["embedding_size"]:
            x = torch.tanh(self.dense(x))  # (B, T, C2)
        else:
            x = torch.tanh(self.dense_1(x))  # (B, T, C2)
        x = F.softmax(x, dim=-1)  # (B, T, C2)
        x = x * C2
        # print("additive_attention: %s vs %s => %s" % (context.size(), sequence.size(), x.size()))
        return x, x * sequence

    def forward(self, click_label, letor_features):
        list_size = click_label.size(1)
        batch_size = click_label.size(0)
        inputs = []

        # context
        with torch.no_grad():
            # context = torch.mean(torch.stack(learning_model.context_embedding), dim=0)  # (B, T, C)
            context = torch.mean(letor_features, dim=1, keepdim=True)  # (B, 1, C)
            context_dim = context.size(-1)
            # print("context dim: " + str(context_dim))
            context = context.repeat(1, list_size, 1)  # (B, T, C)
            context = context.to(torch.float32)

        # position embedding
        position = torch.arange(list_size, device=click_label.device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        position = self.position_embedding(position).squeeze(2)  # (1, T, E)
        position = position.repeat(batch_size, 1, 1)  # (B, T, E)
        inputs.append(position.float())

        # click label embedding
        click_label = click_label.unsqueeze(-1)  # (B, T, 1)
        if self.hparams["embedding_size"] != 0:
            click_label = self.click_label_embedding(click_label.long()).squeeze(2)  # (B, T, E)
        else:
            click_label = click_label * 2 - 1
        inputs.append(click_label.float())

        x = torch.cat(inputs, dim=-1)  # (B, T, C + E)

        # print(x.shape)
        # print(context.shape)
        #
        # print(context.dtype)
        # print(x.dtype)

        # attention
        att_x, x = self.additive_attention(context, x)
        l1 = self.hparams["position_embedding_size"]
        # l2 = l1 + self.hparams["embedding_size"]
        # l3 = l2 + 1

        # shift
        x = x.unbind(dim=1)
        x = [torch.zeros_like(x[0])] + list(x) + [torch.zeros_like(x[0])]
        forward = torch.stack(x[:-2], dim=1)
        p = self.lstm_layer(forward)


        backward = torch.stack(x[-1:1:-1], dim=1)
        q = self.lstm_layer(backward)
        q = torch.flip(q, dims=[1])
        x = torch.cat([p, q], dim=-1)
        att_y, x = self.additive_attention(context, x)
        y = self.dense_2(x).squeeze(-1)

        return y

class DLA_IOBM(BaseAlgorithm):
    """The Dual Learning Algorithm for unbiased learning to rank.

    This class implements the Dual Learning Algorithm (DLA) based on the input layer
    feed. See the following paper for more information on the algorithm.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

    """

    def __init__(self, data_set, exp_settings):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build DLA_IOBM')

        self.hparams = ultra.utils.hparams.HParams(
            # learning_rate=0.05,                 # Learning rate.
            learning_rate=exp_settings['ln'],
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_loss',            # Select Loss function
            # the function used to convert logits to probability distributions
            logits_to_prob='softmax',
            # The learning rate for ranker (-1 means same with learning_rate).
            propensity_learning_rate=-1,
            ranker_loss_weight=1.0,            # Set the weight of unbiased ranking loss
            # Set strength for L2 regularization.
            l2_loss=0.0,
            max_propensity_weight=-1,      # Set maximum value for propensity weights
            constant_propensity_initialization=False,
            # Set true to initialize propensity with constants.
            grad_strategy='ada',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.cuda = torch.device('cuda')
        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.test_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        if 'selection_bias_cutoff' in exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
            #self.propensity_model = DenoisingNet(self.rank_list_size)

            # self.propensity_para = [torch.tensor([1.0 - i/10]) for i in range(self.rank_list_size)]
            # self.propensity_para = [torch.tensor([0.0]) for i in range(self.rank_list_size)]
        self.model = self.create_model(self.feature_size)

        self.IOBM_model = IOBM(self.feature_size, self.rank_list_size)
        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
            self.IOBM_model = self.IOBM_model.to(device=self.cuda)
            # for i in range(len(self.propensity_para)):
            #     self.propensity_para[i] = self.propensity_para[i].to(device=self.cuda)
            #     self.propensity_para[i].requires_grad = True
                #print(self.propensity_para[i].is_leaf)
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))

        if self.hparams.propensity_learning_rate < 0:
            self.propensity_learning_rate = float(self.hparams.learning_rate)
        else:
            self.propensity_learning_rate = float(self.hparams.propensity_learning_rate)
        self.learning_rate = float(self.hparams.learning_rate)

        self.global_step = 0

        # Select logits to prob function
        self.logits_to_prob = nn.Softmax(dim=-1)
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob

        self.optimizer_func = torch.optim.Adagrad
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD

        print('Loss Function is ' + self.hparams.loss_func)
        # Select loss function
        self.loss_func = None
        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss_func = self.sigmoid_loss_on_list
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss_func = self.pairwise_loss_on_list
        else:  # softmax loss without weighting
            self.loss_func = self.softmax_loss

    def get_input_feature_list(self, input_id_list):
        """Copy from base_algorithm.get_ranking_scores()
        """
        PAD_embed = np.zeros((1, self.feature_size), dtype=np.float32)
        letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(torch.from_numpy(np.take(letor_features, input_id_list[i], 0)))
        return input_feature_list

    def separate_gradient_update(self):
        # denoise_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()
        propensity_params = self.IOBM_model.parameters()
        # Select optimizer

        if self.hparams.l2_loss > 0:
            # for p in denoise_params:
            #    self.exam_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * self.l2_loss(p)
            for p in propensity_params:
                self.exam_loss += self.hparams.l2_loss * self.l2_loss(p)
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss
        # opt_para = []
        # for i in range(len(self.propensity_para)):
        #     opt_para.append(self.optimizer_func([self.propensity_para[i]], self.propensity_learning_rate))
        opt_ranker = self.optimizer_func(self.model.parameters(), self.learning_rate)
        opt_propensity = self.optimizer_func(self.IOBM_model.parameters(), self.propensity_learning_rate)

        opt_ranker.zero_grad()
        opt_propensity.zero_grad()

        # for i in range(len(self.propensity_para)):
        #     self.propensity_para[i].retain_grad()

        self.loss.backward()

        # for i in range(len(self.propensity_para)):
        #     print(self.propensity_para[i].is_leaf)
        #     print(self.propensity_para[i].grad)

        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(self.IOBM_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)

        # for i in range(len(self.propensity_para)):
        #     opt_para[i].step()
        opt_ranker.step()
        opt_propensity.step()
        # for i in range(len(self.propensity_para)):
        #     if self.propensity_para[i].grad != None:
        #         self.propensity_para[i].data = self.propensity_para[i].data - self.propensity_learning_rate * self.propensity_para[i].grad
        #     #self.propensity_para[i] = self.propensity_para[i] + self.propensity_learning_rate * self.propensity_para[i].grad
        # for i in range(len(self.propensity_para)):
        #     if self.propensity_para[i].grad != None:
        #         self.propensity_para[i].grad.zero_()
        # total_norm = 0
        #
        # for p in denoise_params:
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # for p in ranking_model_params:
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # self.norm = total_norm

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        # Build model
        self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.model.train()
        self.IOBM_model.train()
        self.create_input_feed(input_feed, self.rank_list_size)
        train_output = self.ranking_model(self.model,
            self.rank_list_size)


        input_feature_list = self.get_input_feature_list(np.transpose(self.docid_inputs[:self.rank_list_size]))
        # print(len(input_feature_list))
        lector_features = torch.stack(input_feature_list, dim=0).to(device=self.cuda)
        # print(lector_features.shape)

        # train_output = torch.nan_to_num(train_output_raw)  # the output of the ranking model may contain nan


        # self.propensity_model.train()
        # propensity_labels = torch.transpose(self.labels,0,1)

        # self.propensity = self.propensity_model()

        # print(self.propensity_para)
        # self.propensity_parameter = []
        # for i in range(len(self.propensity_para)):
        #     self.propensity_parameter.append(torch.sigmoid(self.propensity_para[i]))
        #print(self.propensity_parameter)
        # labels = torch.unbind(self.labels, 0)

        # self.prop = [torch.cat(self.propensity_parameter) for _ in range(len(self.labels))]
        # self.propensity = torch.stack(self.prop, 0)
        # print(self.propensity)
        # print(self.propensity.shape)

        # propensity_values = torch.squeeze(self.IOBM_model(), dim=0)
        #
        # positions = [torch.tensor(input_feed["positions"][i]).to(device=self.cuda) for i in
        #              range(len(input_feed["positions"]))]
        # propensities = []
        # for i in range(len(positions)):
        #     propensities.append(torch.gather(propensity_values, 0, positions[i]))
        #
        # self.propensity = torch.stack(propensities, dim=0)

        batch_size = len(self.labels)

        self.propensity = self.IOBM_model(self.labels, lector_features)
        # .repeat(batch_size, 1)


        # with torch.no_grad():
        #     self.propensity_weights = self.get_normalized_weights(
        #         self.logits_to_prob(self.propensity))

        with torch.no_grad():
            self.propensity_weights = torch.ones_like(self.propensity).to(device=self.cuda) / torch.sigmoid(self.propensity)

        self.rank_loss = self.loss_func(
            train_output, self.labels, self.propensity_weights)
        # pw_list = torch.unbind(
        #     self.propensity_weights,
        #     dim=1)  # Compute propensity weights
        # for i in range(len(pw_list)):
        #     self.create_summary('Inverse Propensity weights %d' % i,
        #                         'Inverse Propensity weights %d at global step %d' % (i, self.global_step),
        #                         torch.mean(pw_list[i]), True)
        #
        # self.create_summary('Rank Loss', 'Rank Loss at global step %d' % self.global_step, torch.mean(self.rank_loss),
        #                     True)

        # Compute examination loss
        with torch.no_grad():
            self.relevance_weights = self.get_normalized_weights(
                self.logits_to_prob(train_output))

        self.exam_loss = self.loss_func(
            self.propensity,
            self.labels,
            self.relevance_weights)
        # rw_list = torch.unbind(
        #     self.relevance_weights,
        #     dim=1)  # Compute propensity weights
        # for i in range(len(rw_list)):
        #     self.create_summary('Relevance weights %d' % i,
        #                         'Relevance weights %d at global step %d' %(i, self.global_step),
        #                         torch.mean(rw_list[i]), True)
        #
        # self.create_summary('Exam Loss', 'Exam Loss at global step %d' % self.global_step, torch.mean(self.exam_loss),
        #                     True)

        # Gradients and SGD update operation for training the model.
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss
        self.separate_gradient_update()

        # self.create_summary('Gradient Norm', 'Gradient Norm at global step %d' % self.global_step, self.norm, True)
        # self.create_summary('Learning Rate', 'Learning_rate at global step %d' % self.global_step, self.learning_rate,
        #                     True)
        # self.create_summary( 'Final Loss', 'Final Loss at global step %d' % self.global_step, self.loss,
        #                     True)

        self.clip_grad_value(self.labels, clip_value_min=0, clip_value_max=1)
        # pad_removed_train_output = self.remove_padding_for_metric_eval(
        #     self.docid_inputs, train_output)
        # for metric in self.exp_settings['metrics']:
        #     for topn in self.exp_settings['metrics_topn']:
        #         list_weights = torch.mean(
        #             self.propensity_weights * self.labels, dim=1, keepdim=True)
        #         metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
        #             self.labels, pad_removed_train_output, None)
        #         self.create_summary('%s_%d' % (metric, topn),
        #                             '%s_%d at global step %d' % (metric, topn, self.global_step), metric_value, True)
        #         weighted_metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
        #             self.labels, pad_removed_train_output, list_weights)
        #         self.create_summary('Weighted_%s_%d' % (metric, topn),
        #                             'Weighted_%s_%d at global step %d' % (metric, topn, self.global_step),
        #                             weighted_metric_value, True)
        # loss, no outputs, summary.
        # print(" Loss %f at Global Step %d: " % (self.loss.item(),self.global_step))
        print(" Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        self.train_summary['rank_loss'] = self.rank_loss.item()
        self.train_summary['exam_loss'] = self.exam_loss.item()
        self.train_summary['loss'] = self.loss.item()
        self.global_step+=1
        return self.loss, None, self.train_summary

    def validation(self, input_feed, is_online_simulation=False):
        self.model.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)
        with torch.no_grad():
            self.output = self.ranking_model(self.model,
                                             self.max_candidate_num)
        if not is_online_simulation:
            pad_removed_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, self.output)
            # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
            for metric in self.exp_settings['metrics']:
                topn = self.exp_settings['metrics_topn']
                metric_values = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(self.labels, pad_removed_output, None)
                for topn, metric_value in zip(topn, metric_values):
                    self.create_summary('%s_%d' % (metric, topn),
                                        '%s_%d' % (metric, topn), metric_value.item(), False)
        return None, self.output, self.eval_summary # no loss, outputs, summary.

    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = torch.unbind(
            propensity, dim=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = torch.stack(pw_list, dim=1)
        if self.hparams.max_propensity_weight > 0:
            self.clip_grad_value(propensity_weights,clip_value_min=0,
                clip_value_max=self.hparams.max_propensity_weight)
        return propensity_weights

    def clip_grad_value(self, parameters, clip_value_min, clip_value_max) -> None:
        r"""Clips gradient of an iterable of parameters at specified value.

        Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            clip_value (float or int): maximum allowed value of the gradients.
                The gradients are clipped in the range
                :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        clip_value_min = float(clip_value_min)
        clip_value_max = float(clip_value_max)
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.clamp_(min=clip_value_min, max=clip_value_max)