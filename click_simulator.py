import os
import sys
import random
import json
import math

class ClickModel:
    def __init__(self, neg_click_prob=0.0, pos_click_prob=1.0,
                 relevance_grading_num=4, eta=1.0, epsilon=0.1):
        self.exam_prob = None
        # self.setExamProb(eta)
        self.eta = eta
        self.epsilon = epsilon
        self.setClickProb(
            neg_click_prob,
            pos_click_prob,
            relevance_grading_num)

    @property
    def model_name(self):
        return 'click_model'

    # Serialize model into a json.
    def getModelJson(self):
        desc = {
            'model_name': self.model_name,
            'eta': self.eta,
            'click_prob': self.click_prob,
            'exam_prob': self.exam_prob
        }
        return desc

    # Generate noisy click probability based on relevance grading number
    # Inspired by ERR
    def setClickProb(self, neg_click_prob, pos_click_prob,
                     relevance_grading_num):
        b = (pos_click_prob - neg_click_prob) / \
            (pow(2, relevance_grading_num) - 1)
        a = neg_click_prob - b
        self.click_prob = [
            a + pow(2, i) * b for i in range(relevance_grading_num + 1)]

    # Set the examination probability for the click model.
    def setExamProb(self, eta):
        self.eta = eta
        return

    # Sample clicks for a list
    def sampleClicksForOneList(self, label_list):
        return None

class PositionBiasedModel(ClickModel):

    @property
    def model_name(self):
        return 'position_biased_model'

    def setExamProb(self, eta, exam_prob=None):
        self.eta = eta
        if exam_prob is None:
            self.original_exam_prob = [1.0/1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0, 1.0/5.0,
                                       1.0/6.0, 1.0/7.0, 1.0/8.0, 1.0/9.0, 1.0/10.0]
            self.exam_prob = [pow(x, eta) for x in self.original_exam_prob]
        else:
            self.exam_prob = []
            for i in exam_prob:
                self.exam_prob.append(i)

    def sampleClicksForOneList(self, label_list):
        click_list = []
        for rank in range(len(label_list)):
            click = self.sampleClick(rank, label_list[rank])
            click_list.append(click)
        return click_list

    def sampleClick(self, rank, relevance_label):
        if not relevance_label == int(relevance_label):
            print('RELEVANCE LABEL MUST BE INTEGER!')
        relevance_label = int(relevance_label) if relevance_label > 0 else 0
        exam_p = self.getExamProb(rank)
        click_p = self.click_prob[relevance_label if relevance_label < len(
            self.click_prob) else -1]
        if random.random() < self.epsilon:
            click_p = 1.0
        click = 1.0 if random.random() < exam_p * click_p else 0.0
        return click

    def getExamProb(self, rank):
        return self.exam_prob[rank if rank < len(self.exam_prob) else -1]


class DependentClickModel(ClickModel):

    @property
    def model_name(self):
        return 'dependent_click_model'

    def setExamProb(self, eta):
        self.exam_prob = [1.0 for x in range(10)]

    def setContProb(self, beta, eta, cont_prob=None):
        self.beta = beta
        self.eta = eta
        if cont_prob is None:
            self.cont_prob = [self.beta * pow(1.0 / j, self.eta) for j in range(1, 11)]
        else:
            self.cont_prob = []
            for i in cont_prob:
                self.cont_prob.append(i)

    def sampleClicksForOneList(self, label_list):
        click_list = []
        abandoned = False
        for rank in range(len(label_list)):
            if not abandoned:
                click, cont_p = self.sampleClick(rank, label_list[rank])
                if int(click) == 1:
                    abandoned = (random.random() < 1 - cont_p)
                click_list.append(click)
            else:
                for i in range(rank, len(label_list)):
                    click_list.append(0.0)
                break
        return click_list

    def sampleClick(self, rank, relevance_label):
        if not relevance_label == int(relevance_label):
            print('RELEVANCE LABEL MUST BE INTEGER!')
        relevance_label = int(relevance_label) if relevance_label > 0 else 0
        cont_p = self.getContProb(rank)
        click_p = self.click_prob[relevance_label if relevance_label < len(
            self.click_prob) else -1]
        if random.random() < self.epsilon:
            click_p = 1.0
        click = 1.0 if random.random() < exam_p * click_p else 0.0
        return click, cont_p

    def getExamProb(self, rank):
        return self.exam_prob[rank if rank < len(self.exam_prob) else -1]

    def getContProb(self, rank):
        return self.cont_prob[rank if rank < len(self.cont_prob) else -1]


class ComparisionBasedClickModel(ClickModel):
    @property
    def model_name(self):
        return 'comparision_based_click_model'

    def setExamProb(self, eta, exam_prob=None):
        self.eta = eta
        if exam_prob is None:
            self.original_exam_prob = [1.0 / 1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0,
                                       1.0 / 6.0, 1.0 / 7.0, 1.0 / 8.0, 1.0 / 9.0, 1.0 / 10.0]
            self.exam_prob = [4 * pow(x, eta) for x in self.original_exam_prob]
        else:
            self.exam_prob = []
            for i in exam_prob:
                self.exam_prob.append(i)

    # def setG(self, beta, g=None):
        # self.beta = beta
        # if g is None:
        #     self.original_g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        #     self.g = [2 * pow(x, beta) for x in self.original_g]
        # else:
        #     self.g = []
        #     for i in g:
        #         self.g.append(i)
    def setG(self, g):
        self.g = g

    def setSatProb(self, w):
        self.sat_prob = [w * x for x in self.click_prob]


    def sampleClicksForOneList(self, label_list):
        click_list = [0 for _ in range(len(label_list))]
        sat_flag = 0
        pos = 0
        while sat_flag==0 and pos<(len(label_list)-1):
            r = random.random()
            p0 = math.exp(label_list[pos] + self.getExamProb(pos) - click_list[pos]* 4)
            p1 = math.exp(label_list[pos+1] + self.getExamProb(pos+1) - click_list[pos+1]* 4)
            # g_exp = math.exp(self.g[pos])
            g_exp = math.exp(self.g)
            z = g_exp + p0 + p1
            if r < p0/z:
                click_list[pos] = 1
                if random.random() < self.sat_prob[label_list[pos]]:
                    sat_flag = 1
            elif r < (p0+p1)/z:
                click_list[pos+1] = 1
                if random.random() < self.sat_prob[label_list[pos+1]]:
                    sat_flag = 1
            else:
                pos = pos + 1

        return click_list

    def getExamProb(self, rank):
        return self.exam_prob[rank if rank < len(self.exam_prob) else -1]