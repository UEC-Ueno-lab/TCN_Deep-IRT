# Code reused from https://github.com/jennyzhang0215/DKVMN.git
import numpy as np
import math
import numpy as np
import torch
from torch.utils import data
import time
import torch
# from utils import open_json, dump_json

class PID_DATA(object):
    def __init__(self, n_question,  n_subject, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question
        self.n_subject = n_subject
    # data format
    # id, true_student_id
    # pid1, pid2, ...
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        p_data = []
        a_data = []
        qa_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 4 == 0:
                student_id = lineID//4
            if lineID % 4 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q)-1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            if lineID % 4 == 2:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]

            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    problem_sequence = []
                    answer_sequence = []
                    qa_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            # Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            Xindex = int(P[i]) + int(A[i]) * self.n_subject
                            question_sequence.append(int(Q[i]))
                            problem_sequence.append(int(P[i]))
                            qa_sequence.append(Xindex)
                            answer_sequence.append(int(A[i]))
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    p_data.append(problem_sequence)
                    qa_data.append(qa_sequence)
                    a_data.append(answer_sequence)
                    

        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        a_dataArray = np.zeros((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat

        
        return q_dataArray, p_dataArray, qa_dataArray, a_dataArray


class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen
    # data format
    # id, true_student_id
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        a_data = []
        idx_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 0:
                student_id = lineID//3
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q)-1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    qa_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            # answer_sequence.append(Xindex)
                            answer_sequence.append(A[i])
                            qa_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    a_data.append(answer_sequence)
                    qa_data.append(qa_sequence)
                    idx_data.append(student_id)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        a_dataArray = np.zeros((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_dataArray, q_dataArray, qa_dataArray, a_dataArray

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, q_data, pid, qa_data, a_data, mask_a, mask_skill):
        'Initialization'
        #QuestionId,UserId,AnswerId,IsCorrect,CorrectAnswer,AnswerValue
        self.q_data = q_data
        self.pid = pid
        self.qa_data = qa_data
        self.a_data = a_data
        self.mask_a = mask_a
        self.mask_skill = mask_skill

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.a_data)

    def __getitem__(self, index):
        'Generates one sample of data'
        out = {
            "q_ids": self.q_data[index],
            "subject_ids": self.pid[index],
            "qa": self.qa_data[index],
            "labels": self.a_data[index],
            "mask": self.mask_a[index],
            "subject_mask": self.mask_skill[index]
        }
        # return self.q_data[index],self.pid[index],self.a_data[index],self.mask_a[index],self.mask_skill[index]
        return out


class my_collate(object):
    def __init__(self):
        pass

    def __call__(self, batch_raw):
        #{'subject_ids': subject_ids, 'q_ids': q_ids, 'correct_ans': correct_ans,'ans': ans, 'labels': labels}
        batch = batch_raw
        L = [len(d['q_ids']) for d in batch]
        T, B = max(L), len(L)
        n_list = [d["q_ids"] for d in batch]

        q_list = [torch.LongTensor(d["q_ids"]) for d in batch]
        q_ids = torch.stack(q_list, 1).squeeze()

        pid_list = [torch.LongTensor(d["subject_ids"]) for d in batch]
        pid = torch.stack(pid_list, 1).squeeze()

        qa_list = [torch.LongTensor(d["qa"]) for d in batch]
        qa = torch.stack(qa_list, 1).squeeze()

        a_list = [torch.LongTensor(d["labels"]) for d in batch]
        a_data = torch.stack(a_list, 1).squeeze()

        mask_a_list = [torch.LongTensor(d["mask"]) for d in batch]
        mask_a = torch.stack(mask_a_list, 1).squeeze()

        mask_skill_list = [torch.LongTensor(d["subject_mask"]) for d in batch]
        mask_skill = torch.stack(mask_skill_list, 1).squeeze()

        out = {
            "q_ids": q_ids,
            "subject_ids": pid,
            "labels": a_data,
            "qa": qa,
            "mask": mask_a,
            "subject_mask": mask_skill
        }
        return out


def make_mask(que, ans, skill):
    mask_que = np.zeros_like(que)
    mask_ans = np.zeros_like(ans)
    mask_skill = np.zeros_like(skill)
    mask_que[que!=0] = 1.
    mask_ans[que!=0] = 1.
    mask_skill[que!=0] = 1.
    return mask_ans, mask_skill


class JsonDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data):
        'Initialization'
        #QuestionId,UserId,AnswerId,IsCorrect,CorrectAnswer,AnswerValue
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.data[index]


class json_collate(object):
    def __init__(self):
        pass

    def __call__(self, batch_raw):
        #{'subject_ids': subject_ids, 'q_ids': q_ids, 'correct_ans': correct_ans,'ans': ans, 'labels': labels}
        batch = batch_raw
        L = [len(d['q_id']) for d in batch]
        T, B = max(L), len(L)
        n_list = [d["q_id"] for d in batch]

        q_list = [torch.LongTensor(d["q_id"]) for d in batch]
        q_ids = torch.stack(q_list, 1).squeeze()

        pid_list = [torch.LongTensor(d["s_id"]) for d in batch]
        pid = torch.stack(pid_list, 1).squeeze()

        qa_list = [torch.LongTensor(d["qa"]) for d in batch]
        qa = torch.stack(qa_list, 1).squeeze()

        a_list = [torch.LongTensor(d["ans"]) for d in batch]
        a_data = torch.stack(a_list, 1).squeeze()

        mask_a_list = [torch.LongTensor(d["mask"]) for d in batch]
        mask_a = torch.stack(mask_a_list, 1).squeeze()

        mask_skill_list = [torch.LongTensor(d["subject_mask"]) for d in batch]
        mask_skill = torch.stack(mask_skill_list, 1).squeeze()

        out = {
            "q_ids": q_ids,
            "subject_ids": pid,
            "labels": a_data,
            "qa": qa,
            "mask": mask_a,
            "subject_mask": mask_skill
        }
        return out