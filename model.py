import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import create_parser
params = create_parser()

class MEMORY(nn.Module):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, qa_embed_dim):
        super().__init__()
        self.memory_size = memory_size
        self.qa_embed_dim = qa_embed_dim
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.erase_net = nn.Sequential(
            nn.Linear(self.qa_embed_dim,
                      self.memory_value_state_dim), nn.Sigmoid()
        )
        self.add_net = nn.Sequential(
            nn.Linear(self.qa_embed_dim,
                      self.memory_value_state_dim), nn.Tanh()
        )

        # hypernet
        self.erase_mv_net = nn.Sequential(
            nn.Linear(
                    self.memory_value_state_dim*self.memory_size,
                    self.memory_value_state_dim,
                    ), nn.Sigmoid()
        )
        self.zt_add_net = nn.Sequential(
            nn.Linear(self.memory_value_state_dim,
                      self.memory_value_state_dim), nn.Tanh()
        )
        self.add_mv_net = nn.Sequential(
            nn.Linear(
                    self.memory_value_state_dim*self.memory_size,
                    self.memory_value_state_dim,
                    ), nn.Tanh()
        )
        self.content_m_round0 = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(self.memory_size*self.memory_value_state_dim, self.qa_embed_dim),
            nn.Sigmoid())
        self.memory_m_round1 = nn.Sequential(
            nn.Linear(self.qa_embed_dim, self.memory_size*self.memory_value_state_dim),
            nn.Sigmoid())
        # self.content_m_round2 = nn.Sequential(
        #     nn.Flatten(), 
        #     nn.Linear(self.memory_size*self.memory_value_state_dim, self.memory_size),
        #     nn.Sigmoid())
        # self.memory_m_round3 = nn.Sequential(
        #     nn.Linear(self.memory_size, self.memory_size*self.memory_value_state_dim),
        #     nn.Sigmoid())
        # self.content_m_round4 = nn.Sequential(
        #     nn.Flatten(), 
        #     nn.Linear(self.memory_size*self.memory_value_state_dim, self.memory_size),
        #     nn.Sigmoid())

        self.zt_signal = nn.Sequential(
            nn.Linear(self.qa_embed_dim,
                      self.memory_value_state_dim)
        )

        self.zt_signal_Mv = nn.Sequential(
            nn.Linear(self.memory_value_state_dim*self.memory_size, self.memory_value_state_dim)
        )


    def attention(self, control_input, memory):
        similarity_score = torch.matmul(
            control_input, torch.t(memory))  # BS, MS
        m = nn.LogSoftmax(dim=1)
        log_correlation_weight = m(similarity_score)
        return log_correlation_weight.exp()

    def read(self, memory_value, read_weight):
        read_weight = torch.reshape(
            read_weight, shape=(-1, 1, self.memory_size))
        read_content = torch.matmul(read_weight, memory_value)
        read_content = torch.reshape(read_content,  
                                     shape=(-1, self.memory_value_state_dim))
        return read_content  
    
    def conv_read(self, memory_value, read_weight):
        read_weight = torch.reshape(
            read_weight, shape=(params.seqlen, -1, 1, self.memory_size))
        read_content = torch.matmul(read_weight, memory_value)
        read_content = torch.reshape(read_content, 
                                     shape=(params.seqlen, -1, self.memory_value_state_dim))
        return read_content

    def write(self, control_input, memory, write_weight):

        erase_signal = self.erase_net(control_input)
        add_signal = self.add_net(control_input)

        erase_mult = 1 - torch.matmul(torch.reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                      torch.reshape(erase_signal, shape=(-1, 1, self.memory_value_state_dim)))

        aggre_add_signal = torch.matmul(torch.reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                        torch.reshape(add_signal, shape=(-1, 1, self.memory_value_state_dim)))
        new_memory = memory * erase_mult + aggre_add_signal
        return new_memory

    def hyper_write(self, control_input, memory, write_weight):
        memory_m_0 = memory
        content_m_0 = control_input * self.content_m_round0(memory_m_0)
        memory_m_1 = memory_m_0 * torch.reshape(self.memory_m_round1(content_m_0), shape=(-1, self.memory_size, self.memory_value_state_dim))
        # content_m_1 = content_m_0 * self.content_m_round2(memory_m_1)
        # memory_m_2 = memory_m_1 * torch.reshape(self.memory_m_round3(content_m_1), shape=(-1, self.memory_size, self.memory_value_state_dim))
        # content_m_2 = content_m_1 * self.content_m_round4(memory_m_2)

        control_input = content_m_0
        memory_pre = memory_m_1

        erase_signal = self.erase_net(control_input)
        
        erase_signal_mv = self.erase_mv_net(nn.Flatten()(memory_pre))
        erase_signal = nn.Sigmoid()(erase_signal  +  erase_signal_mv)

        
        zt_signal = self.zt_signal(control_input)
        zt_signal_mv = self.zt_signal_Mv(nn.Flatten()(memory_pre))
        
        zt_signal = nn.Sigmoid()(zt_signal  + zt_signal_mv)

        add_signal = self.zt_add_net(zt_signal)
        add_signal_mv = self.add_mv_net(nn.Flatten()(memory_pre))

        add_signal = nn.Tanh()(add_signal  + add_signal_mv)
        erase_mult = 1 - torch.matmul(torch.reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                      torch.reshape(erase_signal, shape=(-1, 1, self.memory_value_state_dim)))

        aggre_add_signal = torch.matmul(torch.reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                        torch.reshape(add_signal, shape=(-1, 1, self.memory_value_state_dim)))
        new_memory = memory_pre * erase_mult + aggre_add_signal

        return new_memory

        

class ConvMem01(nn.Module):
    def __init__(self, n_question, n_subject, hidden_dim, q_dim, dropout, s_dim, memory_size=50):
        super().__init__()
        self.n_question = n_question
        self.n_subject = n_subject
        self.memory_size = memory_size
        self.memory_key_state_dim = s_dim
        self.memory_value_state_dim = hidden_dim
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.s_embeddings = nn.Embedding(n_subject, s_dim)
        self.qa_dim = q_dim*2
        self.qa_embeddings = nn.Embedding(n_question*2+1, self.qa_dim)
        self.dropout = nn.Dropout(dropout)


        # Initialize Memory
        self.init_memory_key = nn.Parameter(
            0.01*torch.randn(self.memory_size, self.memory_key_state_dim))
        self.init_memory_value = nn.Parameter(
            0.01 * torch.randn(self.memory_size, self.memory_value_state_dim))

        if params.skill_item:
            qa_embed_dim=self.qa_dim
        else:
            qa_embed_dim=self.qa_dim
        self.memory = MEMORY(memory_size=self.memory_size, memory_key_state_dim=self.memory_key_state_dim,
                             memory_value_state_dim=self.memory_value_state_dim, qa_embed_dim=qa_embed_dim)

        self.pred_in_feature = hidden_dim + s_dim + q_dim
        self.layers = nn.Sequential(
            nn.Linear(self.pred_in_feature,
                      self.pred_in_feature), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.pred_in_feature, self.pred_in_feature), nn.GELU(), nn.Dropout(dropout))

        self.theta_nn = nn.Sequential(
            nn.Linear(self.memory_value_state_dim,
                      self.memory_size), nn.GELU(), nn.Dropout(dropout)
                      )
        self.item_nn = nn.Sequential(
            nn.Linear(q_dim,
                      q_dim), nn.GELU(), nn.Dropout(dropout)
                      )
        self.subject_nn = nn.Sequential(
            nn.Linear(q_dim,
                      q_dim), nn.GELU(), nn.Dropout(dropout)
                      )
        self.item_subject_out = nn.Sequential(
            nn.Linear(q_dim,
                      1), nn.Tanh(), nn.Dropout(dropout)
                      )
        # self.theta_out = nn.Sequential(
        #     nn.Linear(self.memory_size,
        #               1), nn.Tanh(), nn.Dropout(dropout), nn.LayerNorm(1)
        #               )
        self.theta_out = nn.Sequential(
            nn.Linear(self.memory_size,
                      1), nn.Tanh(), nn.Dropout(dropout)
                      )

        kernel_size1 = 2
        dilation_size1 = 1
        self.padding1 = (kernel_size1-1) * dilation_size1
        self.conv1 = nn.Sequential(
            weight_norm(torch.nn.Conv1d(
                self.memory_value_state_dim,
                self.memory_value_state_dim, kernel_size1,
                stride=1, 
                padding=self.padding1,
                dilation=dilation_size1, groups=1, bias=True, padding_mode='zeros')
            ), nn.GELU(), nn.Dropout(dropout)
            )

        dilation_size2 = 2
        self.padding2 = (kernel_size1-1) * dilation_size2
        self.conv2 = nn.Sequential(
            weight_norm(torch.nn.Conv1d(
                self.memory_value_state_dim,
                self.memory_value_state_dim, kernel_size1,
                stride=1, 
                padding=self.padding2,
                dilation=dilation_size2, groups=1, bias=True, padding_mode='zeros')),
            nn.GELU(), nn.Dropout(dropout)
            )
        dilation_size3 = 4
        kernel_size3 = 2
        self.padding3 = (kernel_size3-1) * dilation_size3
        self.conv3 = nn.Sequential(
            weight_norm(torch.nn.Conv1d(
            self.memory_value_state_dim,
            self.memory_value_state_dim, kernel_size3,
            stride=1, 
            padding=self.padding3,
            dilation=dilation_size3, groups=1, bias=True, padding_mode='zeros')),
            nn.GELU(), nn.Dropout(dropout)
            )
        dilation_size4 = 8
        kernel_size4 = 2
        self.padding4 = (kernel_size4-1) * dilation_size4
        self.conv4 = nn.Sequential(
            weight_norm(torch.nn.Conv1d(
            self.memory_value_state_dim,
            self.memory_value_state_dim, kernel_size4,
            stride=1, 
            padding=self.padding4,
            dilation=dilation_size4, groups=1, bias=True, padding_mode='zeros')),
            nn.GELU(), nn.Dropout(dropout)
            )
        dilation_size5 = 16
        kernel_size5 = 2
        self.padding5 = (kernel_size5-1) * dilation_size5
        self.conv5 = nn.Sequential(
            weight_norm(torch.nn.Conv1d(
            self.memory_value_state_dim,
            self.memory_value_state_dim, kernel_size5,
            stride=1, 
            padding=self.padding5,
            dilation=dilation_size5, groups=1, bias=True, padding_mode='zeros')),
            nn.GELU(), nn.Dropout(dropout)
            )
        dilation_size6 = 32
        kernel_size6 = 2
        self.padding6 = (kernel_size6-1) * dilation_size6
        self.conv6 = nn.Sequential(
            weight_norm(torch.nn.Conv1d(
            self.memory_value_state_dim,
            self.memory_value_state_dim, kernel_size6,
            stride=1, 
            padding=self.padding6,
            dilation=dilation_size6, groups=1, bias=True, padding_mode='zeros')),
            nn.GELU(), nn.Dropout(dropout)
            )
        dilation_size7 = 64
        kernel_size7 = 2
        self.padding7 = (kernel_size7-1) * dilation_size7
        self.conv7 = nn.Sequential(
            weight_norm(torch.nn.Conv1d(
            self.memory_value_state_dim,
            self.memory_value_state_dim, kernel_size7,
            stride=1, 
            padding=self.padding7,
            dilation=dilation_size7, groups=1, bias=True, padding_mode='zeros')),
            nn.GELU(), nn.Dropout(dropout)
            )
        dilation_size8 = 128
        kernel_size8 = 2
        self.padding8 = (kernel_size8-1) * dilation_size8
        self.conv8 = nn.Sequential(
            weight_norm(torch.nn.Conv1d(
            self.memory_value_state_dim,
            self.memory_value_state_dim, kernel_size8,
            stride=1, 
            padding=self.padding8,
            dilation=dilation_size8, groups=1, bias=True, padding_mode='zeros')),
            nn.GELU(), nn.Dropout(dropout)
            )

        self.res_conv = nn.Sequential(weight_norm(torch.nn.Conv1d(
            self.memory_value_state_dim,
            self.memory_value_state_dim, 1)), nn.GELU(), nn.Dropout(dropout)
        )
        # self.res_conv = torch.nn.Conv1d(
        #     self.memory_value_state_dim,
        #     self.memory_value_state_dim, 1)
        self.conv_nn = nn.Sequential(
            nn.Linear(self.memory_value_state_dim,
                      self.memory_size), nn.Tanh(), nn.Dropout(dropout)
                      )

        
        # self.conv_nn2 = nn.Sequential(
        #     nn.Linear(self.memory_value_state_dim,
        #               self.memory_size), nn.Tanh(), nn.Dropout(dropout)
        #               )
        self.out_nn = nn.Sequential(
            nn.Linear(2,
                      1), nn.Tanh(), nn.Dropout(dropout)
                      )
        
        

    def forward(self, batch):
        labels, qa_in = batch['labels'].to(device), batch['qa'].to(device)
        seq_len, batch_size = labels.shape[0], labels.shape[1]
        mask = batch['mask'].to(device).unsqueeze(2)

        if params.data_name == "assist2009_all":
            subj_in = batch["subject_ids"].to(device)
            subj_mask = batch["subject_mask"].to(device)
            subjects = torch.sum(
            self.s_embeddings(subj_in.to(device)) * subj_mask.to(device).unsqueeze(3), dim=2)
        else:
            subj_in = torch.unsqueeze(batch["subject_ids"], 2).to(device)
            subj_mask = torch.unsqueeze(batch["subject_mask"], 2).to(device)
            subjects = torch.sum(
                self.s_embeddings(subj_in.int().to(device)) * subj_mask.int().to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = (self.q_embeddings(
            batch['q_ids'].int().to(device)))

    
        qa_embed = self.qa_embeddings(qa_in)*mask
        if params.skill_item:
            lstm_input = [qa_embed]
        else:
            lstm_input = [qa_embed]
        lstm_input = self.dropout(torch.cat(lstm_input, dim=-1))

        qs = torch.cat([subjects], dim=-1)
        memory_value = self.init_memory_value[None, :, :].expand(
            batch_size, -1, -1)
        init_memory_key = self.init_memory_key

        mem = self.memory
        correlation_weight_l = []
        value_read_content_l = []
        for i in range(seq_len):
            # Attention
            q = qs[i]
            correlation_weight = mem.attention(q, init_memory_key)
            # Read Process
            read_content = mem.read(memory_value, correlation_weight)
            # save intermedium data
            correlation_weight_l.append(correlation_weight[None,:, :])
            value_read_content_l.append(read_content[None, :, :])
            

            # Write Process
            qa = lstm_input[i]
            # memory_value = mem.hyper_write(qa, memory_value, correlation_weight)
            memory_value = mem.write(qa, memory_value, correlation_weight)

        forward_ht = torch.cat(value_read_content_l, dim=0)
        cor_weight_ht = torch.cat(correlation_weight_l, dim=0)

        # memory conv layer
        conv_ht = forward_ht.permute(1, 2, 0)
        conv_in = self.res_conv(conv_ht)
        conv_ht = self.conv1(conv_ht)[:, :, :-self.padding1].contiguous()
        conv_ht = self.conv2(conv_ht)[:, :, :-self.padding2].contiguous()
        conv_ht = self.conv3(conv_ht)[:, :, :-self.padding3].contiguous()
        conv_ht = self.conv4(conv_ht)[:, :, :-self.padding4].contiguous()
        conv_ht = self.conv5(conv_ht)[:, :, :-self.padding5].contiguous()
        conv_ht = self.conv6(conv_ht)[:, :, :-self.padding6].contiguous()
        conv_ht = self.conv7(conv_ht)[:, :, :-self.padding7].contiguous()
        conv_ht = self.conv8(conv_ht)[:, :, :-self.padding8].contiguous()
        conv_in += conv_ht
        conv_in = conv_in.permute(2, 0, 1)
        conv_in = self.conv_nn(conv_in)
        conv_in = torch.sum(cor_weight_ht*conv_in, dim=2).unsqueeze(2)
        
        forward_ht = self.theta_nn(forward_ht)
        forward_ht = torch.sum(cor_weight_ht*forward_ht, dim=2).unsqueeze(2)


        # concat
        conv_for_ht = forward_ht+conv_in


        # question&subject net
        questions = self.item_nn(questions)
        subjects = self.subject_nn(subjects)
        
        item_subject = questions+subjects
        out_item_sub = self.item_subject_out(item_subject)

        out_theta = 3.0*conv_for_ht

        output = out_theta - out_item_sub
        m = nn.Sigmoid()
        return m(output), out_theta, out_item_sub

class DeepTsutsumi(nn.Module):
    def __init__(self, n_question, n_subject, hidden_dim, q_dim, dropout, s_dim, memory_size=50):
        super().__init__()
        self.n_question = n_question
        self.memory_size = memory_size
        self.memory_key_state_dim = s_dim
        self.memory_value_state_dim = hidden_dim
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.s_embeddings = nn.Embedding(n_subject, s_dim)
        self.qa_embeddings = nn.Embedding(n_question*2+1, q_dim)
        self.answer_embeddings = nn.Embedding(2, s_dim)
        self.label_embeddings = nn.Embedding(2, s_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize Memory
        self.init_memory_key = nn.Parameter(
            0.01*torch.randn(self.memory_size, self.memory_key_state_dim))
        self.init_memory_value = nn.Parameter(
            0.01 * torch.randn(self.memory_size, self.memory_value_state_dim))

        self.memory = MEMORY(memory_size=self.memory_size, memory_key_state_dim=self.memory_key_state_dim,
                             memory_value_state_dim=self.memory_value_state_dim, qa_embed_dim=s_dim)

        self.pred_in_feature = hidden_dim + s_dim + q_dim
        self.layers = nn.Sequential(
            nn.Linear(self.pred_in_feature,
                      self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.pred_in_feature, self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout))

        self.theta_nn = nn.Sequential(
            nn.Linear(self.memory_value_state_dim,
                      self.memory_size), nn.Tanh(), nn.Dropout(dropout)
                      )
        self.item_nn = nn.Sequential(
            nn.Linear(q_dim,
                      q_dim), nn.ReLU(), nn.Dropout(dropout)
                      )
        self.subject_nn = nn.Sequential(
            nn.Linear(q_dim,
                      q_dim), nn.ReLU(), nn.Dropout(dropout)
                      )
        self.item_subject_out = nn.Sequential(
            nn.Linear(q_dim,
                      1), nn.Tanh(), nn.Dropout(dropout)
                      )
        self.theta_out = nn.Sequential(
            nn.Linear(self.memory_size,
                      1), nn.Tanh(), nn.Dropout(dropout)
                      )

        self.output_layer = nn.Linear(self.pred_in_feature, 1)
        

    def forward(self, batch):
        labels, qa_in = batch['labels'].to(device), batch['qa'].to(device)
        seq_len, batch_size = labels.shape[0], labels.shape[1]
        mask = batch['mask'].to(device).unsqueeze(2)

        subj_in = torch.unsqueeze(batch["subject_ids"], 2)
        subj_mask = torch.unsqueeze(batch["subject_mask"], 2)
        subjects = torch.sum(
            self.s_embeddings(subj_in.int().to(device)) * subj_mask.int().to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = (self.q_embeddings(
            batch['q_ids'].int().to(device)))  # T, B,q_dim

    
        qa_embed = self.qa_embeddings(qa_in)*mask

        lstm_input = [qa_embed]
        lstm_input = torch.cat(lstm_input, dim=-1)
        qs = torch.cat([subjects], dim=-1)

        memory_value = self.init_memory_value[None, :, :].expand(
            batch_size, -1, -1)
        init_memory_key = self.init_memory_key

        mem = self.memory
        correlation_weight_l = []
        value_read_content_l = []
        for i in range(seq_len):
            # Attention
            q = qs[i]
            correlation_weight = mem.attention(q, init_memory_key)
            # Read Process
            read_content = mem.read(memory_value, correlation_weight)
            # save intermedium data
            correlation_weight_l.append(correlation_weight[None,:, :])
            value_read_content_l.append(read_content[None, :, :])
            # Write Process
            qa = lstm_input[i]
            # memory_value = mem.write(qa, memory_value, correlation_weight)
            memory_value = mem.hyper_write(qa, memory_value, correlation_weight)

        forward_ht = torch.cat(value_read_content_l, dim=0)
        cor_weight_ht = torch.cat(correlation_weight_l, dim=0)

        forward_ht = self.theta_nn(forward_ht)
        forward_ht = torch.sum(cor_weight_ht*forward_ht, dim=2).unsqueeze(2)
        questions = self.item_nn(questions)
        subjects = self.subject_nn(subjects)
    
        item_subject = questions+subjects
        out_item_sub = self.item_subject_out(item_subject)     

        out_theta = 3.0*forward_ht
        output = out_theta - out_item_sub
        m = nn.Sigmoid()
        return m(output), out_theta, out_item_sub