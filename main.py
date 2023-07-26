import numpy as np
from numpy.core.arrayprint import format_float_scientific
import torch
import gzip
import json
import time
import pandas as pd
import random
import yaml
from utils import open_json, dump_json, compute_auc, compute_accuracy,compute_kappa_f1_score, result_save
from dataset import Dataset, my_collate
from model import DKVMN, DeepTsutsumi, ConvMem01, ConvMem02, ConvMem03, ConvHyper, ConvMemhyper01, ConvMemDouble, ConvMemTune
from model2 import ConvMemItem01, ConvMultiLayer01, ConvTheta01, SimpleTCN
from model_akt import AttentionModel
from model_window import Conv_L8, Conv_L7, Conv_L6, Conv_L5, Conv_L4, Conv_L3, Conv_L2, Conv_L1
from train import train_model
import os
from config import create_parser
from dataset import DATA, PID_DATA, make_mask

params = create_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    params = create_parser()
    params = create_parser()
    # make output directory
    if not os.path.exists(params.out_dir):
        os.makedirs(params.out_dir)

    train_data_path = params.input_path+params.data_name+"_train"+str(params.fold)+".csv"
    valid_data_path = params.input_path+params.data_name+"_valid"+str(params.fold)+".csv"
    test_data_path = params.input_path+params.data_name+"_test"+str(params.fold)+".csv"
    
    if params.data_name in {"junyi", "statics", "assist2015", "assist2017", "assist2009_updated"}:
        dat = DATA(n_question=params.n_question,
                    seqlen=params.seqlen, separate_char=',')
    else:
        dat = PID_DATA(
            n_question=params.n_question,
            n_subject=params.n_subject,
            seqlen=params.seqlen, separate_char=',')

    train_q_data, train_pid, train_qa_data, train_a_data = dat.load_data(train_data_path)
    train_mask_a, train_mask_skill = make_mask(train_q_data, train_a_data, train_pid)
    valid_q_data, valid_pid, valid_qa_data, valid_a_data = dat.load_data(valid_data_path)
    valid_mask_a, valid_mask_skill = make_mask(valid_q_data, valid_a_data, valid_pid)
    print(valid_data_path)
    print(valid_q_data.shape)

    # train_dataset, val_dataset, test_dataset = Dataset(train_data), Dataset(val_data), Dataset(test_data)
    train_dataset = Dataset(
        train_q_data, train_pid, train_qa_data, train_a_data, train_mask_a, train_mask_skill
    )
    valid_dataset = Dataset(
        valid_q_data, valid_pid, valid_qa_data, valid_a_data,
        valid_mask_a, valid_mask_skill
    )
    
    collate_fn = my_collate()
    num_workers = 2
    bs = params.batch_size


    if "simu_item" in params.data_name:
        test_dataset = Dataset(
            valid_q_data, valid_pid, valid_qa_data, valid_a_data,
            valid_mask_a, valid_mask_skill
        )
        test_loader = torch.utils.data.DataLoader(
        test_dataset,  collate_fn=collate_fn, batch_size=128,
        num_workers=num_workers, shuffle=False, drop_last=False)
    else:
        test_q_data, test_pid, test_qa_data, test_a_data = dat.load_data(test_data_path)
        test_mask_a, test_mask_skill = make_mask(test_q_data, test_a_data, test_pid)
        test_dataset = Dataset(
             test_q_data, test_pid, test_qa_data, test_a_data,
            test_mask_a, test_mask_skill
        )
        test_loader = torch.utils.data.DataLoader(
        test_dataset,  collate_fn=collate_fn, batch_size=bs,
        num_workers=num_workers, shuffle=False, drop_last=False)

    print(bs)
    print(f"data:{params.dataset}, fold:{params.fold}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,  collate_fn=collate_fn, batch_size=bs,
        num_workers=num_workers, shuffle=True, drop_last=False)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,  collate_fn=collate_fn, batch_size=bs,
        num_workers=num_workers, shuffle=False, drop_last=False)

        
    if params.model_type=="dkvmn":
        model = DKVMN(
            n_question=params.n_question,
            n_subject=params.n_subject,
            s_dim=params.s_dim,
            hidden_dim=params.hidden_dim,
            q_dim=params.q_dim,
            dropout=0.25).to(device)
    elif params.model_type=="window_exp":
        m_list = [Conv_L1, Conv_L2, Conv_L3, Conv_L4, Conv_L5, Conv_L6, Conv_L7, Conv_L8]
        model = m_list[params.conv_nlayer-1](
            n_question=params.n_question,
            n_subject=params.n_subject,
            s_dim=params.s_dim,
            hidden_dim=params.hidden_dim,
            q_dim=params.q_dim,
            dropout=0.25).to(device)
    elif params.model_type=="tsutsumi":
        model = DeepTsutsumi(
            n_question=params.n_question,
            n_subject=params.n_subject,
            s_dim=params.s_dim,
            hidden_dim=params.hidden_dim,
            q_dim=params.q_dim,
            dropout=0.25).to(device)
    elif params.model_type=="convmem01":
        model = ConvMem01(
            n_question=params.n_question,
            n_subject=params.n_subject,
            s_dim=params.s_dim,
            hidden_dim=params.hidden_dim,
            q_dim=params.q_dim,
            dropout=0.25).to(device)
    elif params.model_type=="convmem02":
        model = ConvMem02(
            n_question=params.n_question,
            n_subject=params.n_subject,
            s_dim=params.s_dim,
            hidden_dim=params.hidden_dim,
            q_dim=params.q_dim,
            dropout=0.25).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=0.8)

    
    for epoch in range(params.epochs):
        train_model(params, model, train_loader, valid_loader, test_loader, optimizer, scheduler, epoch)
    
    from train import test_auc, test_accuracy, test_loss
    with open(params.out_dir+params.save_name+"result.log", mode='w') as f:
                s = f"test_auc, test_acc, test_loss = {test_auc}, {test_accuracy}, {test_loss}\n"
                f.write(s)
    result_save(params.out_dir+params.save_name+"result.json", params.fold, test_auc, test_accuracy, test_loss)
    print("done")
