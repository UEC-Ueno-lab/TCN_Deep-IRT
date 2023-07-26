from  utils import compute_auc, compute_accuracy
import argparse
from config import create_parser
from dataset import DATA, PID_DATA, make_mask
import torch
from torch import nn
import numpy as np
import os
params = create_parser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_loss, best_val_acc, best_val_auc, best_epoch = None, None, None, -1
test_accuracy, test_auc, test_loss = -1, -1, -1
avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score = -1,-1,-1,-1

def compute_l2_loss(w):
    return torch.square(w).sum()

def compute_l1_loss(w):
    return torch.abs(w).sum()

def train_model(params, model, train_loader, valid_loader, test_loader, optimizer, scheduler, epoch):
    global best_val_loss, best_val_acc, best_val_auc, best_epoch, test_accuracy, test_auc, test_loss
    global avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score
    batch_idx = 0
    model.train()
    train_loss, all_preds, all_targets = 0., [], []
    for batch in train_loader:
        optimizer.zero_grad()
        output, _, _ = model(batch)
        #
        loss_fn = nn.BCELoss(reduction='none')
        output = output.squeeze(2)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device).unsqueeze(2)
        loss = loss_fn(output, labels.float())
        loss = loss * mask.squeeze(2)
        loss = loss.mean() / torch.mean((mask).float())
        if params.model_type=="akt":
            loss += model.c_reg_loss
        
        # Compute l2 loss component
        # l2_weight = 0.00001
        # l2_parameters = []
        # for parameter in model.parameters():
        #     l2_parameters.append(parameter.view(-1))
        # l2 = l2_weight * compute_l1_loss(torch.cat(l2_parameters))
        # loss += l2
        # print("here is l2", l2)

        
        # q_ids = batch['q_ids'].numpy()
        # u_ids = batch['user_ids']
        loss.backward()
        optimizer.step()

        training_flag = batch['mask'].numpy()
        target = batch['labels'].numpy()
        output = output.detach().cpu()

        all_preds.append(output[training_flag == 1])
        all_targets.append(target[training_flag == 1])
        train_loss += float(loss.detach().cpu().numpy())
        batch_idx += 1

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    model.eval()
    train_auc = compute_auc(all_target, all_pred)
    train_accuracy = compute_accuracy(all_target, all_pred)
    val_res = test_model(model, valid_loader)
    val_accuracy, val_auc, val_loss = val_res["acc"], val_res["auc"], val_res["loss"]
    # scheduler.step(val_auc)

    print(f'Train Epoch {epoch} \nLoss={train_loss/batch_idx:.4f}')
    print(f'Train Auc: {train_auc:.4f} Train acc: {train_accuracy:.4f}')
    print(f'Val loss={val_loss:.4f}')
    print(f'Val Auc: {val_auc:.4f} Val acc: {val_accuracy:.4f}')

    # if best_val_acc is None or val_accuracy > best_val_acc:
    # if best_val_auc is None or val_auc > best_val_auc:
    if best_val_loss is None or val_loss < best_val_loss:
        best_val_acc = val_accuracy
        best_val_auc = val_auc
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                 },
                os.path.join(params.out_dir, params.save_name
                            )
                )
        test_res = test_model(model, test_loader)
        test_accuracy = test_res['acc']
        test_auc = test_res['auc']
        test_loss = test_res['loss']
        if "simu_item" in params.data_name:
            np.save(params.out_dir+params.save_name+".npy", np.array(test_res))

    # if not "simu_item" in params.data_name:
        print(f'Train Epoch {epoch} Best Val Acc: {best_val_acc:.4f} Test Acc: {test_accuracy:.4f} Test Auc: {test_auc:.4f}')
        print("="*10)

    model.train()

def test_model(model, data_loader):
    model.eval()
    all_preds, all_targets, all_thetas, all_masks = [], [], [], []
    test_qids, test_dist = [], []
    loader = data_loader
    batch_idx=0
    val_loss = 0
    for batch in loader:
        with torch.no_grad():
            output, theta, beta = model(batch)
        
        loss_fn = nn.BCELoss(reduction='none')
        output = output.squeeze(2)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device).unsqueeze(2)
        loss = loss_fn(output, labels.float())
        loss = loss * mask.squeeze(2)
        loss = loss.mean() / torch.mean((mask).float())
        # loss = loss+model.c_reg_loss

        target = batch['labels'].numpy()
        validation_flag = batch['mask'].numpy()
        output = output.detach().cpu()
        theta = theta.detach().cpu()

        all_thetas.append(theta[:])
        all_masks.append(validation_flag[:])
        all_preds.append(output[validation_flag == 1])
        all_targets.append(target[validation_flag == 1])
        val_loss += float(loss.detach().cpu().numpy())
        batch_idx+=1
    all_theta = np.concatenate(all_thetas, axis=1)
    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return {"acc":accuracy, "auc":auc, "loss":val_loss/batch_idx, "theta":all_theta}
