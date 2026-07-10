import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import os
from tqdm import tqdm
import argparse


layer_len = 57
head_len = 24
num_feature = 128

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="./data/features",
                    help="Directory with extracted key tensors from feature_extract.py")
parser.add_argument("--output_dir", type=str, default="classifier_ckpt/flux_reproduced",
                    help="Directory to save weight.pkl and accuracy.pkl "
                         "(loadable directly via --classifier_dir at inference time)")
args = parser.parse_args()

input_dir = args.input_dir
head_type = 'key'
timesteps = [652, 814]

os.makedirs(args.output_dir, exist_ok=True)

with open(f"{input_dir}/y_train_data.pkl", "rb") as f:
    y_train_ = pickle.load(f)
with open(f"{input_dir}/y_test_data.pkl", "rb") as f:
    y_test_ = pickle.load(f)

model_weight = torch.zeros(layer_len, head_len, num_feature + 1)
accuracy_dict = {}

for l in tqdm(range(layer_len)):
    for h in range(head_len):
        for idx, t in enumerate(timesteps):
            x_train_ = torch.load(f"{input_dir}/x_train_{head_type}_{t}_{l}_{h}.pt", weights_only=True).float().numpy()
            x_test_ = torch.load(f"{input_dir}/x_test_{head_type}_{t}_{l}_{h}.pt", weights_only=True).float().numpy()

            if idx == 0:
                x_train = x_train_
                y_train = y_train_
                x_test = x_test_
                y_test = y_test_
            else:
                x_train = np.concatenate([x_train, x_train_], axis=0)
                y_train = np.concatenate([y_train, y_train_], axis=0)
                x_test = np.concatenate([x_test, x_test_], axis=0)
                y_test = np.concatenate([y_test, y_test_], axis=0)

        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        x_train = torch.tensor(x_train)
        x_test = torch.tensor(x_test)

        y = y_train.bool()
        mu_true  = x_train[y].mean(dim=0)
        mu_false = x_train[~y].mean(dim=0)
        direction = mu_true - mu_false

        n = direction.norm() + 1e-12
        direction_norm = direction / n

        proj_vals = x_train @ direction_norm
        sigma = proj_vals.std().item()
        model_weight[l, h, :128] = direction_norm * sigma

        proj_true  = (x_train[y]  @ model_weight[l, h, :128]).mean() if y.any() else torch.tensor(0.)
        proj_false = (x_train[~y] @ model_weight[l, h, :128]).mean() if (~y).any() else torch.tensor(0.)
        model_weight[l, h, 128] = -(proj_true + proj_false) / 2

        preds_train = list(map(lambda x: 1 if x > 0 else 0,
                               (x_train @ model_weight[l, h, :128] + model_weight[l, h, 128]).float().numpy()))
        preds_test  = list(map(lambda x: 1 if x > 0 else 0,
                               (x_test  @ model_weight[l, h, :128] + model_weight[l, h, 128]).float().numpy()))
        accuracy_dict[(l, h)] = (accuracy_score(y_train, preds_train), accuracy_score(y_test, preds_test))

sorted_accuracy = sorted(accuracy_dict.items(), key=lambda kv: kv[1][1], reverse=True)

# Saved in the flat layout consumed by the inference pipeline
# (flux_osi_modules/pipeline.py loads {dir}/weight.pkl and {dir}/accuracy.pkl).
with open(f"{args.output_dir}/weight.pkl", "wb") as f:
    pickle.dump(model_weight, f)
with open(f"{args.output_dir}/accuracy.pkl", "wb") as f:
    pickle.dump(sorted_accuracy, f)

print(f"Saved classifier to {args.output_dir}/{{weight,accuracy}}.pkl")
