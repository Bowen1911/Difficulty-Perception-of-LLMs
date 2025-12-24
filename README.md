# LLMs Intrinsic Difficulty Perception

Code for paper `Probing Implicit Difficulty Perception in Large Language Models`.

## Pipeline

![Pipeline](img/framework.jpg)

## Probe Model

How to load Qwen2.5-7B-Instruct's profile model:

```python
import torch
import torch.nn as nn

class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

lieanr_model = torch.load('models/difficulty_probe_qwen2.5.pth')
```

We also provide the probe weights for DeepSeek-R1-Distill_Qwen-7B and Llama3.1-8B-Instruct, which can be loaded similarly.

We provide the probe weights of Qwen2.5-7B-Instruct separately, and it is easy to separate weights from the model. When calculating the difficulty score in attention module, it is only necessary to directly load the probe weights, for example:

```python
probe_weight = torch.load('models/difficulty_vector_qwen2.5.pth')
```

## Attention Heads Ablation

Refer to `attn_head_ablation.py`.

## Attention Head Perceptual Control

Refer to `deepmath_weighted_llm_emb.py`.

## Data Preparation of Training Probe

The DeepMath complete dataset contains 103k samples, of which we used 11.7k. In the data we use, 64% of the training set (11.7K × 0.64 = 7488), 16% of the validation set, and 20% of the test set. In addition, we also used the complete GSM8K as the test set.

We are not using the entire dataset, but we have also not deliberately selected based on sample features. Below is the entire process of our data division:

1. Load data. We selected a portion of the samples for sample balance and computational efficiency.

```python
from datasets import load_dataset

ds = load_dataset(".../deepmath")  # Local path of deepmath. Data can also directly download from huggingface, which is exactly the same. https://huggingface.co/datasets/zwhe99/DeepMath-103K

import pandas as pd

records = []

# We only selected samples with difficulty ranging from 3.0 to 9.0, in order to 
# achieve sample balance, as the number of samples with different difficulty 
# levels is not consistent.
for dt in ds['train']:
    if dt['difficulty'] in [4.5, 5.0, 4.0, 3.0, 5.5, 8.0, 6.5, 8.5, 7.0, 9.0, 3.5, 7.5, 6.0]:
        records.append({
            "difficulty": dt["difficulty"],
            "templated_question": [{"role": "user", "content": dt["question"]}],
            "final_answer": dt["final_answer"],
            "topic": dt["topic"],
            "emb_qwen2.5": None,
        })

# convert to DataFrame
df = pd.DataFrame(records)

# Each difficulty level selects the top 900 samples (to save time).
dfd = df.groupby("difficulty").head(900).reset_index(drop=True)
```

2. We obtain model representation here, store it together with the dataset.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model = AutoModelForCausalLM.from_pretrained(".../qwen2.5-7b-instruct", dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(".../qwen2.5-7b-instruct", use_fast=False)
model.eval()

with torch.no_grad():
    for idx, row in tqdm(dfd.iterrows(), total=len(dfd)):
        chat = row["templated_question"]
        inputs = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda:0")

        outputs = model(inputs, output_hidden_states=True)
        emb = outputs.hidden_states[-1][:, -1, :].squeeze().cpu().tolist()

        dfd.at[idx, "emb_qwen2.5"] = emb

dfd.to_parquet("data/statistics/deepmath_embedding.parquet", index=False)
```

3. Train test set preparation. We divide the training and validation sets, as well as the complete process of training a linear probe model.

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ---------------------------
# data preparation
# ---------------------------
X = np.stack(dfd["emb_qwen2.5"].values).astype(np.float32)
y = dfd["difficulty"].values.astype(np.float32).reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# train-test
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train-val
train_size = int(0.8 * len(X_train_full))
val_size = len(X_train_full) - train_size
X_train, X_val = torch.from_numpy(X_train_full[:train_size]), torch.from_numpy(X_train_full[train_size:])
y_train, y_val = torch.from_numpy(y_train_full[:train_size]), torch.from_numpy(y_train_full[train_size:])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# ---------------------------
# NN model
# ---------------------------
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # only a linear projection

    def forward(self, x):
        return self.linear(x)

model = RegressionNN(input_dim=3584).to('cuda')  # qwen 3584, llama 4096

# ---------------------------
# Loss function & Optimizer (with L2 regularization)
# ---------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=2e-4)  # weight_decay=L2

# ---------------------------
# train loop
# ---------------------------
epochs = 80
train_losses = []
val_losses = []

for epoch in range(epochs):
    # train
    model.train()
    epoch_train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to('cuda'), yb.to('cuda')
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * xb.size(0)
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # validation
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to('cuda'), yb.to('cuda')
            pred = model(xb)
            loss = criterion(pred, yb)
            epoch_val_loss += loss.item() * xb.size(0)
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
```
