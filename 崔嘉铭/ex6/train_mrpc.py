# train_mrpc.py
# 目标：在 MRPC 上用 bert-base-uncased 跑通训练（可交实验6.1）
# 兼容：官方 MRPC txt (msr_paraphrase_train/test.txt) 或自定义 tsv/csv
# 依赖：torch, transformers  (可选：sklearn 用于更好评估；没有也能跑)

import os
import csv
import argparse
import random
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel


# -----------------------------
# 1) 数据集：读取 MRPC
# -----------------------------
def _try_read_mrpc_official_txt(path: str) -> List[Tuple[str, str, int]]:
    """
    读取微软 MRPC 官方格式（常见文件名：msr_paraphrase_train.txt / msr_paraphrase_test.txt）
    典型列：Quality(0/1)  #1 ID  #2 ID  sentence1  sentence2
    分隔符通常是 \t
    """
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()  # 跳过表头
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            # 尝试解析：label 在第0列，句子在倒数两列
            if len(parts) < 5:
                continue
            try:
                label = int(parts[0])
            except:
                continue
            s1 = parts[-2]
            s2 = parts[-1]
            if s1 and s2:
                rows.append((s1, s2, label))
    return rows


def _try_read_generic_table(path: str) -> List[Tuple[str, str, int]]:
    """
    读取通用 tsv/csv：需要能找到 label/sentence1/sentence2 三列
    支持列名：label, sentence1, sentence2 或者 s1, s2 等
    """
    if not os.path.exists(path):
        return []

    ext = os.path.splitext(path)[1].lower()
    delimiter = "\t" if ext in [".tsv", ".txt"] else ","

    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return []

        # 列名容错
        def pick(d: Dict[str, str], keys: List[str]) -> str:
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return ""

        for d in reader:
            label_str = pick(d, ["label", "Quality", "gold_label", "is_duplicate", "target"])
            s1 = pick(d, ["sentence1", "s1", "text1", "question1", "sent1"])
            s2 = pick(d, ["sentence2", "s2", "text2", "question2", "sent2"])
            if not (label_str and s1 and s2):
                continue
            try:
                label = int(float(label_str))
            except:
                continue
            rows.append((s1, s2, label))
    return rows


def load_mrpc(data_dir: str) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    统一入口：自动在 data_dir 下找 MRPC 文件。
    返回：train_rows, test_rows
    """
    candidates_train = [
        os.path.join(data_dir, "msr_paraphrase_train.txt"),
        os.path.join(data_dir, "train.tsv"),
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "train.txt"),
    ]
    candidates_test = [
        os.path.join(data_dir, "msr_paraphrase_test.txt"),
        os.path.join(data_dir, "test.tsv"),
        os.path.join(data_dir, "test.csv"),
        os.path.join(data_dir, "test.txt"),
    ]

    train_rows, test_rows = [], []

    for p in candidates_train:
        train_rows = _try_read_mrpc_official_txt(p)
        if train_rows:
            break
        train_rows = _try_read_generic_table(p)
        if train_rows:
            break

    for p in candidates_test:
        test_rows = _try_read_mrpc_official_txt(p)
        if test_rows:
            break
        test_rows = _try_read_generic_table(p)
        if test_rows:
            break

    if not train_rows:
        raise FileNotFoundError(
            f"在 {data_dir} 下没有找到可解析的 MRPC 训练文件。\n"
            f"请确保存在：msr_paraphrase_train.txt 或 train.tsv/train.csv"
        )
    if not test_rows:
        # test 不是必须，但建议有
        print(f"[WARN] 在 {data_dir} 下没有找到可解析的测试文件，将只训练不测试。")

    return train_rows, test_rows


class MRPCDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, str, int]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        s1, s2, label = self.rows[idx]
        return s1, s2, label


# -----------------------------
# 2) 模型：BERT + 简单分类头
# -----------------------------
class BertMRPCClassifier(nn.Module):
    def __init__(self, bert_name: str = "bert-base-uncased", dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)  # 二分类 -> 1 个 logit

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = out.pooler_output  # [B, 768]
        x = self.dropout(pooled)
        logit = self.fc(x).squeeze(-1)  # [B]
        return logit


# -----------------------------
# 3) 训练 / 评估
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()

    for batch in loader:
        input_ids, attention_mask, token_type_ids, labels = [x.to(device) for x in batch]
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = bce(logits, labels.float())
        total_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="MRPC 数据目录")
    parser.add_argument("--bert_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    # 读数据
    train_rows, test_rows = load_mrpc(args.data_dir)
    print(f"[INFO] train samples: {len(train_rows)}")
    print(f"[INFO] test  samples: {len(test_rows)}")

    train_ds = MRPCDataset(train_rows)
    test_ds = MRPCDataset(test_rows) if test_rows else None

    tokenizer = BertTokenizer.from_pretrained(args.bert_name)

    def collate_fn(batch):
        s1_list, s2_list, y_list = zip(*batch)
        enc = tokenizer(
            list(s1_list),
            list(s2_list),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
        )
        labels = torch.tensor(y_list, dtype=torch.long)
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(enc["input_ids"])
        return enc["input_ids"], enc["attention_mask"], token_type_ids, labels

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    model = BertMRPCClassifier(args.bert_name).to(device)

    # 训练设置：logits + BCEWithLogitsLoss（二分类常用写法）
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids, attention_mask, token_type_ids, labels = [x.to(device) for x in batch]

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = bce(logits, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if step % 20 == 0 or step == 1:
                avg_loss = total_loss / max(total, 1)
                acc = correct / max(total, 1)
                print(f"Epoch {epoch} Step {step:04d} | loss={avg_loss:.4f} | acc={acc:.4f}")

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        print(f"[EPOCH {epoch}] train loss={train_loss:.4f} acc={train_acc:.4f}")

        if test_loader is not None:
            test_loss, test_acc = evaluate(model, test_loader, device)
            print(f"[EPOCH {epoch}]  test loss={test_loss:.4f} acc={test_acc:.4f}")

    print("[OK] Training finished.")


if __name__ == "__main__":
    main()
