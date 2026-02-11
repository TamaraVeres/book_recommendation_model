import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.model_supervised import get_feature_dicts

    
class ContrastivePairDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.df_books = df.drop_duplicates(subset=["book_id"]).set_index("book_id")
        self.all_book_ids = df["book_id"].unique()
        self.user_books = df.groupby("user_id")["book_id"].apply(set).to_dict()
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        user_id = row["user_id"]
        
        user_features = {
            "age_category": torch.tensor(row["age_category"], dtype=torch.long),
            "country": torch.tensor(row["country"], dtype=torch.long),
            "educated": torch.tensor(row["educated"], dtype=torch.float32),
        }
        pos_book_features = {
            "author": torch.tensor(row["book_author"], dtype=torch.long),
            "publisher": torch.tensor(row["publisher"], dtype=torch.long),
            "book_age": torch.tensor(row["book_age"], dtype=torch.float32),
        }

        interacted = self.user_books[user_id]
        while True:
            neg_book_id = np.random.choice(self.all_book_ids)
            if neg_book_id not in interacted:
                break
       
        neg_row = self.df_books.loc[neg_book_id]
        neg_book_features = {
        "author": torch.tensor(neg_row["book_author"], dtype=torch.long),
        "publisher": torch.tensor(neg_row["publisher"], dtype=torch.long),
        "book_age": torch.tensor(neg_row["book_age"], dtype=torch.float32),
       }

        return user_features, pos_book_features, neg_book_features

def _to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}

def train_unsupervised_model(user_encoder, book_encoder, dataloader, num_epochs=10, lr=0.001, margin=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_encoder.to(device)
    book_encoder.to(device)
    optimizer = optim.Adam(list(user_encoder.parameters()) + list(book_encoder.parameters()), lr=lr, weight_decay=0.0001)
    criterion = nn.MarginRankingLoss(margin=margin)
    
    for epoch in range(num_epochs):
        user_encoder.train()
        book_encoder.train()
        total_loss = 0.0
        for user_features, pos_book_features, neg_book_features in dataloader:
            user_features = _to_device(user_features, device)
            pos_book_features = _to_device(pos_book_features, device)
            neg_book_features = _to_device(neg_book_features, device)
            u = user_encoder(user_features)
            v_pos = book_encoder(pos_book_features)
            v_neg = book_encoder(neg_book_features)
            s_pos = F.cosine_similarity(u, v_pos)
            s_neg = F.cosine_similarity(u, v_neg)
            loss = criterion(s_pos, s_neg, torch.ones_like(s_pos, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} | Contrastive loss: {total_loss / len(dataloader):.4f}")
        
def evaluate_unsupervised_model(user_encoder, book_encoder, df, batch_size=256, margin=0.05):
    dataset = ContrastivePairDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    user_encoder.eval()
    book_encoder.eval()
    criterion = nn.MarginRankingLoss(margin=margin)
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for user_features, pos_book_features, neg_book_features in dataloader:

            u = user_encoder(user_features)
            v_pos = book_encoder(pos_book_features)
            v_neg = book_encoder(neg_book_features)
            s_pos = F.cosine_similarity(u, v_pos)
            s_neg = F.cosine_similarity(u, v_neg)
            y = torch.ones_like(s_pos)
            loss = criterion(s_pos, s_neg, y)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches if total_batches > 0 else 0.0

def precision_recall_at_k_unsupervised(user_encoder, book_encoder, df, k=20, relevance_threshold=7):
    user_encoder.eval()
    book_encoder.eval()
    precisions = []
    recalls = []
    with torch.no_grad():
        for user_id in df["user_id"].unique():
            user_data = df[df["user_id"] == user_id].reset_index(drop=True)
            relevant_books = set(user_data[user_data["book_rating"] >= relevance_threshold]["book_id"].values)
            if len(relevant_books) == 0:
                continue
            user_features, book_features = get_feature_dicts(user_data)
            U = user_encoder(user_features)
            V = book_encoder(book_features)
            scores = F.cosine_similarity(U, V).cpu().numpy()
            
            n = len(scores)
            top_k = min(k, n)
            top_k_idx = np.argsort(scores)[-top_k:]
            recommended_books = set(user_data["book_id"].values[top_k_idx])
            hits = len(recommended_books & relevant_books)
            precision = hits / top_k
            recall = hits / len(relevant_books)
            precisions.append(precision)
            recalls.append(recall)
            
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    print(f"Contrastive Precision@{k}: {mean_precision:.4f} | Recall@{k}: {mean_recall:.4f}")
    return mean_precision, mean_recall
