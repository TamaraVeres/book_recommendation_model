import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset



class User_Book_Dataset(Dataset):
    def __init__(self,df_final):
        self.age = torch.tensor(df_final["age_category"].values, dtype=torch.long)
        self.country = torch.tensor(df_final["country"].values, dtype=torch.long)
        self.educated = torch.tensor(df_final["educated"].values, dtype=torch.int32)
        
        self.author = torch.tensor(df_final["book_author"].values, dtype=torch.long)
        self.publisher = torch.tensor(df_final["publisher"].values, dtype=torch.long)
        self.book_age = torch.tensor(df_final["book_age"].values, dtype=torch.float32)
        
        self.rating = torch.tensor(df_final["book_rating"].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.age)
 
     
    def __getitem__(self, index):
        user_features = {
            "age_category": self.age[index],
            "country": self.country[index],
            "educated": self.educated[index],
        }

        book_features = {
            "author": self.author[index],
            "publisher": self.publisher[index],
            "book_age": self.book_age[index],
        }
        rating = self.rating[index]
        return user_features, book_features, rating

class UserEncoder(nn.Module):
    def __init__(self, n_age_bins, n_countries, embedding_dim=8, latent_dim=32):
        super().__init__()
        self.age_embedding = nn.Embedding(n_age_bins, embedding_dim)
        self.country_embedding = nn.Embedding(n_countries, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
            
            
    def forward(self, user_features):
        age_vector = self.age_embedding(user_features["age_category"])
        country_vector = self.country_embedding(user_features["country"])
        educated_vector = user_features["educated"].unsqueeze(1)
        x=torch.cat([age_vector, country_vector, educated_vector], dim=1)
        x=self.mlp(x)
        return nn.functional.normalize(x, dim=1)
    
    

class BookEncoder(nn.Module):
    def __init__(self, n_authors, n_publishers, embedding_dim=16, latent_dim=32):
        super().__init__()
        self.author_embedding = nn.Embedding(n_authors, embedding_dim)
        self.publisher_embedding = nn.Embedding(n_publishers, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        
    def forward(self, book_features):
        author_vector = self.author_embedding(book_features["author"])
        publisher_vector = self.publisher_embedding(book_features["publisher"])
        book_age = book_features["book_age"].unsqueeze(1)
        x=torch.cat([author_vector, publisher_vector, book_age], dim=1)
        x=self.mlp(x)
        return nn.functional.normalize(x, dim=1)
    
    
    
def train_model(user_encoder, book_encoder, dataloader, num_epochs=50, lr=1e-3):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        list(user_encoder.parameters()) + list(book_encoder.parameters()), lr=lr
    )

    for epoch in range(num_epochs):
        total_loss = 0.0

        for user_features, book_features, rating in dataloader:
            U = user_encoder(user_features)
            V = book_encoder(book_features)
            logits = (U * V).sum(dim=1)
            pred_rating = 1.0 + 9.0 * ((logits + 1.0) / 2.0).clamp(0.0, 1.0)
            rating = rating.float().view(-1)
            loss = loss_fn(pred_rating, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        rmse = avg_loss ** 0.5
        print(f"Epoch {epoch+1}/{num_epochs} | Avg MSE: {avg_loss:.4f} | RMSE: {rmse:.4f}")



def precision_recall_at_k(user_encoder, book_encoder, test_df, k=10, relevance_threshold=7):
    user_encoder.eval()
    book_encoder.eval()

    precisions = []
    recalls = []

    with torch.no_grad():
        for user_id in test_df["user_id"].unique():
            user_data = test_df[test_df["user_id"] == user_id].reset_index(drop=True)
            relevant_books = set(
                user_data[user_data["book_rating"] >= relevance_threshold]["book_id"].values
            )

            if len(relevant_books) == 0:
                continue

            user_features = {
                "age_category": torch.tensor(user_data["age_category"].values, dtype=torch.long),
                "country": torch.tensor(user_data["country"].values, dtype=torch.long),
                "educated": torch.tensor(user_data["educated"].values, dtype=torch.float32),
            }
            book_features = {
                "author": torch.tensor(user_data["book_author"].values, dtype=torch.long),
                "publisher": torch.tensor(user_data["publisher"].values, dtype=torch.long),
                "book_age": torch.tensor(user_data["book_age"].values, dtype=torch.float32),
            }

            U = user_encoder(user_features)
            V = book_encoder(book_features)
            logits = (U * V).sum(dim=1)
            scores = logits.numpy().flatten()

            n = len(scores)
            top_k = min(k, n)
            top_k_idx = np.argsort(scores)[-top_k:]
            recommended_books = set(user_data["book_id"].values[top_k_idx])

            hits = len(recommended_books & relevant_books)
            precision = hits / top_k
            recall = hits / len(relevant_books)

            precisions.append(precision)
            recalls.append(recall)

    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    print(f"Precision@{k}: {mean_precision:.4f} | Recall@{k}: {mean_recall:.4f}")
    return mean_precision, mean_recall