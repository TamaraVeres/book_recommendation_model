from src.data_preparation import load_merge_data, rename_columns, clean_and_filter_data, feature_engineering, scale_numeric_features, encode_categorical_features, split_by_user
from src.model_supervised import User_Book_Dataset, UserEncoder, BookEncoder, train_model, precision_recall_at_k, RatingHead, get_feature_dicts, predict_ratings
from src.model_unsupervised import ContrastivePairDataset, train_unsupervised_model, evaluate_unsupervised_model, precision_recall_at_k_unsupervised
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from sklearn.manifold import TSNE
import torch.nn.functional as F



def prepare_data():
    df_final = load_merge_data(file_path_books="../data/Books.csv", file_path_ratings="../data/Ratings.csv", file_path_users="../data/Users.csv")
    print(df_final.head())
    df_final = rename_columns(df_final)
    df_final = clean_and_filter_data(df_final)
    df_final = feature_engineering(df_final)
    df_final.to_csv("../data/merged_book_data.csv", index=False)
    print(df_final.head())
    
    df_final = scale_numeric_features(df_final)
    df_final = encode_categorical_features(df_final)
    df_final.to_csv("../data/encoded_book_data.csv", index=False)
    print(df_final.head())
    train_df, val_df, test_df = split_by_user(df_final)
    return df_final, train_df, val_df, test_df
    
    
def supervised_model(df_final, train_df, val_df, test_df):
    train_dataset = User_Book_Dataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    user_encoder = UserEncoder(n_age_bins=len(df_final["age_category"].unique()), n_countries=len(df_final["country"].unique()))
    book_encoder = BookEncoder(n_authors=len(df_final["book_author"].unique()), n_publishers=len(df_final["publisher"].unique()))
    rating_head = RatingHead()
    
    train_model(user_encoder, book_encoder, rating_head, train_dataloader)
    k_1, k_5, k_10, k_20, k_30, k_40 = 1, 5, 10, 20, 30, 40
    print("================================================")
    print("========= Precision and Recall for validation set:")
    print(f"Precision@{k_1} and Recall@{k_1} for validation set:")
    precision_recall_at_k(user_encoder, book_encoder, rating_head, df_final, val_df, k_1)
    print("================================================")
    print(f"Precision@{k_5} and Recall@{k_5} for validation set:")
    precision_recall_at_k(user_encoder, book_encoder, rating_head, df_final, val_df, k_5)
    print("================================================")
    print(f"Precision@{k_10} and Recall@{k_10} for validation set:")
    precision_recall_at_k(user_encoder, book_encoder, rating_head, df_final, val_df, k_10)
    print("================================================")
    print(f"Precision@{k_20} and Recall@{k_20} for validation set:")
    precision_recall_at_k(user_encoder, book_encoder, rating_head, df_final, val_df, k_20)
    print("================================================")
    print(f"Precision@{k_30} and Recall@{k_30} for validation set:")
    precision_recall_at_k(user_encoder, book_encoder, rating_head, df_final, val_df, k_30)
    print("================================================")
    print(f"Precision@{k_40} and Recall@{k_40} for validation set:")
    precision_recall_at_k(user_encoder, book_encoder, rating_head, df_final, val_df, k_40)
    print("================================================")
    print(f"Precision@{k_20} and Recall@{k_20} for test set:")
    precision_recall_at_k(user_encoder, book_encoder, rating_head, df_final, test_df, k_20)

    # print top 10 for one user
    user_encoder.eval()
    book_encoder.eval()
    rating_head.eval()
    uid = np.random.choice(test_df["user_id"].unique())
    user_dataframe = test_df[test_df["user_id"] == uid].reset_index(drop=True)
    global_mean = df_final["book_rating"].mean()
    user_features, book_features = get_feature_dicts(user_dataframe)
    pred_rating_1_10 = predict_ratings(
        user_encoder, book_encoder, rating_head,
        user_features, book_features,
        global_mean, user_dataframe["user_bias"].values, user_dataframe["book_bias"].values,
        clip_1_10=True,
    )
    top10 = np.argsort(pred_rating_1_10)[-10:][::-1]
    print(f"\nTop 10 for user {uid}:")
    for i, j in enumerate(top10, 1):
        print(f"  {i}. {user_dataframe['book_title'].iloc[j]}  pred={pred_rating_1_10[j]:.2f}  actual={user_dataframe['book_rating'].iloc[j]:.1f}")

def unsupervised_model(df_final, train_df, val_df, test_df):
    dataset = ContrastivePairDataset(train_df)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    user_encoder = UserEncoder(
        n_age_bins=len(df_final["age_category"].unique()),
        n_countries=len(df_final["country"].unique()),
    )
    book_encoder = BookEncoder(
        n_authors=len(df_final["book_author"].unique()),
        n_publishers=len(df_final["publisher"].unique()),
    )
    train_unsupervised_model(user_encoder, book_encoder, dataloader)
    val_loss = evaluate_unsupervised_model(user_encoder, book_encoder, val_df)
    test_loss = evaluate_unsupervised_model(user_encoder, book_encoder, test_df)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    for k in [10, 20, 30, 40, 50]:
        precision_recall_at_k_unsupervised(user_encoder, book_encoder, val_df, k=k)
    for k in [10, 20, 30, 40, 50]:
        precision_recall_at_k_unsupervised(user_encoder, book_encoder, test_df, k=k)

    # print top 10 for one user
    user_encoder.eval()
    book_encoder.eval()
    uid = np.random.choice(test_df["user_id"].unique())
    user_dataframe = test_df[test_df["user_id"] == uid].reset_index(drop=True)
    user_features, book_features = get_feature_dicts(user_dataframe)
    with torch.no_grad():
       U = user_encoder(user_features)
       V = book_encoder(book_features)
       scores = F.cosine_similarity(U, V).numpy()
    top10 = np.argsort(scores)[-10:][::-1]
    print(f"\nTop 10 for user {uid}:")
    for i, j in enumerate(top10, 1):
       print(f"  {i}. {user_dataframe['book_title'].iloc[j]}  sim={scores[j]:.4f}  actual={user_dataframe['book_rating'].iloc[j]:.1f}")
    
    
    # visualization 
    user_encoder.cpu()
    book_encoder.cpu()
    user_encoder.eval()
    book_encoder.eval()

    user_features_batch, pos_book_features_batch, neg_book_features_batch = next(iter(dataloader))

    with torch.no_grad():
        U = user_encoder(user_features_batch)             
        V_pos = book_encoder(pos_book_features_batch)      
        V_neg = book_encoder(neg_book_features_batch)      

    X = torch.vstack([U, V_pos, V_neg]).cpu().numpy()      
    labels = np.array(
        ["user"] * U.size(0) +
        ["pos"] * V_pos.size(0) +
        ["neg"] * V_neg.size(0)
    )

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        X_2d[labels == "user", 0], X_2d[labels == "user", 1],
        c="red", marker="X", s=80, label="users",
    )
    plt.scatter(
        X_2d[labels == "pos", 0], X_2d[labels == "pos", 1],
        c="green", s=10, alpha=0.7, label="positive books",
    )
    plt.scatter(
        X_2d[labels == "neg", 0], X_2d[labels == "neg", 1],
        c="gray", s=10, alpha=0.4, label="negative books",
    )

    plt.title("Users, positives, and negatives in unsupervised latent space (t-SNE 2D)")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(mode: str = "supervised"):
    df_final, train_df, val_df, test_df = prepare_data()
    if mode == "supervised":
        supervised_model(df_final, train_df, val_df, test_df)
    elif mode == "unsupervised":
        unsupervised_model(df_final, train_df, val_df, test_df)
    else:
        raise ValueError(f"Invalid mode: {mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="supervised", choices=["supervised", "unsupervised"])
    args = parser.parse_args()
    main(mode=args.mode)
