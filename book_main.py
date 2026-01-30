from src.data_preparation import load_merge_data, rename_columns, clean_and_filter_data, feature_engineering, scale_numeric_features, encode_categorical_features, split_by_user
from src.model import User_Book_Dataset, UserEncoder, BookEncoder, train_model, precision_recall_at_k
import torch
from torch.utils.data import DataLoader
import numpy as np

def main():
    df_final = load_merge_data(file_path_books="./data/books.csv", file_path_ratings="./data/ratings.csv", file_path_users="./data/users.csv")
    print(df_final.head())
    df_final = rename_columns(df_final)
    df_final = clean_and_filter_data(df_final)
    df_final = feature_engineering(df_final)
    df_final.to_csv("./data/merged_book_data.csv", index=False)
    print(df_final.head())
    df_final = scale_numeric_features(df_final)
    df_final = encode_categorical_features(df_final)
    df_final.to_csv("./data/encoded_book_data.csv", index=False)
    print(df_final.head())
    train_df, test_df = split_by_user(df_final)
    dataset = User_Book_Dataset(train_df)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    user_encoder = UserEncoder(n_age_bins=len(df_final["age_category"].unique()), n_countries=len(df_final["country"].unique()))
    book_encoder = BookEncoder(n_authors=len(df_final["book_author"].unique()), n_publishers=len(df_final["publisher"].unique()))
    train_model(user_encoder, book_encoder, dataloader)
    precision_recall_at_k(user_encoder, book_encoder, test_df)
    
    # print top 10 for one user
    user_encoder.eval()
    book_encoder.eval()
    
    # We ge the user and their rows
    uid = test_df["user_id"].unique()[0]
    user_dataframe = test_df[test_df["user_id"] == uid].reset_index(drop=True)
    
    def mapping_formula(dot_prod_cross):
        return 1.0 + 9.0 * ((dot_prod_cross + 1.0) / 2.0).clip(0, 1)
    
    user_features = {"age_category": torch.tensor(user_dataframe["age_category"].values, dtype=torch.long), "country": torch.tensor(user_dataframe["country"].values, dtype=torch.long), "educated": torch.tensor(user_dataframe["educated"].values, dtype=torch.float32)}
    book_features = {"author": torch.tensor(user_dataframe["book_author"].values, dtype=torch.long), "publisher": torch.tensor(user_dataframe["publisher"].values, dtype=torch.long), "book_age": torch.tensor(user_dataframe["book_age"].values, dtype=torch.float32)}
    with torch.no_grad():
        dot_prod_cross = (user_encoder(user_features) * book_encoder(book_features)).sum(dim=1).numpy()
    pred = 1.0 + 9.0 * ((dot_prod_cross + 1.0) / 2.0).clip(0, 1)
    top10 = np.argsort(pred)[-10:][::-1]
    print(f"\nTop 10 for user {uid}:")
    for i, j in enumerate(top10, 1):
        print(f"  {i}. {user_dataframe['book_title'].iloc[j]}  pred={pred[j]:.2f}  actual={user_dataframe['book_rating'].iloc[j]:.1f}")
   
if __name__ == "__main__":
    main()