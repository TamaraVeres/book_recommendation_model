from src.data_preparation import load_merge_data, rename_columns, clean_and_filter_data, feature_engineering, scale_numeric_features, encode_categorical_features, split_by_user
from src.model import User_Book_Dataset, UserEncoder, BookEncoder, train_model, precision_recall_at_k, RatingHead, get_feature_dicts, predict_ratings
from torch.utils.data import DataLoader
import numpy as np

def main():
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
    train_dataset = User_Book_Dataset(train_df)
    
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    user_encoder = UserEncoder(n_age_bins=len(df_final["age_category"].unique()), n_countries=len(df_final["country"].unique()))
    book_encoder = BookEncoder(n_authors=len(df_final["book_author"].unique()), n_publishers=len(df_final["publisher"].unique()))
    rating_head = RatingHead()
    
    train_model(user_encoder, book_encoder, rating_head, train_dataloader)
    print("Precision and Recall for validation set:")
    precision_recall_at_k(user_encoder, book_encoder, rating_head, df_final, val_df)
    print("Precision and Recall for test set:")
    precision_recall_at_k(user_encoder, book_encoder, rating_head, df_final, test_df)

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

if __name__ == "__main__":
    main()