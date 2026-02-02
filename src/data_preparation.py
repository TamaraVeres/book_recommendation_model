import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_merge_data(file_path_books, file_path_ratings, file_path_users):
    df_books = pd.read_csv(file_path_books)
    df_ratings = pd.read_csv(file_path_ratings)
    df_users = pd.read_csv(file_path_users)
    df_books_ratings = pd.merge(df_books, df_ratings, on="ISBN", how="left")
    df_final = pd.merge(df_books_ratings, df_users, on="User-ID", how="left")
    df_final = df_final.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"])
    return df_final

def rename_columns(df_final):
    df_final = df_final.rename(columns={
        "ISBN": "book_id",
        "Book-Title": "book_title",
        "Book-Author": "book_author",
        "Year-Of-Publication": "year_of_publication",
        "Publisher": "publisher",
        "User-ID": "user_id",
        "Book-Rating": "book_rating",
        "Location": "location",
        "Age": "age",
    })
    return df_final


def clean_and_filter_data(df_final, min_user = 50, min_ratings = 10):
    print("================================================")
    print("========= BEFORE FILTERING MISSING VALUES ========")
    missing_values = df_final.isnull().sum()
    print(df_final.info())
    print(missing_values)
    df_final = df_final.dropna()
    print("================================================")
    print("========= FILTERING OUT 0 RATINGS ========")
    print(f"Rows before removing 0 ratings: {len(df_final)}")
    print(f"Rating distribution before:{df_final['book_rating'].value_counts().sort_index()}")
    df_final = df_final[df_final["book_rating"] > 0]
    print(f"Rows after removing 0 ratings: {len(df_final)}")
    print(f"Rating distribution after:{df_final['book_rating'].value_counts().sort_index()}")
    print(df_final.isnull().sum())
    print(df_final.info()) 
    print("================================================")
    print("========= FILTERING OUT INVALID YEARS ========")
    df_final["year_of_publication"] = pd.to_numeric(df_final["year_of_publication"], errors="coerce")
    df_final = df_final.dropna(subset=["year_of_publication"])
    print("================================================")
    print("========= FILTERING OUT USERS WHO READ LESS THAN 10 BOOKS ========")
    user_book_count = df_final.groupby("user_id")["book_id"].nunique()  
    df_final = df_final[df_final["user_id"].isin(user_book_count[user_book_count >= 10].index)]
    return df_final

def feature_engineering(df_final, current_year=2026):
    #normalize the ratings
    df_final["book_avg_rating"] = df_final.groupby("book_id")["book_rating"].transform("mean")
    df_final["user_avg_rating"] = df_final.groupby("user_id")["book_rating"].transform("mean")
    global_mean = df_final["book_rating"].mean()
    df_final["user_bias"] = df_final["user_avg_rating"] - global_mean
    df_final["book_bias"] = df_final["book_avg_rating"] - global_mean
    df_final["normalized_rating"] = df_final["book_rating"] - (global_mean + df_final["user_bias"] + df_final["book_bias"])
    df_final["book_age"] = current_year - df_final["year_of_publication"].astype(int)
    # user features
    age_bins = [0,18, 25, 35, 45, 55, 65, 120]
    age_labels = [0, 1, 2, 3, 4, 5, 6] 
    df_final["age_category"] = pd.cut(df_final["age"].astype(int), bins=age_bins, labels=False, include_lowest=True)
    df_final["country"] = df_final["location"].str.rsplit(",", n=1).str[-1].str.strip()
    df_final["educated"] = (df_final.groupby("user_id")["user_id"].transform("count") >= 50).astype(int)
    return df_final


def scale_numeric_features(df_final):
    scaler = MinMaxScaler() 
    df_final[["book_age"]] = scaler.fit_transform(df_final[["book_age"]])
    return df_final

def encode_categorical_features(df_final):
  country_encoder = LabelEncoder()
  author_encoder = LabelEncoder()
  publisher_encoder = LabelEncoder()
  age_category_encoder = LabelEncoder()
  title_encoder = LabelEncoder()
  df_final["country"] = country_encoder.fit_transform(df_final["country"])
  df_final["book_author"] = author_encoder.fit_transform(df_final["book_author"])
  df_final["publisher"] = publisher_encoder.fit_transform(df_final["publisher"])
  df_final["age_category"] = age_category_encoder.fit_transform(df_final["age_category"])
  return df_final


def split_by_user(df_final, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    users = df_final["user_id"].unique() 
    # first split
    temp_size = val_size + test_size
    train_users, temp_users = train_test_split(
        users, train_size=train_size, test_size=temp_size, random_state=random_state
    )
    # second split
    test_within_temp = test_size / temp_size
    val_users, test_users = train_test_split(
        temp_users, test_size=test_within_temp, random_state=random_state
    )
    train_df = df_final[df_final["user_id"].isin(train_users)]
    val_df = df_final[df_final["user_id"].isin(val_users)]
    test_df = df_final[df_final["user_id"].isin(test_users)]
    return train_df, val_df, test_df
