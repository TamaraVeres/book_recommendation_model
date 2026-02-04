# Book recommendation system using a supervised model

The goal of this project is to develop a book recommendation system using a supervised learning approach based on a neural network.

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis:

```bash
python book_main.py
```

---

## Project Structure

```
book_reccomendation/
├── book_main.py                        
├── README.md
├── requirements.txt
└── src/
    ├── data_preparation.py   
    └── model.py             
```

Data paths (CSVs) are expected under `../data/` (e.g. `Books.csv`, `Ratings.csv`, `Users.csv`).

---

## Data Inspection and Preparation

### Data Sources and Merging

The dataset used in this project is constructed by merging three separate data sources: books, ratings, and users.The datasets are merged using the book ISBN and user identifiers to create a unified table containing book metadata, user information, and ratings.
After merging, non-informative image URL columns are removed, as they are not relevant for model training.

### Column Renaming

To improve readability and consistency throughout the project, column are renamed.
For example, identifiers such as ISBN and User-ID are renamed to book_id and user_id, and rating-related fields are renamed accordingly.

### Data Inspection and Cleaning

An initial inspection of the dataset is performed to identify missing values and incorrect data types. Rows containing missing values are removed to ensure data consistency and avoid issues during model training. Next, ratings with a value of 0 are filtered out. These ratings do not represent explicit user feedback and could introduce noise into the supervised learning process.
The year_of_publication feature is converted to a numeric format, and invalid or missing years are removed. This ensures that derived features, such as book age, are computed correctly.
Finally, users who have rated fewer than 10 books are excluded from the dataset. This step ensures that each user has sufficient ratings, which is important for learning meaningful user representations.

### Feature Engineering

Several features are engineered to enrich the dataset:

- **Rating-based features:**
  - Average rating per user
  - Average rating per book
  - User bias and book bias relative to the global mean rating
  - A normalized rating that removes global, user, and book bias components

- **Book features:**
  - Book age, computed as the difference between the current year and the year of publication

- **User features:**
  - Age categories created by binning user ages into predefined intervals
  - Country extracted from the user location field
  - An education proxy feature indicating whether a user has rated at least 50 books

These engineered features help the neural network capture both user preferences and book characteristics more effectively.

### Feature Scaling

Numerical features, specifically book age, are scaled using Min-Max normalization.
This ensures that numeric inputs lie within a consistent range, which improves the stability and convergence of neural network training.

### Categorical Encoding

Categorical features such as country, book author, publisher, and age category are encoded using label encoding.
This transforms categorical values into numerical representations suitable for embedding layers in the neural network model.

### Train–Validation–Test Split

To prevent information leakage and ensure a fair evaluation, the dataset is split by user, rather than by individual interactions.

- 70% of users are assigned to the training set
- 15% of users to the validation set
- 15% of users to the test set

This user-level split guarantees that no user appears in more than one subset, allowing the model to be evaluated on completely unseen users during validation and testing.

## Model Architecture and Training

### Dataset Construction

A custom PyTorch Dataset class is implemented to structure the data for training.
Each data sample consists of:

- **User features:**
  - Age category
  - Country
  - Education indicator

- **Book features:**
  - Author
  - Publisher
  - Book age

- **Target variable:**
  - Normalized rating — the residual after subtracting global mean, user bias, and book bias from the raw book rating.

### Encoders

- **User Encoder** — Transforms user attributes into a fixed-length latent vector:
  - Categorical features (age category and country) are represented using embedding layers.
  - The education indicator is concatenated as a numerical feature.
  - All features are passed through a multi-layer perceptron (MLP) with ReLU activations.
  - The output embedding is L2-normalized to stabilize training and ensure comparable latent scales.

- **Book Encoder** — Maps book attributes into the same latent space as users:
  - Author and publisher are embedded using learned embedding layers.
  - Book age is incorporated as a numerical feature.
  - The combined features are processed through an MLP architecture mirroring the user encoder.
  - The final book representation is also L2-normalized.

### Rating Prediction

- User and book embeddings are combined via a dot product, giving a compatibility score in a bounded range.
- This score is passed through a small linear layer that predicts the normalized rating (residual), not the raw 1–10 rating.
- The full predicted rating is obtained as: baseline + predicted residual, where the baseline is "global_mean + user_bias + book_bias" for each user–book pair. Predictions are clipped to the [1, 10] scale for display or ranking.

### Training Procedure

- The model is trained with the Adam optimizer and a fixed learning rate.
- During each epoch:
  - User and book embeddings are computed.
  - Predicted ratings are generated from the dot product of embeddings.
  - MSE loss is calculated and backpropagated.
  - Model parameters are updated via gradient descent.
- Training performance is monitored using MSE and Root Mean Squared Error (RMSE).

### Evaluation Metrics

- The model is assessed using Precision@K and Recall@K with *K* = 20.
- A book is considered **relevant** if its **book rating is ≥ 7** (predefined threshold).
- For each user, the top-*K* books with the highest **full predicted ratings** (baseline + residual) are selected.
- **Precision@K** — proportion of recommended books that are relevant.
- **Recall@K** — proportion of relevant books that appear in the top-*K* recommendations.
- These metrics are computed on both the **validation** and **test** sets, the final reported values are the averages of Precision@K and Recall@K across all users in each set.

## Results

The model was trained for 50 epochs with Adam (learning rate 0.001, weight decay 0.0001). 
Summary of performance:

| Set        | Precision@20 | Recall@20 | Accuracy | MSE    | RMSE   |
|------------|---------------|-----------|--------------|--------|--------|
| Validation | 0.8171        | 0.8078    | 0.7958       | 2.0489 | 1.4314 |
| Test       | 0.7998        | 0.8091    | 0.7837       | 1.9462 | 1.3951 |

**Training:** Final epoch training MSE 0.8857 (RMSE 0.9411).

- **Precision@10** — About 87% of the top-10 recommended books are relevant (rating ≥ 7) on both validation and test.
- **Recall@10** — About 56–57% of each user’s relevant books appear in the top-10 recommendations.
- **MSE / RMSE** — Rating prediction error on unseen users: test RMSE ≈ 1.37 on the 1–10 scale.


