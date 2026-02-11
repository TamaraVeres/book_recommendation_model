# Book Recommendation System

The goal of this project is to develop a book recommendation system using two complementary approaches based on neural networks:

1. **Supervised mode** — learns to predict explicit ratings using labeled user–book interactions.
2. **Unsupervised mode** — learns user and book representations through contrastive learning, without requiring explicit rating labels.

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run in **supervised** mode (default):

```bash
python book_main.py --mode supervised
```

Run in **unsupervised** (contrastive) mode:

```bash
python book_main.py --mode unsupervised
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
    ├── model_supervised.py
    └── model_unsupervised.py
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

## Shared Components

### Encoders

Both modes share the same encoder architectures, producing embeddings in a common latent space:

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

---

## Mode 1: Supervised Model

### Dataset Construction

A custom PyTorch Dataset class is implemented to structure the data for training.
Each data sample consists of:

- **User features:** Age category, country, education indicator
- **Book features:** Author, publisher, book age
- **Target variable:** Normalized rating — the residual after subtracting global mean, user bias, and book bias from the raw book rating.

### Rating Prediction

- User and book embeddings are combined via a dot product, giving a compatibility score in a bounded range.
- This score is passed through a small linear layer (Rating Head) that predicts the normalized rating (residual), not the raw 1–10 rating.
- The full predicted rating is obtained as: baseline + predicted residual, where the baseline is "global_mean + user_bias + book_bias" for each user–book pair. Predictions are clipped to the [1, 10] scale for display or ranking.

### Training Procedure

- The model is trained with the Adam optimizer (learning rate 0.001, weight decay 0.0001) for 50 epochs.
- During each epoch:
  - User and book embeddings are computed.
  - Predicted ratings are generated from the dot product of embeddings.
  - MSE loss is calculated and backpropagated.
  - Model parameters are updated via gradient descent.
- Training performance is monitored using MSE and Root Mean Squared Error (RMSE).

### Evaluation Metrics (Supervised)

- The model is assessed using Precision@K and Recall@K.
- A book is considered **relevant** if its book rating >= 7 (predefined threshold).
- For each user, the top-*K* books with the highest full predicted ratings (baseline + residual) are selected.
- **Precision@K** — proportion of recommended books that are relevant.
- **Recall@K** — proportion of relevant books that appear in the top-*K* recommendations.
- These metrics are computed on both the validation and test sets; the final reported values are the averages across all users in each set.

### Supervised Results

The model was trained for 50 epochs with Adam (learning rate 0.001, weight decay 0.0001).
Summary of performance:

| Set        | Precision@20 | Recall@20 | Accuracy | MSE    | RMSE   |
|------------|--------------|-----------|----------|--------|--------|
| Validation | 0.8171       | 0.8078    | 0.7958   | 2.0489 | 1.4314 |
| Test       | 0.7998       | 0.8091    | 0.7837   | 1.9462 | 1.3951 |

**Training:** Final epoch training MSE 0.8857 (RMSE 0.9411).

- **Precision@20** — About 82% (validation) and 80% (test) of the top-20 recommended books are relevant (rating >= 7).
- **Recall@20** — About 81% (validation) and 81% (test) of each user's relevant books appear in the top-20 recommendations.
- **Accuracy** — Binary like/dislike classification over all rated items: ~80% (validation), ~78% (test).
- **MSE / RMSE** — Rating prediction error on unseen users: test RMSE ≈ 1.40 on the 1–10 scale.

---

## Mode 2: Unsupervised (Contrastive) Model

### Motivation

The supervised model requires explicit rating labels to learn. The unsupervised mode instead learns user and book representations purely from the structure of user–book interactions, which books a user has interacted with and which they have not, without relying on the actual rating values during training. 

### Contrastive Learning Approach

The unsupervised model is trained using a **contrastive learning** framework. The core idea is to learn embeddings such that a user's representation is close to the books they have interacted with (positives) and far from books they have not interacted with (negatives).

#### Pair Construction

A custom `ContrastivePairDataset` generates training triplets:

- **Anchor:** The user, represented by their feature vector (age category, country, education indicator).
- **Positive:** A book the user has actually interacted with, represented by its feature vector (author, publisher, book age).
- **Negative:** A randomly sampled book that the user has **not** interacted with, drawn uniformly from all books in the training set.

Each training sample thus consists of one user, one positive book, and one negative book — a **1:1 positive-to-negative ratio**. The negative book is re-sampled randomly each epoch, so the model sees a diverse set of negatives over the course of training.

#### Loss Function

Training uses **MarginRankingLoss** with cosine similarity as the scoring function:

1. Compute the user embedding **u**, positive book embedding **v_pos**, and negative book embedding **v_neg** using the shared encoders.
2. Calculate cosine similarity scores: s_pos = cosine(u, v_pos) and s_neg = cosine(u, v_neg).
3. The MarginRankingLoss enforces: s_pos - s_neg >= margin, where margin = 0.5.

This encourages the model to place each user closer to their interacted books than to random unrelated books in the latent space by at least the specified margin.

#### Training Procedure

- The same User Encoder and Book Encoder architectures are reused.
- The model is trained with the Adam optimizer (learning rate 0.001, weight decay 0.0001) for 10 epochs.
- Batch size is 512.
- No rating is needed, recommendations are based directly on cosine similarity between user and book embeddings.

### Evaluation (Unsupervised)

#### Contrastive Loss

The validation and test sets are evaluated by constructing contrastive pairs in the same way as training and computing the average MarginRankingLoss (with a reduced margin of 0.05 for evaluation).

#### Precision@K and Recall@K

For each user in the evaluation set:
1. Compute cosine similarity between the user embedding and all book embeddings in that user's interaction set.
2. Rank books by similarity score and select the top-*K*.
3. A book is considered **relevant** if its actual rating >= 7.
4. Precision@K and Recall@K are computed per user and averaged across all users.

### Unsupervised Results

The model was trained for 10 epochs. Summary of performance:

| Set        | Precision@20 | Recall@20 | Loss   |
|------------|--------------|-----------|--------|
| Validation | 0.7646       | 0.7676    | 0.1330 |
| Test       | 0.7531       | 0.7782    | 0.1369 |

**Training:** Final epoch contrastive loss: 0.2979.

- **Precision@20** — About 76% (validation) and 75% (test) of the top-20 recommended books are relevant.
- **Recall@20** — About 77% (validation) and 78% (test) of each user's relevant books appear in the top-20 recommendations.
