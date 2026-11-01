# Anime Recommendation with Neural Collaborative Filtering

![Framework](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-embeddings-orange)
![Dataset](https://img.shields.io/badge/dataset-Anime%20Recommendation%20Database%202020-lightgrey)

An embedding-based collaborative-filtering recommender, trained on millions of user–anime ratings. The learned embeddings support three tasks: finding similar anime, finding similar users, and recommending titles to a given user.

*Project, Sep 2023 – Jan 2024.*

## Method

- **Data:** user ratings from the [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) (`animelist.csv`, `anime.csv`, `anime_with_synopsis.csv`). Only users with at least 400 ratings are kept, and ratings are min-max scaled to [0, 1]. A copy of the files is linked in [`Dataset`](Dataset).
- **Model:** learned user and anime embeddings combined by a normalized dot product, followed by a dense layer, batch normalization and a sigmoid. It is trained with binary cross-entropy on the scaled ratings using Adam, batch size 10,000, 20 epochs, and a warm-up + exponential-decay learning-rate schedule.
- **Fit:** validation MAE reaches about 0.19 on the [0, 1] rating scale.
- **Similar anime:** nearest neighbours in the anime-embedding space (cosine similarity).
- **Similar users:** nearest neighbours in the user-embedding space.
- **Personalized recommendations:** draws on what the user's nearest neighbours in embedding space rated highly. The user's favourite genres are profiled alongside for context.

## Examples

**Similar anime**

![Similar anime](https://github.com/Shahriarmsakib/Recommender-System-for-Anime-Recommendations/assets/114893131/d7a45a96-a830-4f01-a10f-b79472cd96bb)

**Similar users**

![Similar users](https://github.com/Shahriarmsakib/Recommender-System-for-Anime-Recommendations/assets/114893131/b0b89b6a-b3ae-4797-a1d3-fd8c13022afa)

**User preference profile (user 16012)**

![User preferences](https://github.com/Shahriarmsakib/Recommender-System-for-Anime-Recommendations/assets/114893131/682bab24-72f8-429b-a987-bed24420d1da)

**Top-5 recommendations (user 16012)**

![Top recommendations](https://github.com/Shahriarmsakib/Recommender-System-for-Anime-Recommendations/assets/114893131/2eaf602b-ffe6-4204-964e-879149e85fe0)

## Usage

```bash
pip install tensorflow pandas numpy matplotlib wordcloud
jupyter notebook Reccomandation.ipynb
```

Place the three CSV files next to the notebook. A GPU or TPU is strongly recommended, since one epoch takes about 16 minutes on the full ratings table.
