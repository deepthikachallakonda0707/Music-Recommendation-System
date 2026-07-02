# Music Recommendation System

A hybrid music recommendation engine built in Python, combining popularity-based 
and item-similarity (collaborative filtering) approaches to suggest songs to users 
based on listening history.

> **Note:** This project was completed as part of a guided ML training program 
> (Acmegrade). The core recommender logic (`Recommenders.py`) was provided as 
> instructional material; the data pipeline, model application, and analysis 
> notebook were completed by me as part of the training.

## Overview

This project implements two recommendation strategies:
1. **Popularity-based recommender** — recommends the most-listened-to songs 
   across all users (non-personalized, same output regardless of user).
2. **Item similarity recommender** — recommends songs based on collaborative 
   filtering, using a co-occurrence matrix built from Jaccard similarity between 
   songs based on shared listeners (personalized, varies per user).

## Dataset

The dataset consists of two files:
- `triplets_file.csv` — user_id, song_id, and listen_count (how many times a 
  user played a song)
- `song_data.csv` — song_id, title, release, artist_name, and year

Data is a combined/merged set of user listening records and song metadata. 
For this project, a 50,000-record subset was used for model training and 
performance.

*Note: raw data files are not included in full in this repo due to size/licensing 
— see `data/` for a sample structure.*

## Approach

1. **Data preparation**: merged the triplet and metadata files on `song_id`, 
   removed duplicate song entries, and created a combined `song` feature 
   (title + artist name).
2. **Popularity model**: grouped and ranked songs by total listen count across 
   all users to generate a top-10 popularity list.
3. **Item similarity model**: built a song-to-song co-occurrence matrix using 
   Jaccard similarity (intersection over union of listeners between song pairs), 
   then generated personalized top-10 recommendations per user based on their 
   listening history.

## Tech Stack

- Python (pandas, NumPy)
- Jupyter Notebook / Anaconda

## How to Run

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Place `triplets_file.csv` and `song_data.csv` in the `data/` folder
4. Open and run `Music_Recommendation_System.ipynb`

## Files

- `Recommenders.py` — core recommender classes (`popularity_recommender_py`, 
  `item_similarity_recommender_py`)
- `Music_Recommendation_System.ipynb` — data loading, preprocessing, model 
  training, and example recommendations

## Example Output

The item similarity model can also recommend songs based on a given song list, 
independent of a specific user — e.g., given ["Oliver James - Fleet Foxes", 
"The End - Pearl Jam"], it returns similar songs based on shared listener overlap 
across the dataset.

## Future Improvements

- Incorporate content-based filtering using song metadata (genre, artist, year) 
  as a third recommendation strategy
- Add evaluation metrics (precision@k, recall@k) to quantitatively compare the 
  two approaches
- Scale beyond the 50,000-record subset to the full dataset
