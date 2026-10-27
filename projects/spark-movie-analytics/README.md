# Spark Movie-Rating Analytics

Original CSCE 5300 notebook using Spark DataFrames and RDDs to inspect movie ratings, join movie titles with rating counts, find highly rated movies, and identify active users.

## Run

Open `movie_ratings.ipynb` in Google Colab or a Jupyter environment with PySpark and a compatible Java runtime. Install the packages in `requirements.txt` for local use.

Provide CSV files with `movieId,title` columns for movies and `userId,movieId,rating` columns for ratings. Update `movies_path` and `ratings_path` in the notebook to point to your files, then run cells in order. The source retains its original Colab paths and installation cell.

Dataset files are not included. Notebook outputs and metadata were cleared; no new dataset results are claimed. This notebook complements the existing [Hadoop/Hive report repository](https://github.com/MdKhizar01/Hadoop-MapReduce-Hive-on-AWS-Big-Data-Processing).
