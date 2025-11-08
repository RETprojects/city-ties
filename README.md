# city-ties
Discover which cities have some unexpected things in common! This application uses NLP and Wikipedia pages to find attributes that cities around the world have in common.

## TODO
- [X] Get data for each capital (or each city w/ at least 1M inhabitants)
    - [X] Summary
    - [X] Whole page
- [ ] Enable user to search for cities which have a certain attribute
- [ ] Determine for a city the "recommended" cities
    - [X] Clustering
        - [ ] Consider using overlapping clustering algorithm like fuzzy c-means, NOT an exclusive algorithm like k-means
        - [X] Determine optimal number of clusters empirically (using set of capitals (or cities w/ >1M inhabitants))
        - [ ] Consider using LLM embeddings or sentence transformers (as opposed to TF-IDF, GloVe, or BERT, for example)
        - [ ] Parallel processing (scikit-learn's KMeans already uses OpenMP-based parallelism through Cython w/ a low memory footprint)
    - [ ] Use clustering results to display "neighbors" of each city based on cluster number
- [ ] Map showing cities
