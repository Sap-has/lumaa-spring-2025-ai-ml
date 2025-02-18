**Introduction**
This is a program when given a user input will recommend ther user movies similar to their input. It will allow for user to choose which data will be used to compare such as genre, overview or combined. 
It will also allow for user to choose which algorithm to pick TF-IDF or SBERT. 
It will output the top 5 movies most similar to the user's liking from their input.

1. **Dataset**
    - Found in Kaggle
    - https://www.kaggle.com/datasets/ashpalsingh1525/imdb-movies-dataset
    - Was cleaned to only include the first 499 entries with name, genre and overview being the only columns present
    - You have many options from using an API key from kaggle or downloading a zip file which is what I did

2. **Aproach**
    - Using TF-IDF with cosine similarity for simple reccomendations
    - Using Word Embedding using SBERT for deeper search

3. **Setup**
    **Requirements:**
    - Python 3.8+
    - Virtual environment (recommended)

    **Installation**
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    venv\Scripts\activate  # Windows

    # Install dependencies
    pip install -r requirements.txt

    **Running The Program**
    python movie_recc.py

4. **Example input and outputs**
    user input
    'I enjoy comidies that have action and are fantasy related.'
    'combined'
    'sbert'

    output
    '''
    Top Recommendations:
    Babylon
    Frozen II
    Groot Takes a Bath
    Barbie
    King Shakir Recycle
    '''