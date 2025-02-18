import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

def load_data(file_path):
    """Load and preprocess movie dataset from CSV."""
    df = pd.read_csv(file_path)
    df.dropna(subset=["genre", "overview"], inplace=True)  # Remove rows with missing values
    df["combined_text"] = df["genre"] + " " + df["overview"]
    return df

def tfidf_recommendation(df, user_input, column):
    """Recommend movies using TF-IDF and Cosine Similarity."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df[column].tolist() + [user_input])
    cosine_sim = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])
    top_indices = cosine_sim.flatten().argsort()[-5:][::-1]  # Top 5 matches
    return df.iloc[top_indices]["names"].tolist()

def sbert_recommendation(df, user_input, column):
    """Recommend movies using Sentence-BERT embeddings."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    movie_embeddings = model.encode(df[column].tolist(), convert_to_tensor=True)
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(user_embedding, movie_embeddings)
    top_indices = cosine_sim.argsort(descending=True)[0][:5]  # Top 5 matches
    return df.iloc[top_indices.cpu().numpy()][:]["names"].tolist()

def main():
    file_path = "movies.csv"
    df = load_data(file_path)
    user_input = input("Enter a movie description or genre preference: ")
    '''
    approach = input("Choose approach (genre/overview/combined): ").strip().lower()
    method = input("Choose method (tfidf/sbert): ").strip().lower()
    
    column = "genre" if approach == "genre" else "overview" if approach == "overview" else "combined_text"
    
    if method == "tfidf":
        recommendations = tfidf_recommendation(df, user_input, column)
    elif method == "sbert":
        recommendations = sbert_recommendation(df, user_input, column)
    else:
        print("Invalid method. Choose 'tfidf' or 'sbert'.")
        return
    
    print("Top Recommendations:")
    for movie in recommendations:
        print(movie)
    '''

    approaches = {"genre": "genre", "overview": "overview", "combined": "combined_text"}
    methods = {"tfidf": tfidf_recommendation, "sbert": sbert_recommendation}
    
    for approach_name, column in approaches.items():
        print(f"\nApproach: {approach_name.capitalize()}")
        for method_name, method_func in methods.items():
            print(f"Method: {method_name.upper()}")
            recommendations = method_func(df, user_input, column)
            print("Top Recommendations:")
            for movie in recommendations:
                print(movie)
            print("-" * 40)
    
if __name__ == "__main__":
    main()
