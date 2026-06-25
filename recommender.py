import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df["skills"] = df["skills"].str.lower().str.replace("_", " ")
    return df


def build_vectorizer(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["skills"])
    return vectorizer, tfidf_matrix


def get_user_input():
    print("\nTech Stack Recommender")
    print("=" * 40)
    print("Enter your skills one by one.")
    print("Press ENTER after each skill.")
    print("Type 'done' when finished (minimum 3 skills required).")
    print("=" * 40)

    user_skills = []
    while True:
        skill = input(f"Skill {len(user_skills) + 1}: ").strip()
        if skill.lower() == "done":
            if len(user_skills) < 3:
                print("Please enter at least 3 skills before typing 'done'.")
            else:
                break
        elif skill == "":
            print("Skill cannot be empty. Please try again.")
        else:
            user_skills.append(skill.lower().replace("_", " "))
            print(f"  Added: {skill}")

    return user_skills


def get_recommendations(user_skills, df, vectorizer, tfidf_matrix, top_n=3):
    user_profile = " ".join(user_skills)
    user_vector = vectorizer.transform([user_profile])
    scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    df = df.copy()
    df["score"] = scores
    top_results = df.sort_values("score", ascending=False).head(top_n)
    return top_results


def display_recommendations(recommendations, user_skills):
    print("\n" + "=" * 40)
    print("Your Skills:", ", ".join(user_skills))
    print("=" * 40)
    print("Top Recommended Career Paths:")
    print("-" * 40)

    if recommendations["score"].max() == 0.0:
        print("No strong matches found.")
        print("Try using more specific or technical skill names.")
        return

    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
        percentage = row["score"] * 100
        bar_length = int(percentage / 5)
        bar = "#" * bar_length
        print(f"\n  Rank {rank}: {row['job_role']}")
        print(f"  Match Score : {percentage:.1f}%")
        print(f"  [{bar:<20}]")

    print("\n" + "=" * 40)


def main():
    dataset_path = "raw_skills.csv"

    df = load_dataset(dataset_path)
    vectorizer, tfidf_matrix = build_vectorizer(df)

    while True:
        user_skills = get_user_input()
        recommendations = get_recommendations(user_skills, df, vectorizer, tfidf_matrix, top_n=3)
        display_recommendations(recommendations, user_skills)

        again = input("\nWould you like to try with different skills? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\nThank you for using Tech Stack Recommender.")
            break


if __name__ == "__main__":
    main()
