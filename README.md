# Tech Stack Recommender

A content-based filtering recommendation system that maps a user's skills to the most relevant tech career paths using TF-IDF vectorization and Cosine Similarity. Built as Project 3 of the DecodeLabs AI Industrial Training Kit (Batch 2026).

---

## What It Does

The system accepts a minimum of three skills as input, transforms them into a weighted TF-IDF vector, and computes cosine similarity between the user profile and every job role in the dataset. The top 3 most relevant career paths are returned as ranked recommendations with match scores.

---

## Concepts Applied

**Content-Based Filtering**
Recommendations are driven purely by item attributes (job role skill sets) matched against user preferences, with no dependency on other users' historical interaction data.

**TF-IDF Vectorization**
Handled by `sklearn.feature_extraction.text.TfidfVectorizer`. Each job role and the user profile are converted into weighted numerical vectors. Skills that are common across all roles receive a lower weight, while specific and distinguishing skills receive a higher weight.

**Cosine Similarity**
Handled by `sklearn.metrics.pairwise.cosine_similarity`. Similarity is measured by the angle between two vectors rather than their magnitude, so the direction of interests is compared rather than the volume of skills listed.

**4-Step Ranking Pipeline**
1. Ingestion: user skills are captured and normalized
2. Scoring: cosine similarity is calculated against every role in the dataset
3. Sorting: results are ordered in descending order by score
4. Filtering: output is truncated to Top-3 to prevent choice overload

---

## Project Structure

```
DecodeLabs-Internship-Project-3/
    recommender.py
    raw_skills.csv
    requirements.txt
    README.md
```

---

## Dataset

`raw_skills.csv` contains 20 job roles across the tech industry, each associated with 10 relevant skills. Roles include Data Scientist, Machine Learning Engineer, DevOps Engineer, Cloud Architect, NLP Engineer, Cybersecurity Analyst, and others.

---

## Requirements

Python 3.x and the following libraries:

```
pandas
scikit-learn
numpy
```

---

## Installation and Setup

**Clone the repository**

```bash
   git clone https://github.com/Moosa26/DecodeLabs-Internship-Project-3.git
   cd DecodeLabs-Internship-Project-3
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the script**

```bash
python recommender.py
```

---

## Usage

Enter skills one at a time when prompted. Multi-word skills should use underscores.

```
Tech Stack Recommender
========================================
Enter your skills one by one.
Press ENTER after each skill.
Type 'done' when finished (minimum 3 skills required).
========================================
Skill 1: Python
Skill 2: TensorFlow
Skill 3: NLP
Skill 4: done
```

**Sample Output**

```
========================================
Your Skills: python, tensorflow, nlp
========================================
Top Recommended Career Paths:
----------------------------------------

  Rank 1: NLP Engineer
  Match Score : 48.3%
  [#########           ]

  Rank 2: AI Research Scientist
  Match Score : 43.1%
  [########            ]

  Rank 3: Machine Learning Engineer
  Match Score : 31.7%
  [######              ]

========================================
```

---

## Cold Start Handling

Content-based filtering is inherently robust against the Item Cold Start problem. New job roles added to `raw_skills.csv` are immediately available for recommendation based on their listed skills without requiring any interaction history.

For the User Cold Start problem, the system enforces a minimum of three skill inputs to ensure sufficient data density for meaningful similarity computation.

---

## Author

Built as part of the DecodeLabs AI Industrial Training Program, Batch 2026.
