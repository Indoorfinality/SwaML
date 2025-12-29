import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np

#Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.5
)

# ---------------- Heuristic functions ---------------- #


def is_id_column(series):
    """Check if a pandas Series is an ID column (unique values)."""
    return series.nunique() / len(series) >0.9

def is_text_column(series):
    """Check if a pandas Series is a text column"""
    if series.dtype == 'object':
        avg_len = series.dropna().astype(str).apply(len).mean()
        return avg_len >15
    return False



def detect_target_candidates(df):
    candidates = []
    for col in df.columns:
        series = df[col]
        if is_id_column(series):
            continue
        if is_text_column(series):
            continue
        if series.nunique() <=1:
            continue
        candidates.append(col)
    return candidates

def select_best_target(df, candidates):
    scores = {}

    for col in candidates:
        series = df[col]
        n = len(series)

        nunique = series.nunique(dropna=True)
        unique_ratio = nunique / n

        if nunique <= 1 or unique_ratio > 0.95:
            continue

        cardinality_score = 1 - abs(unique_ratio - 0.1)

        missing_score = 1 - series.isna().mean()

        probs = series.value_counts(normalize=True, dropna=True)
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        entropy_score = entropy / np.log2(nunique + 1e-9)

        dtype_score = 1.0 if pd.api.types.is_numeric_dtype(series) else 0.6

        scores[col] = (
            0.35 * cardinality_score +
            0.25 * entropy_score +
            0.25 * missing_score +
            0.15 * dtype_score
        )

    if not scores:
        return None

    return max(scores, key=scores.get)


# ---------------- Gemini LLM detection ---------------- #


def suggest_target_llm(columns):
    """
    Uses Gemini to suggest target column.
    Returns None if Gemini fails.
            """

    if not GOOGLE_API_KEY:
        return None
    
    try:


        prompt = f"""

You are an expert Automatic Machine Learning system.
        Dataset columns:
        {', '.join(columns)}

        Task:
        Identify the most likely TARGET column for a machine learning model.

        Rules:
        - Output ONLY the column name
        - Do NOT explain your choice
"""
        response = llm.invoke(prompt)
        target = response.content.strip()

        if target not in columns:
            raise ValueError("Gemini suggested invalid column")
        
        return target
    
    except Exception as e:
        print(f"Gemini target detection failed: {e}m using heuristic fallback.")
        return None
    

# ---------------- Main target detection function ---------------- #

def detect_target(df):
    candidates = detect_target_candidates(df)

    #Gemini LLM suggestion
    target = suggest_target_llm(candidates)

# fallback to heuristic if Gemini fails
    if target is None:
        target = select_best_target(df, candidates)
    
    confirm = input(f"Detected target column '{target}'. Confirm? (y/n): ")
    if confirm.lower() != "y":
        target = input("Enter target column manually: ")

    return target

# ---------------- Column drop detection ---------------- #

def suggest_drops_llm(columns, target_column):
    if not GOOGLE_API_KEY:
        return []
    try:
        prompt = f"""
    You are an expert Automatic Machine Learning system.

    Columns: {', '.join(columns)}
    Target: {target_column}
    Task:
    Suggest columns VERY LIKELY IRRELEVANT/NOISY for predicting target.
    Examples: Unique IDs, names, high-missing (>70%), high-unique codes.

    Output ONLY comma-separated list to drop, or 'NONE'.
    No explanation.

        """
        response = llm.invoke(prompt)
        suggestion = response.content.strip()
        if suggestion.upper() == "NONE":
            return []
        drops = [col.strip() for col in suggestion.split(",") if col.strip()]
        return drops
    except Exception as e:
        print(f"Gemini drop suggestion failed: {e}, using heuristic fallback.")
        return []
    
def heuristic_columns_to_drop(df, target_column):
        drops = set()
        for col in df.columns:
            if col == target_column:
                continue
            series = df[col]
            if series.nunique() <=1 or series.isna().mean() >0.7 or is_id_column(series) or is_text_column(series):
                drops.add(col)
            elif series.dtype == 'object' and series.nunique() > 50:
                drops.add(col)
        return list(drops)

    
def detect_columns_to_drop(df, target_column):
    columns = list(df.columns)
    llm_drops = suggest_drops_llm(columns, target_column)
    if llm_drops:
        invalid = [col for col in llm_drops if col not in columns or col == target_column]
        if not invalid:
            print (f"LLM suggested dropping columns: {llm_drops}")
            return llm_drops
        print (f"LLM suggested invalid columns to drop: {invalid}, using heuristic fallback.")
        print ("Using heuristic to detect columns to drop.")
        return heuristic_columns_to_drop(df, target_column)





    


