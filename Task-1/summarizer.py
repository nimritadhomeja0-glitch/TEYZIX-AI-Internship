"""
AI-Powered Document Summarization System
Task ID: AI-INT-1
TEYZIX CORE Internship - June Batch
Author: [Your Name]
"""

import os
import re
import math
import string
from collections import Counter

# NLTK setup
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Optional PDF support
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ─────────────────────────────────────────────
# 1. INPUT SYSTEM
# ─────────────────────────────────────────────

def load_from_text_file(filepath):
    """Load text from a .txt file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_from_pdf(filepath):
    """Load text from a .pdf file."""
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    text = ""
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_user_input():
    """Accept direct text input from the user."""
    print("\nPaste your text below. Type 'END' on a new line when done:\n")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    return '\n'.join(lines)


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_text(text):
    """
    Full preprocessing pipeline:
    - Sentence segmentation
    - Tokenization
    - Lowercasing
    - Stopword removal
    Returns original sentences and cleaned word lists per sentence.
    """
    # Sentence segmentation
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    stop_words = set(stopwords.words('english'))
    cleaned_sentences = []

    for sentence in sentences:
        # Lowercase
        lowered = sentence.lower()
        # Tokenize
        tokens = word_tokenize(lowered)
        # Remove stopwords and punctuation
        filtered = [
            w for w in tokens
            if w not in stop_words and w not in string.punctuation and w.isalpha()
        ]
        cleaned_sentences.append(filtered)

    return sentences, cleaned_sentences


# ─────────────────────────────────────────────
# 3. SUMMARIZATION LOGIC
# ─────────────────────────────────────────────

def frequency_based_scoring(sentences, cleaned_sentences):
    """Score sentences based on word frequency."""
    # Build word frequency across all sentences
    all_words = [word for tokens in cleaned_sentences for word in tokens]
    word_freq = Counter(all_words)

    # Normalize frequencies
    max_freq = max(word_freq.values()) if word_freq else 1
    for word in word_freq:
        word_freq[word] /= max_freq

    # Score each sentence
    scores = []
    for tokens in cleaned_sentences:
        score = sum(word_freq.get(w, 0) for w in tokens)
        score = score / len(tokens) if tokens else 0
        scores.append(score)

    return scores


def tfidf_scoring(sentences, cleaned_sentences):
    """Score sentences using TF-IDF approach."""
    n = len(cleaned_sentences)
    if n == 0:
        return []

    # TF: term frequency per sentence
    tf_scores = []
    for tokens in cleaned_sentences:
        freq = Counter(tokens)
        total = len(tokens) if tokens else 1
        tf = {w: c / total for w, c in freq.items()}
        tf_scores.append(tf)

    # IDF: inverse document frequency
    all_words = set(w for tokens in cleaned_sentences for w in tokens)
    idf = {}
    for word in all_words:
        doc_count = sum(1 for tokens in cleaned_sentences if word in tokens)
        idf[word] = math.log(n / (1 + doc_count))

    # TF-IDF sentence score
    scores = []
    for tf in tf_scores:
        score = sum(tf[w] * idf.get(w, 0) for w in tf)
        score = score / len(tf) if tf else 0
        scores.append(score)

    return scores


def rank_sentences(sentences, freq_scores, tfidf_scores):
    """Combine both scores and rank sentences."""
    combined = []
    for i, sentence in enumerate(sentences):
        combined_score = (freq_scores[i] * 0.5) + (tfidf_scores[i] * 0.5)
        combined.append((i, sentence, combined_score))
    # Sort by score descending
    ranked = sorted(combined, key=lambda x: x[2], reverse=True)
    return ranked


def generate_summary(sentences, cleaned_sentences, num_sentences=3):
    """Generate extractive summary by picking top N sentences."""
    if not sentences:
        return "No content to summarize.", []

    num_sentences = min(num_sentences, len(sentences))

    freq_scores = frequency_based_scoring(sentences, cleaned_sentences)
    tfidf_scores = tfidf_scoring(sentences, cleaned_sentences)
    ranked = rank_sentences(sentences, freq_scores, tfidf_scores)

    # Pick top N, restore original order
    top = sorted(ranked[:num_sentences], key=lambda x: x[0])
    summary = ' '.join([s[1] for s in top])

    return summary, ranked


# ─────────────────────────────────────────────
# 4. ANALYTICS MODULE
# ─────────────────────────────────────────────

def word_frequency_analysis(cleaned_sentences, top_n=10):
    """Return top N most frequent words."""
    all_words = [w for tokens in cleaned_sentences for w in tokens]
    freq = Counter(all_words)
    return freq.most_common(top_n)


def extract_keywords(cleaned_sentences, top_n=5):
    """Extract most important keywords using TF-IDF."""
    n = len(cleaned_sentences)
    if n == 0:
        return []

    all_words = set(w for tokens in cleaned_sentences for w in tokens)
    idf = {}
    for word in all_words:
        doc_count = sum(1 for tokens in cleaned_sentences if word in tokens)
        idf[word] = math.log(n / (1 + doc_count) + 1)

    all_word_list = [w for tokens in cleaned_sentences for w in tokens]
    freq = Counter(all_word_list)
    total = len(all_word_list) or 1

    tfidf_keywords = {w: (freq[w] / total) * idf[w] for w in all_words}
    sorted_keywords = sorted(tfidf_keywords.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]


def sentence_importance_scores(ranked_sentences):
    """Show each sentence with its importance score."""
    return [(i + 1, score, sent[:80] + '...') for i, (_, sent, score) in enumerate(ranked_sentences)]


# ─────────────────────────────────────────────
# 5. OUTPUT SYSTEM
# ─────────────────────────────────────────────

def display_results(original_text, summary, ranked, cleaned_sentences, top_n):
    """Print results in a clean format."""
    print("\n" + "=" * 60)
    print("ORIGINAL TEXT")
    print("=" * 60)
    print(original_text[:1000] + ("..." if len(original_text) > 1000 else ""))

    print("\n" + "=" * 60)
    print(f"SUMMARY (Top {top_n} sentences)")
    print("=" * 60)
    print(summary)

    print("\n" + "=" * 60)
    print("ANALYTICS")
    print("=" * 60)

    print("\n Top Keywords:")
    for word, score in extract_keywords(cleaned_sentences):
        print(f"   {word:<20} score: {score:.4f}")

    print("\n Word Frequency (Top 10):")
    for word, count in word_frequency_analysis(cleaned_sentences):
        print(f"   {word:<20} count: {count}")

    print("\n Sentence Importance Scores:")
    for rank, score, snippet in sentence_importance_scores(ranked):
        print(f"   Rank {rank}: score={score:.4f} | {snippet}")


# ─────────────────────────────────────────────
# 6. FILE EXPORT
# ─────────────────────────────────────────────

def export_summary_txt(summary, output_path="output_summary.txt"):
    """Export summary to a .txt file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("AI-POWERED DOCUMENT SUMMARIZATION SYSTEM\n")
        f.write("TEYZIX CORE Internship - Task AI-INT-1\n")
        f.write("=" * 50 + "\n\n")
        f.write("GENERATED SUMMARY:\n\n")
        f.write(summary)
    print(f"\n Summary exported to: {output_path}")


# ─────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI-POWERED DOCUMENT SUMMARIZATION SYSTEM")
    print("  TEYZIX CORE Internship | Task AI-INT-1")
    print("=" * 60)

    # Input selection
    print("\nSelect input method:")
    print("  1. Load from .txt file")
    print("  2. Load from .pdf file")
    print("  3. Enter text directly")
    choice = input("\nEnter choice (1/2/3): ").strip()

    text = ""
    try:
        if choice == '1':
            path = input("Enter .txt file path: ").strip()
            text = load_from_text_file(path)
        elif choice == '2':
            path = input("Enter .pdf file path: ").strip()
            text = load_from_pdf(path)
        elif choice == '3':
            text = get_user_input()
        else:
            print("Invalid choice. Exiting.")
            return
    except (FileNotFoundError, ImportError) as e:
        print(f"Error: {e}")
        return

    if not text.strip():
        print("No text found. Exiting.")
        return

    # Summary length
    try:
        top_n = int(input("\nHow many sentences in summary? (default 3): ").strip() or 3)
    except ValueError:
        top_n = 3

    # Preprocess
    print("\nProcessing...")
    sentences, cleaned = preprocess_text(text)

    # Summarize
    summary, ranked = generate_summary(sentences, cleaned, num_sentences=top_n)

    # Display
    display_results(text, summary, ranked, cleaned, top_n)

    # Export
    save = input("\nExport summary to file? (y/n): ").strip().lower()
    if save == 'y':
        out_path = input("Output filename (default: output_summary.txt): ").strip() or "output_summary.txt"
        export_summary_txt(summary, out_path)

    print("\nDone! Thank you for using the AI Summarization System.\n")


if __name__ == "__main__":
    main()
