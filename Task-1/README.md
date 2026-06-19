# AI-Powered Document Summarization System
**Task ID:** AI-INT-1  
**Domain:** Artificial Intelligence / NLP  
**Internship:** TEYZIX CORE – June Batch 2026  

---

## Project Overview

This project is an AI-based extractive text summarization system built in Python. It accepts long documents and automatically generates concise, meaningful summaries by identifying and ranking the most important sentences using NLP techniques.

---

## Features

- **3 Input Methods:** Load from `.txt` file, `.pdf` file, or direct text input
- **Full NLP Preprocessing:** Lowercasing, stopword removal, tokenization, sentence segmentation
- **Dual Summarization Approach:**
  - Frequency-based scoring
  - TF-IDF based scoring
  - Combined sentence ranking
- **Analytics Module:** Word frequency analysis, keyword extraction, sentence importance scores
- **Adjustable Summary Length:** Choose how many sentences to include
- **Export:** Save summary to `.txt` file

---

## Tech Stack

- Python 3.x
- NLTK (Natural Language Toolkit)
- PyPDF2 (optional, for PDF input)
- Built-in Python libraries: `math`, `string`, `collections`, `os`, `re`

---

## Installation

```bash
pip install nltk PyPDF2
```

On first run, NLTK data is downloaded automatically (punkt tokenizer + stopwords).

---

## How to Run

```bash
python summarizer.py
```

Then follow the menu:
```
Select input method:
  1. Load from .txt file
  2. Load from .pdf file
  3. Enter text directly
```

**Example with sample file:**
```
Enter choice: 1
Enter .txt file path: sample1.txt
How many sentences in summary? 3
```

---

## Project Structure

```
Task-1/
├── summarizer.py        # Main application
├── sample1.txt          # Sample document (AI & Work)
├── sample2.txt          # Sample document (Climate Change)
├── output_summary.txt   # Generated summary (after running)
└── README.md            # This file
```

---

## How It Works

1. **Input** — Text is loaded from file or entered directly
2. **Preprocessing** — Text is cleaned: lowercased, tokenized, stopwords removed
3. **Scoring** — Each sentence is scored using:
   - **Frequency scoring:** sentences with high-frequency important words score higher
   - **TF-IDF scoring:** sentences with rare but significant terms score higher
   - **Combined score:** average of both methods
4. **Ranking** — Sentences ranked by score; top N selected
5. **Output** — Summary printed and optionally exported

---

## Sample Output

```
ORIGINAL TEXT (truncated):
Artificial intelligence is rapidly transforming industries...

SUMMARY (Top 3 sentences):
AI systems are being deployed to automate tasks, improve decision-making,
and enhance productivity. Machine learning models analyze thousands of
patient records to predict chronic conditions. The key challenge is to
manage the transition in a way that distributes the benefits broadly.

Top Keywords:
   ai                   score: 0.0412
   systems              score: 0.0387
   learning             score: 0.0341
```

---

## Author

**[Nimrita dhomeja]**  
BS Computer Science – Sukkur IBA University  
TEYZIX CORE Internship – June Batch 2026
