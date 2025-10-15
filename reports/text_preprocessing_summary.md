# **Text Preprocessing and Linguistic Analysis Report**

##  Project Overview
**Objective:**  
Evaluate vocabulary diversity, n-gram patterns, and linguistic complexity across dataset classes to inform the text preprocessing pipeline for the classification model.

**Dataset:**  
- Source: `../data/raw/train.json`  
- Classes: `factual`, `irrelevant`, `contradiction`  
- Text fields: `context`, `question`, `answer`  
- Total records: *21,021 samples*

---

##  **Vocabulary and Lexical Diversity**

| Label | Vocabulary Size | Total Words | Lexical Diversity |
|:------|----------------:|-------------:|------------------:|
| Factual | 71,711 | 1,111,522 | 0.065 |
| Irrelevant | 23,231 | 112,755 | 0.206 |
| Contradiction | 24,064 | 115,202 | 0.209 |

**Interpretation:**
- The **factual** class has the largest vocabulary but lowest lexical diversity, suggesting long, information-rich texts with repetitive terminology (e.g., historical entities, dates, places).  
- The **irrelevant** and **contradiction** classes show higher lexical diversity, implying more linguistic variation and potentially shorter, more heterogeneous text segments.  
- This pattern indicates that stopword removal, lemmatization, and normalization will be key to balance token density across classes.

---

##  **Word Clouds per Class**

**Saved Figures:**
- `../reports/figures/wordcloud_factual.png`  
- `../reports/figures/wordcloud_irrelevant.png`  
- `../reports/figures/wordcloud_contradiction.png`  

**Insights:**
- **Factual:** dominated by entities such as *“city”, “century”, “war”, “states”*, reflecting historical and geographical content.  
- **Contradiction:** prevalent words like *“not”, “no”, “never”, “however”* highlight logical negation and contrastive structure.  
- **Irrelevant:** less domain-specific, containing generic contextual words and conversational fillers.  

---

##  **N-gram Frequency Analysis**

| Rank | 1-gram | Frequency |
|------|--------:|-----------:|
| 1 | new | 4,561 |
| 2 | city | 3,814 |
| 3 | states | 3,081 |
| 4 | time | 2,877 |
| 5 | used | 2,833 |

| Rank | 2-gram | Frequency |
|------|--------:|-----------:|
| 1 | united states | 1,859 |
| 2 | new york | 855 |
| 3 | world war | 488 |
| 4 | 19th century | 464 |
| 5 | united kingdom | 380 |

| Rank | 3-gram | Frequency |
|------|--------:|-----------:|
| 1 | new york city | 304 |
| 2 | early 20th century | 95 |
| 3 | east india company | 89 |
| 4 | late 19th century | 88 |
| 5 | million years ago | 77 |

**Saved Plots:**
- `../reports/figures/top_1gram.png`  
- `../reports/figures/top_2gram.png`  
- `../reports/figures/top_3gram.png`

**Observations:**
- The dominance of *location- and time-related n-grams* confirms the factual nature of many entries.  
- Frequent bigrams and trigrams indicate structured narrative contexts—ideal for feature extraction via TF-IDF or contextual embeddings.  
- Repetitive factual phrasing suggests the need for stemming/lemmatization to avoid feature inflation.

---

##  **Sentence and Word Structure**

**Sample Statistics (First 5 Texts):**
| Sentence Count | Word Count | Avg Sentence Length |
|:----------------|:-----------:|:--------------------:|
| 7 | 198 | 28.29 |
| 6 | 140 | 23.33 |
| 7 | 140 | 20.00 |
| 4 | 112 | 28.00 |
| 2 | 80 | 40.00 |

**Insights:**
- The average sentence length (~25–30 tokens) indicates moderate syntactic complexity.  
- Factual texts tend to have longer sentences, while irrelevant samples show shorter, less structured patterns.  
- Sentence segmentation quality is adequate; no excessive punctuation noise detected.

---

##  **Preprocessing Recommendations**

| Step | Justification | Implementation |
|:-----|:---------------|:----------------|
| **Lowercasing** | Normalize case sensitivity | `text.lower()` |
| **Stopword removal** | Reduce non-informative tokens | `nltk.corpus.stopwords` |
| **Lemmatization** | Merge inflected forms | `WordNetLemmatizer()` |
| **Rare token filtering** | Handle vocabulary sparsity | Drop tokens with freq < 5 |
| **Punctuation normalization** | Avoid token splitting issues | Use regex cleanup |
| **Class balancing** | Address factual dominance | Apply `StratifiedKFold` or oversampling |
| **n-gram range (1,2)** | Capture local context patterns | `TfidfVectorizer(ngram_range=(1,2))` |

---

##  **Key Takeaways**
- The dataset shows **class-dependent linguistic signatures**, confirming that feature engineering can exploit stylistic cues.  
- **Factual** samples: longer, entity-rich narratives.  
- **Contradiction** samples: negation-heavy, syntactically compact.  
- **Irrelevant** samples: high lexical diversity, less topic coherence.  
- Preprocessing should prioritize text normalization, controlled token pruning, and stratified sampling to preserve label semantics.

---

##  **Artifacts Generated**
| Artifact | Path |
|:----------|:-----|
| Word clouds | `../reports/figures/wordcloud_*.png` |
| N-gram plots | `../reports/figures/top_*.png` |
| Cleaned text dataframe | `../data/processed/train_clean.csv` |
| Summary report | `../reports/text_preprocessing_summary.md` |

