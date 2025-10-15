import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Define the path to the training data JSON file
train_path = Path("../data/raw/train.json")

# Load the training data from JSON file
with open(train_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert the loaded data into a pandas DataFrame
# This DataFrame will be used for exploratory data analysis (EDA)
df = pd.DataFrame(data)

# Print DataFrame info: show's column types, non-null counts, and memory usage
print("DataFrame Info:")
print(df.info())

# Print the first two rows as an example of the dataset structure
print("Example rows (first 2):")
print(df.head(2))

# Print dataset statistics: count, unique, top, freq, mean, std, etc. for all columns
print("Dataset statistics (describe):")
print(df.describe(include='all'))

# Print the shape of the DataFrame: number of rows and columns
print("Shape of the DataFrame (rows, columns):")
print(df.shape)

# Print missing values per column
print("Missing values per column:")
print(df.isnull().sum())

# Visualize missing values as a heatmap (no color bar)
sns.heatmap(df.isnull(), cbar=False)

# Print the number of duplicated rows in the DataFrame
print("Number of duplicated rows:")
print(df.duplicated().sum())

# Print the duplicated rows themselves for inspection
duplicates = df.duplicated()
print("Duplicated rows:")
print(df[duplicates])

# Remove duplicated rows from the DataFrame
df = df.drop_duplicates()
# Print the new shape after removing duplicates
print("New shape after removing duplicates (rows, columns):")
print(df.shape)

# Calculate the length (number of words) for context, question, and answer columns
# This helps to understand the distribution of text lengths in the dataset
df['context_length'] = df['context'].apply(lambda x: len(str(x).split()))
df['question_length'] = df['question'].apply(lambda x: len(str(x).split()))
df['answer_length'] = df['answer'].apply(lambda x: len(str(x).split()))

# Print statistics for the new length columns
print("Statistics for context, question, and answer lengths:")
print(df[['context_length', 'question_length', 'answer_length']].describe())

# Create histograms for the distributions of context, question, and answer lengths
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot context length distribution
sns.histplot(df['context_length'], bins=50, kde=True, color='#66b3ff', ax=axes[0])
axes[0].set_title("Context Length Distribution", fontsize=12, weight='bold')
axes[0].set_xlabel("Number of Words in Context")
axes[0].set_ylabel("Frequency")
axes[0].set_xlim(0, df['context_length'].quantile(0.95))  # Avoid extreme values

# Plot question length distribution
sns.histplot(df['question_length'], bins=30, kde=True, color='#99cc00', ax=axes[1])
axes[1].set_title("Question Length Distribution", fontsize=12, weight='bold')
axes[1].set_xlabel("Number of Words in Question")
axes[1].set_ylabel("Frequency")
axes[1].set_xlim(0, df['question_length'].quantile(0.99))

# Plot answer length distribution
sns.histplot(df['answer_length'], bins=40, kde=True, color='#ff9966', ax=axes[2])
axes[2].set_title("Answer Length Distribution", fontsize=12, weight='bold')
axes[2].set_xlabel("Number of Words in Answer")
axes[2].set_ylabel("Frequency")
axes[2].set_xlim(0, df['answer_length'].quantile(0.99))

plt.tight_layout()
sns.despine()

# Save the figure with the distributions to the reports/figures directory
plt.savefig('../reports/figures/length_distributions.png')

plt.show()
plt.close()

# Save the EDA summary statistics to a CSV file for further analysis
eda_summary = df.describe(include='all')
eda_summary.to_csv('../reports/eda_summary.csv', index=False)
