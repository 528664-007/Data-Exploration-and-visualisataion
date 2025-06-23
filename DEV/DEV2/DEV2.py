# Step 1: Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
import os

# Ensure required resources are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 2: Load the dataset from file path
# 📁 Change this path to your local file location
file_path = 'C:\\Users\\hp\\Documents\\DEV\\spam.csv'  # Example: 'C:/Users/YourName/Documents/spam.csv'

# Load only the necessary columns
df = pd.read_csv(file_path, encoding='latin-1')[['v1', 'v2']]
df.columns = ['Label', 'Message']

# Step 3: Understand the dataset structure
print("\n📊 Head of dataset:")
print(df.head())

print("\n🔍 Dataset Info:")
print(df.info())

print("\n📈 Summary Statistics:")
print(df.describe())

# Step 4: Check for missing values
print("\n🧹 Missing values:")
print(df.isnull().sum())

# Step 5: Analyze class distribution
print("\n📊 Class Distribution:")
print(df['Label'].value_counts())

sns.countplot(x='Label', data=df)
plt.title("Class Distribution: Spam vs Ham")
plt.show()

# Step 6: Visualize message lengths
df['Message_Length'] = df['Message'].apply(len)

plt.figure(figsize=(10,6))
sns.histplot(data=df, x='Message_Length', hue='Label', bins=50, kde=True, palette='husl')
plt.title("Distribution of Message Lengths")
plt.show()

# Step 7: Word frequency analysis
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

# Create tokens for spam and ham
df['Tokens'] = df['Message'].apply(preprocess_text)

spam_words = df[df['Label'] == 'spam']['Tokens'].sum()
ham_words = df[df['Label'] == 'ham']['Tokens'].sum()

spam_freq = Counter(spam_words).most_common(20)
ham_freq = Counter(ham_words).most_common(20)

# Plotting top spam words
spam_df = pd.DataFrame(spam_freq, columns=['Word', 'Freq'])
plt.figure(figsize=(10,5))
sns.barplot(data=spam_df, x='Freq', y='Word', palette='Reds_r')
plt.title("Top 20 Spam Words")
plt.show()

# Plotting top ham words
ham_df = pd.DataFrame(ham_freq, columns=['Word', 'Freq'])
plt.figure(figsize=(10,5))
sns.barplot(data=ham_df, x='Freq', y='Word', palette='Blues_r')
plt.title("Top 20 Ham Words")
plt.show()

# Step 8: Word clouds
spam_text = ' '.join([' '.join(tokens) for tokens in df[df['Label'] == 'spam']['Tokens']])
ham_text = ' '.join([' '.join(tokens) for tokens in df[df['Label'] == 'ham']['Tokens']])

# Spam word cloud
spam_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(spam_text)
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')
plt.title("📨 Spam Word Cloud")
plt.show()

# Ham word cloud
ham_wc = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(ham_text)
plt.figure(figsize=(10, 5))
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')
plt.title("✉️ Ham Word Cloud")
plt.show()

# Step 9: Correlation (Message_Length vs Label)
plt.figure(figsize=(6,4))
sns.boxplot(x='Label', y='Message_Length', data=df)
plt.title("Boxplot - Message Length by Label")
plt.show()

# Step 10: Conclusion
print("\n📌 EDA Conclusions:")
print("- Spam messages are generally longer than ham messages.")
print("- Words like 'free', 'win', 'urgent' are frequent in spam.")
print("- Ham messages contain casual/conversational words like 'ok', 'come', 'home'.")
print("- Message length is a useful feature for spam detection.")
