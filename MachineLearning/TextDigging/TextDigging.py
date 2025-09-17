# 情感分析
import pandas as pd
from sklearn.model_selection import train_test_split
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora
from textblob import TextBlob  # 导入TextBlob进行情感分析

if __name__ == '__main__':
    # 读取CSV文件
    file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Text_Digging\text.csv"
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # 分词函数
    def tokenize(text):
        return [token for token in simple_preprocess(text) if token not in STOPWORDS]

    # 对文本进行分词，并去除停用词
    df['tokens'] = df['Content'].apply(lambda x: tokenize(x))

    # 划分训练集和测试集
    train_data, test_data = train_test_split(df['tokens'], test_size=0.1, random_state=42)

    # 创建词典
    dictionary = corpora.Dictionary(train_data)

    # 创建文档-词频矩阵
    train_corpus = [dictionary.doc2bow(tokens) for tokens in train_data]
    test_corpus = [dictionary.doc2bow(tokens) for tokens in test_data]

    # 构建LDA模型
    lda_model = gensim.models.LdaModel(train_corpus, num_topics=2, id2word=dictionary, passes=30)

    # 计算情感得分
    def calculate_sentiment(text):
        return TextBlob(text).sentiment.polarity

    # 计算文档情感得分并加权
    doc_sentiment_scores = []
    for doc in df['Content']:
        doc_sentiment = calculate_sentiment(doc)
        doc_topic_probs = lda_model[dictionary.doc2bow(tokenize(doc))]
        weighted_sentiment = sum(topic_prob * doc_sentiment for _, topic_prob in doc_topic_probs)
        doc_sentiment_scores.append(weighted_sentiment)

    # 创建DataFrame保存每个文档的情感得分
    sentiment_df = pd.DataFrame({'Document': range(len(doc_sentiment_scores)), 'Sentiment_Score': doc_sentiment_scores})
    sentiment_df.to_csv('document_sentiment_scores.csv', index=False)

    # 计算主题总体情感值得分
    topic_sentiment_scores = {}

    for i in range(len(lda_model.get_topics())):
        topic_sentiment_scores[i] = 0  # 初始化每个主题的情感值总和

    for i, doc in enumerate(df['Content']):
        doc_sentiment = calculate_sentiment(doc)
        doc_topic_probs = lda_model[dictionary.doc2bow(tokenize(doc))]

        for topic, prob in doc_topic_probs:
            topic_sentiment_scores[topic] += doc_sentiment * prob

    # 创建DataFrame保存每个主题的总体情感值得分
    topic_sentiment_df = pd.DataFrame({'Topic': list(topic_sentiment_scores.keys()), 'Overall_Sentiment_Score': list(topic_sentiment_scores.values())})
    topic_sentiment_df.to_csv('topic_overall_sentiment_scores.csv', index=False)

    # 计算困惑度
    perplexity = lda_model.log_perplexity(test_corpus)
    perplexity_df = pd.DataFrame({'Perplexity': [perplexity]})

    # 计算主题一致性
    coherence_model = gensim.models.CoherenceModel(model=lda_model, texts=test_data, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    coherence_df = pd.DataFrame({'Coherence_Score': [coherence]})

    # 创建DataFrame保存主题词分布
    topic_word_distribution = {'Topic': [], 'Word': [], 'Probability': []}
    topics = lda_model.show_topics(num_topics=-1, formatted=False)
    for topic in topics:
        topic_id = topic[0]
        for word, prob in topic[1]:
            topic_word_distribution['Topic'].append(topic_id)
            topic_word_distribution['Word'].append(word)
            topic_word_distribution['Probability'].append(prob)

    topic_word_distribution_df = pd.DataFrame(topic_word_distribution)
    topic_word_distribution_df.to_csv('topic_word_distribution.csv', index=False)

    # 创建DataFrame保存文档对主题的分配
    document_topic_assignment = {'Document': [], 'Topic': [], 'Probability': []}
    for i, doc in enumerate(test_corpus):
        doc_topic_probs = lda_model.get_document_topics(doc)
        for topic, prob in doc_topic_probs:
            document_topic_assignment['Document'].append(i)
            document_topic_assignment['Topic'].append(topic)
            document_topic_assignment['Probability'].append(prob)

    document_topic_assignment_df = pd.DataFrame(document_topic_assignment)
    document_topic_assignment_df.to_csv('document_topic_assignment.csv', index=False)

# 文本分类
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Read CSV file
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Text_Digging\text.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Assuming 'Content' column contains text data and 'label' column contains labels

# Preprocessing text data
max_words = 2000
max_len = 200

# Remove English stopwords
stopwords = set(ENGLISH_STOP_WORDS)
df['Content'] = df['Content'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords]))

# Tokenization and padding
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['Content'])
sequences = tokenizer.texts_to_sequences(df['Content'])
X = pad_sequences(sequences, maxlen=max_len)

# Preparing labels
y = pd.get_dummies(df['label']).values

# Splitting data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model definition
model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(df['label'].unique()), activation='softmax'))  # Assuming labels are categorical

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Predicting probabilities for all samples
predicted_probabilities = model.predict(X)

# Selecting the class with highest probability as prediction
predicted_labels = predicted_probabilities.argmax(axis=1)

# Adding predicted labels to the DataFrame
df['predicted_label'] = predicted_labels

# Save classification results to CSV
classification_results_path = "classification_results.csv"
df.to_csv(classification_results_path, index=False)

# Output predicted probabilities
predicted_probabilities_df = pd.DataFrame(predicted_probabilities, columns=[f"Probability_{label}" for label in df['label'].unique()])
predicted_probabilities_df.to_csv('predicted_probabilities.csv', index=False)

# Plot learning curves
plt.plot(history.history['accuracy'], color = '#F1948A' , label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color = '#A2D9CE', label='Validation Accuracy')
plt.title('Model Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(frameon=False)
plt.savefig('learning_curves.svg')
plt.show()

# Output word cloud for each category
def generate_wordcloud(text, filename):
    wordcloud = WordCloud(width=2100, height=1000, background_color='white').generate(text)
    plt.figure(figsize=(30, 18))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename, format='svg')
    plt.show()

for label in df['label'].unique():
    text = ' '.join(df[df['label'] == label]['Content'])
    generate_wordcloud(text, f'{label}_wordcloud.svg')
