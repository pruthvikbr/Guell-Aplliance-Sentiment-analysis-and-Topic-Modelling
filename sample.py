import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import pyLDAvis
import pyLDAvis.lda_model
import pyLDAvis.gensim_models as gensimvis
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation



data=pd.read_excel("/Users/pruthvikbr/Documents/UCI/Machine Learning/Assignment 5/Guell_Appliances_Data.xlsx")

# Check for missing values in the important columns
data.isnull().sum()

# Fill missing values in the 'Review' column with an empty string and in the 'Rating' column with the median rating
data['Review'].fillna('', inplace=True)
data['Rating'].fillna(data['Rating'].median(), inplace=True)

# Verify that there are no missing values now
data.isnull().sum()

# Perform sentiment analysis
sentiment_scores = []
for review in data['reviewText']:
    blob = TextBlob(str(review))
    sentiment_scores.append(blob.sentiment.polarity)

# Add sentiment scores to the DataFrame
data['Sentiment_Score'] = sentiment_scores

# Calculate correlation between sentiment score and overall rating
correlation = data['overall'].corr(data['Sentiment_Score'])
print(f'Correlation between Rating and Sentiment Score: {correlation}')

# Plot the correlation
plt.scatter(data['overall'].values,data['Sentiment_Score'].values)
plt.title('Correlation between Rating and Sentiment Score')
plt.xlabel('Rating')
plt.ylabel('Sentiment Score')
plt.show()

#Frequency plot
sns.histplot(data['Sentiment_Score'],kde=True,bins=20)
plt.title("Distrubution of Sentiment score")
plt.xlabel("sentiment score")
plt.ylabel("Frequency")
plt.show()


# Perform topic modeling
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(data['reviewText'])

# Initialize LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(dtm)

# Get top words for each topic
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

# Display top words for each topic
num_top_words = 5
tf_feature_names = vectorizer.get_feature_names_out()
display_topics(lda, tf_feature_names, num_top_words)


# Convert the document-term matrix to a gensim corpus
corpus = gensim.matutils.Sparse2Corpus(dtm, documents_columns=False)

dictionary = Dictionary.from_corpus(corpus, id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
# Convert scikit-learn LDA model to gensim LdaModel
gensim_lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()),
    num_topics=5,  # Change to the number of topics you used in LDA
    alpha='auto'
)

# Visualize the LDA model
vis_data = gensimvis.prepare(gensim_lda_model, corpus, dictionary=dictionary)
pyLDAvis.display(vis_data)
pyLDAvis.save_html(vis_data, '/Users/pruthvikbr/Documents/UCI/Machine Learning/Assignment 5/doc_name.html')

