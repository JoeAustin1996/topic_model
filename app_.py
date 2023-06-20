!pip install gensim
import streamlit as st
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords (run once)
nltk.download('stopwords')

# Load the data
data = pd.read_csv("IndiaMart sector.csv")

# Preprocess the categories column
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

data['categories'] = data['categories'].apply(preprocess_text)

# Create a tab for entering the desired category
st.title("Topic Analysis")
desired_category = st.text_input("Enter Desired Category")

# Filter the dataframe based on the desired category
filtered_df = data[data['categories'].apply(lambda x: desired_category in x)]

# Check if any records match the desired category
if filtered_df.empty:
    st.warning("No records found for the desired category.")
else:
    # Split the data into training and holdout sets
    train_data, holdout_data = train_test_split(filtered_df, test_size=0.2, random_state=42)

    # Create a dictionary and corpus from the training data
    documents = train_data['categories'].tolist()
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Train an LDA model on the training data
    num_topics = 10  # Specify the number of topics
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    # Evaluate the model on the holdout set using perplexity
    holdout_documents = holdout_data['categories'].tolist()
    holdout_corpus = [dictionary.doc2bow(doc) for doc in holdout_documents]
    perplexity = lda_model.log_perplexity(holdout_corpus)

    # Evaluate the model on the holdout set using coherence
    coherence_model = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()

    # Get the unique values of the sector and super_sector columns for the filtered data
    unique_sectors = filtered_df['sector'].unique()
    unique_super_sectors = filtered_df['super_sector'].unique()

    # Calculate topic diversity
    topic_diversity = len(lda_model.get_topics())

    # Calculate topic interpretability
    top_topics = lda_model.top_topics(corpus)
    topic_interpretability = sum([topic[1] for topic in top_topics]) / len(top_topics)

    # Perform hyperparameter tuning
    param_grid = {'num_topics': [5, 10, 15, 20]}
    best_score = -float('inf')
    for num_topics in param_grid['num_topics']:
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
        score = coherence_model.get_coherence(lda_model)
        if score > best_score:
            best_score = score

    # Display the results
    st.subheader("Results")
    st.text("Desired Category: " + desired_category)
    st.text("Perplexity: " + str(perplexity))
    st.text("Coherence: " + str(coherence))
    st.text("Topic Diversity: " + str(topic_diversity))
    st.text("Topic Interpretability: " + str(topic_interpretability))
    st.text("Best Score: " + str(best_score))
    st.text("Unique Sectors: " + ", ".join(unique_sectors[:5]))
    st.text("Unique Super Sectors: " + ", ".join(unique_super_sectors[:5]))
