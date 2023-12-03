import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer  # Ensure you import this if you're using it in your Colab notebook

# Load your pre-trained model and TF-IDF vectorizer using pickle
# Replace 'your_model_path.pkl' and 'your_tfidf_vectorizer_path.pkl' with the actual file paths of your trained model and TF-IDF vectorizer
with open('trained_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vector.sav', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to predict polarity
def predict_polarity(sentence):
    new_input_tfidf = tfidf_vectorizer.transform([sentence])
    predicted_polarity = model.predict(new_input_tfidf)
    return predicted_polarity[0]

# Streamlit app
def main():
    st.title('Sentence Polarity Prediction')
    sentence = st.text_input('Enter a sentence:')
    
    if st.button('Predict'):
        if sentence:
            polarity = predict_polarity(sentence)
            if polarity == 1:
                st.write("The predicted polarity for the input is positive")
            else:
                st.write("The predicted polarity for the input is negative")
        else:
            st.write('Please enter a sentence to predict its polarity.')

if __name__ == '__main__':
    main()
