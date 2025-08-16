import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit App UI
st.set_page_config(page_title="📩 SMS Spam Detection", page_icon="📱", layout="centered")

st.markdown("<h1 style='text-align:center; color:#FF4B4B;'>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:grey;'>Check if a message is spam or not instantly!</p>", unsafe_allow_html=True)


input_sms = st.text_area("✍️ Enter your message here", height=150)

if st.button("🔍 Check Message"):
    if input_sms.strip() == "":
        st.warning("Please enter a message to check.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict
        result = model.predict(vector_input)[0]
        
        # Show result with style
        if result == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")

st.markdown("<hr style='margin-top:50px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:13px; color:grey;'>Made with by Abhay</p>", unsafe_allow_html=True)
