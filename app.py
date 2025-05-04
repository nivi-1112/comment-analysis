import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import re

st.set_page_config(page_title="YouTube Comment Analyzer", layout="wide")
st.title("🎯 YouTube Comment Sentiment & Emotion Analysis")

comment_input = st.text_area("Paste YouTube comments below (one per line):")

if st.button("Analyze"):
    if not comment_input.strip():
        st.warning("Please enter some comments to analyze.")
    else:
        comments = [c.strip() for c in comment_input.split("\n") if c.strip()]
        df = pd.DataFrame(comments, columns=["Comment"])

        # Preprocess
        df["Clean"] = df["Comment"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))

        # Sentiment
        df["Polarity"] = df["Clean"].apply(lambda x: TextBlob(x).sentiment.polarity)
        df["Sentiment"] = df["Polarity"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

        # Simple Emotion
        def detect_emotion(text):
            if "happy" in text or "great" in text:
                return "Joy"
            elif "sad" in text or "cry" in text:
                return "Sadness"
            elif "angry" in text or "mad" in text:
                return "Anger"
            elif "fear" in text or "scared" in text:
                return "Fear"
            else:
                return "Neutral"

        df["Emotion"] = df["Clean"].apply(detect_emotion)

        # Topic Modeling
        vectorizer = CountVectorizer(stop_words="english", max_features=1000)
        dtm = vectorizer.fit_transform(df["Clean"])
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic in lda.components_:
            top_words = [feature_names[i] for i in topic.argsort()[-5:]]
            topics.append(", ".join(top_words))

        df["Topic"] = [topics[i.argmax()] for i in lda.transform(dtm)]

        # Show Table
        st.subheader("📊 Analysis Results")
        st.dataframe(df[["Comment", "Sentiment", "Emotion", "Topic"]])

        # Visuals
        st.subheader("🔍 Sentiment Distribution")
        st.bar_chart(df["Sentiment"].value_counts())

        st.subheader("🔍 Emotion Distribution")
        st.bar_chart(df["Emotion"].value_counts())

        st.subheader("☁️ Word Cloud")
        text = " ".join(df["Clean"])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

        st.subheader("💡 Topics Identified")
        st.write(topics)
