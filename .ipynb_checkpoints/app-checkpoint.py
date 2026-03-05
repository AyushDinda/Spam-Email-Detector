import streamlit as st
import pickle

# Load trained model
model = pickle.load(open("spam_model.pkl", "rb"))

# Page settings
st.set_page_config(page_title="AI Spam Detector", page_icon="📧", layout="centered")

# Header
st.title("📧 AI Email Spam Detector")
st.markdown(
"""
Detect whether an email message is **Spam** or **Not Spam** using a Machine Learning model.
"""
)

st.divider()

# Input box
message = st.text_area("✉️ Enter Email Message", height=150)

# Button
if st.button("🔍 Analyze Message"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:

        prediction = model.predict([message])
        probability = model.predict_proba([message])

        spam_prob = probability[0][1] * 100

        st.divider()

        # Result
        if prediction[0] == 1:
            st.error("🚨 SPAM DETECTED")
        else:
            st.success("✅ NOT SPAM")

        # Probability meter
        st.subheader("Spam Probability")
        st.progress(int(spam_prob))

        st.write(f"**Confidence:** {spam_prob:.2f}%")

st.divider()

# Example messages
st.subheader("Try Example Messages")

st.markdown(
"""
**Spam**
"""
)
st.code("Congratulations! You won a free iPhone. Click here to claim.")
st.markdown(
"""
**Not-Spam**
"""
)
st.code("Hi Ayush, Can we meet tomorrow for the project discussion?")