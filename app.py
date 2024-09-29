import streamlit as st
import google.generativeai as genai
import textwrap
import random
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# Function Definitions
# =============================

def to_markdown(text):
    """
    Convert text to markdown format by replacing bullet points and adding blockquote.
    """
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def calculate_entropy(sequence):
    """
    Calculate the entropy of a given sequence of characters.
    """
    counts = Counter(sequence)
    total_count = len(sequence)
    entropy = 0.0
    for symbol, count in counts.items():
        probability = count / total_count
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_cosine_similarity(text1, text2):
    """
    Calculate the cosine similarity between two texts using TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def genai_generation(prompt):
    """
    Generate content using Google Generative AI based on the provided prompt.
    """
    response = model.generate_content(prompt)
    summary = response.text
    return summary

def creative_generation(prompt):
    """
    Generate creative content by modifying the prompt and using temperature for variability.
    """
    # Enhance the prompt for generating reference dialogues or scenes
    enhanced_prompt = (
        prompt + " Actually no don't give me that but give me 20 short dialogues or scenes "
        "snippets of this same genre, theme or subject that convey deep emotions that I can use "
        "as a reference for this subject, take the content from both well-known and obscure works "
        "of literature"
    )
    
    # Random temperature for variability in generation
    temp_inspir = random.uniform(0.0, 2.0)
    st.write(f"🌀 **Temperature for Inspiration:** {temp_inspir:.1f}")
    
    # Generate inspirational examples
    response = model.generate_content(
        enhanced_prompt,
        generation_config=genai.types.GenerationConfig(temperature=temp_inspir)
    )
    examples = response.text
    
    # Modify the prompt to incorporate the generated examples
    final_prompt = (
        prompt + " Take inspiration about characters (name, attitude, and locations and other Named Entities) "
        "and scenarios from the following contemporary works and replicate the emotional dialogues or scenes "
        "of these examples while writing and copy or take inspiration from its writing style, randomly take "
        "inspiration (names and scenarios) from any of the given examples below (or mixture of them) and add "
        "in your creativity as well to ensure the characters (names), locations and other named entities you "
        "generate are always truly unique: " + examples
    )
    
    # Random temperature for final content generation
    temp_final = random.uniform(0.0, 2.0)
    st.write(f"🎨 **Temperature for Final Output:** {temp_final:.1f}")
    
    # Generate the final creative content
    response = model.generate_content(
        final_prompt,
        generation_config=genai.types.GenerationConfig(temperature=temp_final)
    )
    text = response.text
    return text

# =============================
# Configuration
# =============================

# Configure Google Generative AI with your API key
# 🔒 **Security Note:** It's recommended to store API keys securely using environment variables or Streamlit secrets.
genai.configure(api_key="AIzaSyCvIfy6QXY8i9x9lzVEP92SZRdIJLEJMV4")
model = genai.GenerativeModel('gemini-1.5-flash')

# =============================
# Streamlit App Layout
# =============================

# Set the title and description with icons for better UX
st.set_page_config(page_title="GenAI & Creative Generator", page_icon="🤖", layout="wide")
st.title("🤖 Generative AI & 🎨 Creative Generation Comparison")
st.write("Compare results from **GenAI** and **Creative Generation** to analyze their outputs.")

# Input prompt from user with an input icon
prompt = st.text_input("✏️ Enter your prompt here:")

# Process and display results only if a prompt has been entered
if prompt:
    # Display a spinner while generating content
    with st.spinner("🔄 Generating content... Please wait."):
        # Generate content using both methods
        genai_result = genai_generation(prompt)
        creative_result = creative_generation(prompt)
    
    # Calculate entropy for both results
    genai_entropy = calculate_entropy(genai_result)
    creative_entropy = calculate_entropy(creative_result)
    
    # Calculate cosine similarity between the two results
    cosine_similarity_score = calculate_cosine_similarity(genai_result, creative_result)
    
    # Convert results to markdown format for better readability
    genai_result_md = to_markdown(genai_result)
    creative_result_md = to_markdown(creative_result)
    
    # Display cosine similarity with an info icon
    st.markdown(f"### 🔍 Cosine Similarity between outputs: **{cosine_similarity_score:.4f}**")
    
    # Use Streamlit's column feature to display results side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 **GenAI Generation**") 
        st.write(f"📊 **Entropy:** {genai_entropy:.4f}")
        st.markdown(genai_result_md)
       
    with col2:
        st.subheader("🎨 **Creative Generation**")
        st.write(f"📊 **Entropy:** {creative_entropy:.4f}")
        st.markdown(creative_result_md)
    
    # Optional: Provide a download button for the results
    st.markdown("---")
    st.download_button(
        label="💾 Download GenAI Result",
        data=genai_result,
        file_name='genai_result.txt',
        mime='text/plain'
    )
    st.download_button(
        label="💾 Download Creative Result",
        data=creative_result,
        file_name='creative_result.txt',
        mime='text/plain'
    )
