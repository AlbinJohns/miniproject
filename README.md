# **Mini Project: Boosting Creative Diversity of Generative AI Models via Human Thought Process Mimicry**

This notebook demonstrates an exploration of generative AI models, particularly using Google's Gemini API to generate creative text responses. The project focuses on different techniques for generating creative and contextually rich content, such as multi-level query segmentation (MQS) and random temperature manipulation, and compares the results with a base model.

Additionally, the notebook evaluates the quality of generated content through entropy and intra-cosine similarity metrics to analyze the randomness and similarity between multiple outputs.

## Introduction

This project delves into methods for enhancing the creative diversity of generative AI models by drawing inspiration from how human authors approach creativity. The key technique used here is the **Multi-Level Query Segmentation (MQS)** system, which aims to replicate the way authors take inspiration from existing work.

In addition to MQS, we also employ **randomized temperature manipulation**, a technique designed to mirror the effect of an author's mood on the creative process. For example, if the author feels rebellious or experimental, the model uses a higher temperature setting to promote more diverse and daring outputs. Conversely, a more cautious or neutral mood leads to a lower temperature, producing safer and more predictable results.

The core objective of this project is to introduce human-like factors, such as inspiration and emotional states, into the creative decision-making process of AI, ultimately enabling the generation of content that feels more diverse, dynamic, and in tune with human creativity.

## Methodology

1. **Multi-Level Query Segmentation (MQS)**:  
   The MQS system divides prompts into segments, each representing a different aspect of the inspiration process. These segments allow the AI to take cues from various elements of existing work, effectively replicating how human authors synthesize ideas.

2. **Randomized Temperature Manipulation**:  
   To simulate mood variations in authors, we implement randomized temperature changes in the text generation process. A higher temperature setting corresponds to a more experimental or creative mindset, encouraging less deterministic and more diverse outputs. A lower temperature setting represents a more focused and conventional mindset, producing structured and consistent results.


## Key Components
1. **Dependencies and Configuration:**
   - Installation of necessary libraries and configuration of Google's Gemini model.

2. **Response Generation Techniques:**
   - **Base Model:** A simple text generation using Gemini's generative model.
   - **Multi-Level Query Segmentation (MQS):** Enhancing creativity by prompting the model with contemporary works for inspiration.
   - **Randomized Temperature Manipulation:** Varying the temperature parameter to influence creativity and variability.
   - **Creative Generation:** A hybrid approach combining MQS and temperature manipulation to generate highly creative and diverse outputs.

3. **Quality Analysis of Generated Text:**
   - **Entropy Calculation:** Measuring the unpredictability of the generated text to gauge its creativity.
   - **Intra-Cosine Similarity:** Analyzing the similarity between generated outputs using cosine similarity, variance, and interquartile range (IQR).

4. **Output Comparison and Visualization:**
   - The notebook concludes with cosine similarity comparisons and graphical representations of the variance and IQR of the generated text outputs.

This project serves as an exploration into how various creative generation methods can be quantified and compared for further research and analysis.

## <ins>**Collaborators**</ins>
*   Albin Johns
*   Ashlyn S Pothen
*   Mathew Joe

Hosted at: https://miniproject-creative-gen.streamlit.app/
