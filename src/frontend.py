import streamlit as st
import pandas as pd
import os
import sys
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.prediction_pipeline import CustomData, PredictPipeline



# Load precomputed cluster keywords
keywords_path = r"C:\Users\karthikeya\New_Delhi_Reviews\artifacts\cluster_keywords.json"
with open(keywords_path, "r") as f:
    cluster_keywords = json.load(f)


def generate_wordcloud_from_keywords(keywords, cluster_id):
    """
    Generate a word cloud using precomputed keywords.
    """
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(keywords)

    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f"Word Cloud for Cluster {cluster_id}", fontsize=16)
    ax.axis('off')

    return fig

def predict_datapoint():
    st.title('Text Clustering for Reviews')

    # Input fields
    text = st.text_input('Enter review text:')

    # Prediction button
    if st.button('Predict'):
        if text.strip():  # Ensure text is not empty
            try:
                # Create CustomData instance
                data = CustomData(review_full=text)

                # Get DataFrame and predict
                pred_df = data.get_data_as_data_frame()
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(pred_df)
                predicted_cluster = str(results[0])  # Ensure cluster ID is string

                st.success(f'Predicted Cluster: {predicted_cluster}')

                # Display word cloud for the predicted cluster
                if predicted_cluster in cluster_keywords:
                    st.subheader(f"Word Cloud for Cluster {predicted_cluster}")
                    wordcloud_fig = generate_wordcloud_from_keywords(
                        cluster_keywords[predicted_cluster], predicted_cluster
                    )
                    st.pyplot(wordcloud_fig)
                else:
                    st.warning("No keywords available for the predicted cluster.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some text before predicting.")

# Run the app
if __name__ == "__main__":
    predict_datapoint()
