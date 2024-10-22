import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report

# Load the trained model
model = joblib.load('depression_model.pkl')

# Read the model accuracy from the text file
with open('model_accuracy.txt', 'r') as f:
    model_accuracy = f.read().strip()  # Read and strip any extra whitespace

# Title of the app
st.title("🧠 Depression Prediction App")

# Display overall model accuracy
st.markdown(f"### Overall Model Accuracy: **{model_accuracy}**")

# Usage guidelines
st.markdown("""
### How to Use This App:
1. **Enter Text**: In the text area provided, enter the text you want to analyze for depression.
2. **Predict**: Click the **Predict** button to get the prediction.
3. **View Results**: The app will display whether the text indicates depression or not, along with the confidence levels.
4. **Visualizations**: You can view the class distribution and confusion matrix to understand the model's performance better.
5. **Model Analysis**: Click on the **View Model Analysis** section to see detailed insights about the model and its training process.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
st.sidebar.write("Use this app to predict depression based on text input.")

# Input text area for user input
user_input = st.text_area("📝 Enter your text here:", height=150)

# Button for prediction
if st.button("🔍 Predict"):
    if user_input:
        # Make prediction using the loaded model
        prediction = model.predict([user_input])
        probability = model.predict_proba([user_input])[0]
        
        # Display the result
        if prediction[0] == 0:
            st.success(f"The text indicates **NO depression**. (Confidence: {probability[0]:.2f})")
        else:
            st.warning(f"The text indicates **DEPRESSION**. (Confidence: {probability[1]:.2f})")
        
        # Display probabilities
        st.write("### Prediction Probabilities:")
        st.write(f"- **No Depression**: {probability[0]:.4f}")
        st.write(f"- **Depression**: {probability[1]:.4f}")
    else:
        st.error("⚠️ Please enter some text for prediction.")

# # Display class distribution image
# st.subheader("📊 Class Distribution")
# class_dist_img = Image.open('class_distribution.png')
# st.image(class_dist_img, caption='Class Distribution', use_column_width=True)

# # Display confusion matrix image
# st.subheader("📉 Confusion Matrix")
# confusion_matrix_img = Image.open('confusion_matrix.png')
# st.image(confusion_matrix_img, caption='Confusion Matrix', use_column_width=True)

# # Display classification report
# st.subheader("📋 Classification Report")
# classification_report_text = """
# - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. 
# - **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
# - **F1 Score**: The weighted average of Precision and Recall.
# - **Support**: The number of actual occurrences of the class in the specified dataset.
# """
# st.write(classification_report_text)

# Additional Analysis Section (Hidden by default)
with st.expander("🔍 View Model Analysis", expanded=False):
    st.write("""
    The model was trained on a balanced dataset containing an equal number of samples from both classes (depression and no depression). 
    This approach helps to mitigate bias towards the majority class and improves the model's ability to generalize.

    ### Key Metrics:
    - **Model Accuracy**: The model achieved an accuracy of approximately XX% on the test set, indicating its effectiveness in predicting depression.
    - **Confusion Matrix**: The confusion matrix visualizes the performance of the model, showing the true positive, true negative, false positive, and false negative rates. 
      This helps in understanding where the model is making errors.

    ### Visualizations:
    - The **Class Distribution** plot shows the balance between the two classes in the dataset, confirming that the dataset was balanced before training.
    - The **Confusion Matrix** plot provides insights into the model's performance, allowing us to see how many instances were correctly classified versus misclassified.

    ### Conclusion:
    Overall, the model demonstrates a good ability to predict depression based on the input text. Continuous improvements can be made by exploring different algorithms, tuning hyperparameters, and incorporating more diverse datasets.
    """)