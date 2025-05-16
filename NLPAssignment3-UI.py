import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from html import escape
import pandas as pd

# Load model and tokenizer
@st.cache_resource
def load_model():
    # Load model
    model = DistilBertForSequenceClassification.from_pretrained("my_distilbert_model_version11")

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("my_distilbert_model_version11")
    return model, tokenizer

model, tokenizer = load_model()
id2label = model.config.id2label

def classify_text_with_attention(text, model, tokenizer, id2label=None, max_length=115, top_k=5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length, return_attention_mask=True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze()
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        # Use average of attention weights from last layer
        attentions = outputs.attentions[-1]  # last layer attention
        attn_weights = attentions.mean(dim=1).squeeze(0)  # average across all heads

        # Get attention from [CLS] token to all others
        cls_attention = attn_weights[0][1:-1]  # skip CLS and SEP

        # Token decode and highlight
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[1:-1]
        token_scores = cls_attention.cpu().numpy()

        # Normalize scores for color intensity
        norm_scores = (token_scores - token_scores.min()) / (token_scores.max() - token_scores.min() + 1e-9)

        highlighted = ""
        for token, score in zip(tokens, norm_scores):
            color = int((1 - score) * 255)  # higher attention = darker
            token_text = escape(token.replace("##", ""))
            highlighted += f"<span style='background-color: rgba(255, 200, 0, {score:.2f}); padding: 2px; border-radius: 3px'>{token_text}</span> "

    label = id2label[pred_class] if id2label else str(pred_class)
    problist = probs.tolist()
    return label, confidence, highlighted, problist

# Streamlit UI
st.title("Ticket Classification using BERT | for IT Helpdesks")

input_text = st.text_area("Enter email text to classify", height=200)

if st.button("Classify"):
    if input_text.strip():
        label, confidence, highlighted_text,probs = classify_text_with_attention(input_text, model, tokenizer, id2label)

        st.success(f"**Predicted Label:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # st.subheader("Highlighting text based on Attention")
        # st.markdown(f"<div style='line-height: 1.8;'>{highlighted_text}</div>", unsafe_allow_html=True)

        # Prepare data for chart
        prob_df = pd.DataFrame({
            "Label": [id2label[i] for i in range(len(probs))],
            "Confidence": probs
        })

        # Display bar chart
        st.subheader("Class Confidence Scores")
        st.bar_chart(prob_df.set_index("Label"))

    else:
        st.warning("Please enter some text.")
