import streamlit as st
import asyncio
import sys
import os
import tempfile
import json
import requests
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


st.set_page_config(page_title="Gender Bias Analyzer", layout="wide")

MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".gender_bias_analyzer_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


DEFAULT_EXAMPLES = [
    ("He is bossy", "She is bossy"),
    ("He is emotional", "She is emotional"),
    ("The boy is intelligent", "The girl is intelligent"),
    ("He is aggressive", "She is aggressive"),
    ("He is assertive", "She is assertive"),
    ("He is funny", "She is funny"),
    ("He is creative", "She is creative"),
    ("He is confident", "She is confident"),
]


MODELS = {
    "Twitter-RoBERTa": {
        "path": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "load_func": "load_roberta_model",
        "tokenizer_func": "load_roberta_tokenizer"
    },
    "BERTweet": {
        "path": "finiteautomata/bertweet-base-sentiment-analysis",
        "load_func": "load_bertweet_model",
        "tokenizer_func": "load_bertweet_tokenizer"
    },
    "DistilBERT-base": {
        "path": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        "load_func": "load_distilbert_model",
        "tokenizer_func": "load_distilbert_tokenizer"
    }
}


if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None

def load_roberta_model():
    path = MODELS["Twitter-RoBERTa"]["path"]
    model_dir = os.path.join(MODEL_CACHE_DIR, "twitter-roberta")
    

    if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        try:
            return AutoModelForSequenceClassification.from_pretrained(model_dir)
        except Exception as e:
            st.warning(f"Error loading cached model, downloading fresh: {str(e)}")
    
    os.makedirs(model_dir, exist_ok=True)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.save_pretrained(model_dir)
        return model
    except Exception as e:
        st.error(f"Failed to load Twitter-RoBERTa model: {str(e)}")
        st.stop()

def load_bertweet_model():
    path = MODELS["BERTweet"]["path"]
    model_dir = os.path.join(MODEL_CACHE_DIR, "bertweet")
    
    if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        try:
            return AutoModelForSequenceClassification.from_pretrained(model_dir)
        except Exception as e:
            st.warning(f"Error loading cached model, downloading fresh: {str(e)}")
    

    os.makedirs(model_dir, exist_ok=True)
    try:
        tmp_dir = tempfile.mkdtemp()
        
        config_url = f"https://huggingface.co/{path}/resolve/main/config.json"
        config_path = os.path.join(tmp_dir, "config.json")
        with open(config_path, "wb") as f:
            f.write(requests.get(config_url).content)
        
        weights_url = f"https://huggingface.co/{path}/resolve/main/tf_model.h5"
        weights_path = os.path.join(tmp_dir, "tf_model.h5")
        with open(weights_path, "wb") as f:
            f.write(requests.get(weights_url).content)
        

        model = AutoModelForSequenceClassification.from_pretrained(tmp_dir, from_tf=True)
        model.save_pretrained(model_dir)
        return model
    except Exception as e:
        st.error(f"Failed to load BERTweet model: {str(e)}")
        st.info("Falling back to Twitter-RoBERTa model.")
        return load_roberta_model()

def load_distilbert_model():
    path = MODELS["DistilBERT-base"]["path"]
    model_dir = os.path.join(MODEL_CACHE_DIR, "distilbert")
    

    if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        try:
            return AutoModelForSequenceClassification.from_pretrained(model_dir)
        except Exception as e:
            st.warning(f"Error loading cached model, downloading fresh: {str(e)}")
    
    os.makedirs(model_dir, exist_ok=True)
    try:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(path, from_tf=True)
        except:
            model = AutoModelForSequenceClassification.from_pretrained(path)
        
        model.save_pretrained(model_dir)
        return model
    except Exception as e:
        st.error(f"Failed to load DistilBERT model: {str(e)}")
        st.info("Falling back to Twitter-RoBERTa model.")
        return load_roberta_model()

def load_roberta_tokenizer():
    path = MODELS["Twitter-RoBERTa"]["path"]
    tokenizer_dir = os.path.join(MODEL_CACHE_DIR, "twitter-roberta-tokenizer")

    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json")):
        try:
            return AutoTokenizer.from_pretrained(tokenizer_dir)
        except Exception as e:
            st.warning(f"Error loading cached tokenizer, downloading fresh: {str(e)}")
    

    os.makedirs(tokenizer_dir, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.save_pretrained(tokenizer_dir)
        return tokenizer
    except Exception as e:
        st.error(f"Failed to load Twitter-RoBERTa tokenizer: {str(e)}")
        st.stop()

def load_bertweet_tokenizer():
    path = MODELS["BERTweet"]["path"]
    tokenizer_dir = os.path.join(MODEL_CACHE_DIR, "bertweet-tokenizer")
    
    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json")):
        try:
            return AutoTokenizer.from_pretrained(tokenizer_dir, normalization=True)
        except Exception as e:
            st.warning(f"Error loading cached tokenizer, downloading fresh: {str(e)}")
    
    os.makedirs(tokenizer_dir, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, normalization=True)
        tokenizer.save_pretrained(tokenizer_dir)
        return tokenizer
    except Exception as e:
        st.error(f"Failed to load BERTweet tokenizer: {str(e)}")
        st.info("Falling back to Twitter-RoBERTa tokenizer.")
        return load_roberta_tokenizer()

def load_distilbert_tokenizer():
    path = MODELS["DistilBERT-base"]["path"]
    tokenizer_dir = os.path.join(MODEL_CACHE_DIR, "distilbert-tokenizer")
    
    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json")):
        try:
            return AutoTokenizer.from_pretrained(tokenizer_dir)
        except Exception as e:
            st.warning(f"Error loading cached tokenizer, downloading fresh: {str(e)}")
    
    os.makedirs(tokenizer_dir, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.save_pretrained(tokenizer_dir)
        return tokenizer
    except Exception as e:
        st.error(f"Failed to load DistilBERT tokenizer: {str(e)}")
        st.info("Falling back to Twitter-RoBERTa tokenizer.")
        return load_roberta_tokenizer()


def load_selected_model_and_tokenizer(model_name):
   
    if (st.session_state.model is not None and 
        st.session_state.tokenizer is not None and 
        st.session_state.model_name == model_name):
        return st.session_state.model, st.session_state.tokenizer
    
    model_config = MODELS[model_name]  

    model_load_func = globals()[model_config["load_func"]]
    tokenizer_load_func = globals()[model_config["tokenizer_func"]]
    

    with st.spinner(f"Loading {model_name}... This might take a while if first time"):
        model = model_load_func()
        tokenizer = tokenizer_load_func()
        
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.model_name = model_name
    
    return model, tokenizer


def get_sentiment(text, model, tokenizer, model_name):
    try:
        if model_name == "BERTweet":
  
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
        with st.spinner("Analyzing text..."):
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1).detach().numpy()[0]
        
 
        labels = {}
        for i, label in model.config.id2label.items():
            label_lower = label.lower()
            if "negative" in label_lower or "neg" in label_lower:
                labels["negative"] = int(i)
            elif "neutral" in label_lower or "neu" in label_lower:
                labels["neutral"] = int(i)
            elif "positive" in label_lower or "pos" in label_lower:
                labels["positive"] = int(i)
        
        if "negative" not in labels:
            labels["negative"] = 0
        if "neutral" not in labels:
            labels["neutral"] = 1
        if "positive" not in labels:
            labels["positive"] = 2
            
        return {
            "negative": probs[labels["negative"]],
            "neutral": probs[labels["neutral"]],
            "positive": probs[labels["positive"]]
        }
        
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return {"negative": 0.33, "neutral": 0.33, "positive": 0.34}

def compare_sentiments_double_bar(male_text, female_text, model, tokenizer, model_name):
    male_sentiment = get_sentiment(male_text, model, tokenizer, model_name)
    female_sentiment = get_sentiment(female_text, model, tokenizer, model_name)

    labels = ['Negative', 'Neutral', 'Positive']
    male_values = [male_sentiment['negative'], male_sentiment['neutral'], male_sentiment['positive']]
    female_values = [female_sentiment['negative'], female_sentiment['neutral'], female_sentiment['positive']]
    
    x = np.arange(len(labels))  
    width = 0.35 

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_male = ax.bar(x - width/2, male_values, width, label='Male', color='blue', alpha=0.7)
    bars_female = ax.bar(x + width/2, female_values, width, label='Female', color='pink', alpha=0.7)

    ax.set_xlabel("Sentiment Categories", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"Sentiment Distribution: {male_text} vs {female_text}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    return fig, male_sentiment, female_sentiment

def create_heatmap(examples, model, tokenizer, model_name):
    all_data = []
    

    for i, (male_text, female_text) in enumerate(examples):
      
        st.text(f"Processing example {i+1}/{len(examples)}: {male_text} vs {female_text}")
        
        male_sentiment = get_sentiment(male_text, model, tokenizer, model_name)
        female_sentiment = get_sentiment(female_text, model, tokenizer, model_name)

        differences = [
            female_sentiment['negative'] - male_sentiment['negative'],
            female_sentiment['neutral'] - male_sentiment['neutral'],
            female_sentiment['positive'] - male_sentiment['positive']
        ]
        all_data.append(differences)

    heatmap_data = np.array(all_data)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=[f"{male.split()[-1]} vs {female.split()[-1]}" 
                             for male, female in examples],
                cbar_kws={'label': 'Sentiment Difference (Female - Male)'})
    
    ax.set_title("Gender Bias Heatmap (Female - Male)\nPositive values indicate higher sentiment for Female text", fontsize=14)
    
    plt.tight_layout()
    
    return fig

def calculate_bias_score(male_text, female_text, model, tokenizer, model_name):
    male_sentiment = get_sentiment(male_text, model, tokenizer, model_name)
    female_sentiment = get_sentiment(female_text, model, tokenizer, model_name)

    neg_bias = female_sentiment['negative'] - male_sentiment['negative']
    pos_bias = male_sentiment['positive'] - female_sentiment['positive']

    return (neg_bias + pos_bias) / 2

def render_single_comparison(model, tokenizer, model_name):
    st.subheader("Single Text Pair Analysis")
    
    input_mode = st.radio(
        "Choose input mode:",
        ["Predefined Examples", "Custom Input"],
        horizontal=True
    )
    
    if input_mode == "Predefined Examples":
        selected_example_index = st.selectbox(
            "Choose predefined example:", 
            options=range(len(DEFAULT_EXAMPLES)),
            format_func=lambda i: f"{DEFAULT_EXAMPLES[i][0]} vs {DEFAULT_EXAMPLES[i][1]}"
        )
        
        default_male, default_female = DEFAULT_EXAMPLES[selected_example_index]
        
        col1, col2 = st.columns(2)
        with col1:
            male_text = st.text_input("Male Reference Text", default_male)
        with col2:
            female_text = st.text_input("Female Reference Text", default_female)
    else:
        col1, col2 = st.columns(2)
        with col1:
            male_text = st.text_area("Enter Male Reference Text", "", height=150)
        with col2:
            female_text = st.text_area("Enter Female Reference Text", "", height=150)

    if st.button("Analyze Gender Bias"):
        if male_text and female_text:
            try:
                with st.spinner("Analyzing..."):
                    fig, male_sentiment, female_sentiment = compare_sentiments_double_bar(
                        male_text, female_text, model, tokenizer, model_name
                    )
                    st.pyplot(fig)

                    bias_score = calculate_bias_score(male_text, female_text, model, tokenizer, model_name)
                    st.metric(
                        label="Bias Score",
                        value=f"{bias_score:.2f}",
                        help="Positive score indicates bias against female text. Negative score indicates bias against male text."
                    )
                    
                    # Add more detailed sentiment breakdown
                    st.subheader("Detailed Sentiment Analysis")
                    sent_df = pd.DataFrame({
                        'Sentiment': ['Negative', 'Neutral', 'Positive'],
                        'Male Text': [male_sentiment['negative'], male_sentiment['neutral'], male_sentiment['positive']],
                        'Female Text': [female_sentiment['negative'], female_sentiment['neutral'], female_sentiment['positive']],
                        'Difference (F-M)': [
                            female_sentiment['negative'] - male_sentiment['negative'],
                            female_sentiment['neutral'] - male_sentiment['neutral'],
                            female_sentiment['positive'] - male_sentiment['positive']
                        ]
                    })
                    
                    numeric_cols = ['Male Text', 'Female Text', 'Difference (F-M)']
                    st.dataframe(sent_df.style.format({col: '{:.4f}' for col in numeric_cols}))
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.code(str(e), language="python")

def render_batch_analysis(model, tokenizer, model_name):
    st.subheader("Batch Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with columns 'male_text' and 'female_text'", 
        type="csv"
    )

    analysis_examples = []  
    
    st.download_button(
        label="Download Example CSV Template",
        data=pd.DataFrame({
            'male_text': ["He is a leader", "The man is ambitious"],
            'female_text': ["She is a leader", "The woman is ambitious"]
        }).to_csv(index=False),
        file_name="example_template.csv",
        mime="text/csv"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'male_text' not in df.columns or 'female_text' not in df.columns:
                st.error("CSV must contain columns named 'male_text' and 'female_text'")
            else:
                uploaded_pairs = list(zip(df['male_text'], df['female_text']))
                analysis_examples.extend(uploaded_pairs)
                st.success(f"Loaded {len(uploaded_pairs)} pairs from CSV.")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

    use_default_examples = st.checkbox("Include default examples", value=True)
    
    if use_default_examples:
        analysis_examples.extend(DEFAULT_EXAMPLES)

    if st.button("Generate Comprehensive Analysis"):
        if not analysis_examples:
            st.warning("No examples to analyze. Please upload a CSV or include default examples.")
            return
            
        try:
            with st.spinner("Processing all examples..."):
           
                bias_scores = []
                all_results = []
                
                for i, (male_text, female_text) in enumerate(analysis_examples):
                    st.text(f"Analyzing pair {i+1}/{len(analysis_examples)}: {male_text} vs {female_text}")
                    bias_score = calculate_bias_score(male_text, female_text, model, tokenizer, model_name)
                    bias_scores.append(bias_score)
                    
                    all_results.append({
                        'Male Text': male_text,
                        'Female Text': female_text,
                        'Bias Score': bias_score
                    })
                        
                st.subheader("Combined Analysis Heatmap")
                heatmap_fig = create_heatmap(analysis_examples, model, tokenizer, model_name)
                st.pyplot(heatmap_fig)
                
                avg_bias = sum(bias_scores) / len(bias_scores)
                
                st.metric(
                    label="Average Bias Score",
                    value=f"{avg_bias:.3f}",
                    help="Average bias across all examples. Positive values indicate bias against female text."
                )
                
                results_df = pd.DataFrame(all_results)
                st.subheader("Individual Pair Results")
                st.dataframe(
                    results_df.style.format({'Bias Score': '{:.3f}'})
                    .background_gradient(subset=['Bias Score'], cmap='coolwarm', vmin=-0.5, vmax=0.5)
                )
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            st.code(str(e), language="python")

def explain_bias_score():
    with st.expander("📊 Understanding the Bias Score"):
        st.markdown("""
        **Bias Score Calculation:**  
        `(Female_Negative - Male_Negative) + (Male_Positive - Female_Positive) / 2`  
        
        - **Positive values**: Bias against female text  
        - **Negative values**: Bias against male text  
        - **Near zero**: Balanced sentiment  
        
        The bias score measures how differently sentiment models evaluate identical statements when only the gender references are changed. This measurement helps identify potential gender biases embedded in NLP models.
        """)

def show_troubleshooting():
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        ### Common Issues:
        
        1. **Model Loading Errors**
           - If a specific model fails to load, the system will automatically fall back to the Twitter-RoBERTa model
           - Check your internet connection if you're downloading models for the first time
        
        2. **Slow Performance**
           - The first analysis may take longer as models are loaded and cached
           - Subsequent analyses should be faster as models are now stored locally
        
        3. **Batch Analysis Taking Too Long**
           - Consider reducing the number of examples if batch analysis is timing out
        """)

def main():
    st.title("Gender Bias Text Analyzer")
    
    st.sidebar.header("Model Settings")
    

    selected_model_name = st.sidebar.selectbox(
        "Choose Analysis Model",
        list(MODELS.keys()),
        index=0
    )
    
    if selected_model_name in ["BERTweet", "DistilBERT-base"]:
        st.sidebar.warning(
            f"{selected_model_name} may have loading issues. "
            "If it fails, the app will automatically fall back to Twitter-RoBERTa."
        )
    
    st.sidebar.markdown(f"**Model Path:** `{MODELS[selected_model_name]['path']}`")
    
    try:

        cache_size = 0
        if os.path.exists(MODEL_CACHE_DIR):
            cache_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                        for dirpath, _, filenames in os.walk(MODEL_CACHE_DIR) 
                        for filename in filenames) / (1024 * 1024)  
        
        st.sidebar.markdown(f"**Model Cache:** `{MODEL_CACHE_DIR}`")
        st.sidebar.markdown(f"**Cache Size:** `{cache_size:.1f} MB`")
    except Exception as e:
        st.sidebar.warning(f"Could not calculate cache size: {e}")
    
    if st.sidebar.button("Clear Model Cache"):
        try:
            import shutil
            shutil.rmtree(MODEL_CACHE_DIR)
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            st.sidebar.success("Cache cleared successfully!")
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.model_name = None
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing cache: {str(e)}")
    
    st.sidebar.markdown("---")
    
    show_troubleshooting()
    
    with st.sidebar.expander("About This App"):
        st.markdown("""
        This application analyzes potential gender bias in language models by comparing sentiment scores 
        between equivalent statements that differ only in gender references.
        
        The models used are trained on social media data and may reflect biases present in those datasets.
        """)
    
    try:
        try:
            model, tokenizer = load_selected_model_and_tokenizer(selected_model_name)
            st.success(f"Successfully loaded {selected_model_name}")
        except Exception as e:
            st.error(f"Failed to load {selected_model_name}: {str(e)}")
            st.info("Falling back to Twitter-RoBERTa model")
            selected_model_name = "Twitter-RoBERTa"
            model, tokenizer = load_selected_model_and_tokenizer(selected_model_name)
        
        explain_bias_score()
        
        tab1, tab2 = st.tabs(["Single Pair Analysis", "Batch Analysis"])
        
        with tab1:
            render_single_comparison(model, tokenizer, selected_model_name)
        
        with tab2:
            render_batch_analysis(model, tokenizer, selected_model_name)
    
    except Exception as e:
        st.error(f"Critical error initializing application: {str(e)}")
        st.code(str(e), language="python")

if __name__ == "__main__":
    main()