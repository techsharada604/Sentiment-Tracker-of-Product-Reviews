import streamlit as st
import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import streamlit as st
import requests
from io import StringIO

# Map categories to Google Drive file IDs
category_links = {
    "Health and Personal Care": "1M4zbdO1NcuzEakPuy75jPFs8sVj4hyxN",
    "Appliances": "16RFWv60D-U0DZQ5Dm3wyCWrLnZhuO0tz",
    "Beauty": "19gOrZRHK2qYpgRxuk8S4Udf2U78cw7g1",
    "Fashion": "1IAp6ueVdIOucyFN3GF7LxYscJnkrk99f"
}

@st.cache_data
def load_data(file_id):
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load data.")
        return pd.DataFrame()

# Sidebar category selector
st.sidebar.title("Product Review Category")
category = st.sidebar.selectbox("Select Category", list(category_links.keys()))

# Load and display dataset
df = load_data(category_links[category])
st.title(f"{category} Reviews")
st.dataframe(df.head())



import altair as alt
import plotly.express as px
st.set_page_config(
    page_title="Product Reviews",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded")
alt.themes.enable("dark")
import re

def clean_title(title):
    # Cut off at first special character (dash, comma, parenthesis, etc.)
    return re.split(r"[-,(]", title)[0].strip()

with st.sidebar:
    st.title("Product List")

    product_review_counts = df['title_y'].value_counts().sort_values(ascending=False)
    product_list = product_review_counts.index.tolist()
    cleaned_titles = [clean_title(title) for title in product_list]
    title_map = dict(zip(cleaned_titles, product_list))
    selected_cleaned_title = st.selectbox('Select a product', cleaned_titles, index=len(cleaned_titles)-1)
    selected_product = title_map[selected_cleaned_title]
    df_selected_product = df[df.title_y == selected_product]


st.subheader(selected_product)
st.subheader("Product Overview")
first_row = df_selected_product.iloc[0]

product_description = first_row.get('description', 'Not Available')

with st.expander("Features", expanded=False):
    import ast
    product_features = first_row.get('features', 'Not Available')
    try:
        features_list = ast.literal_eval(product_features)
    except Exception:
        features_list = [product_features]  

    st.markdown("### Features")
    for feature in features_list:
        st.markdown(f"- {feature}")



with st.expander("Description", expanded=False):
    import ast
    raw_description = first_row.get('description', [])
    if isinstance(raw_description, str):
        try:
            raw_description = ast.literal_eval(raw_description)
        except:
            raw_description = [raw_description]

    seen = set()
    cleaned_lines = []
    for line in raw_description:
        line = str(line).strip().strip('.').replace('�', "'")
        if line and line.lower() not in seen:
            seen.add(line.lower())
            cleaned_lines.append(line)
    st.markdown("**Product Description**")
    for line in cleaned_lines:
        if line.lower().startswith("nsf certified") or line.endswith(":"):
            st.markdown(f"**{line}**")
        else:
            st.markdown(f"- {line}")



verified_counts = df_selected_product['verified_purchase'].value_counts().rename({True: 'Verified', False: 'Unverified'})
col1, col2, col3 = st.columns([1, 2,2])
fig = px.pie(
    names=verified_counts.index,
    values=verified_counts.values,
    title=f"Verified vs Unverified Purchases for {selected_product}",
    hole=0.3
)
with col1:
    st.subheader("Verified Purchase Breakdown")
    st.plotly_chart(fig, use_container_width=False)
with col2:
    st.subheader("BERT Sentiment Over Time")
    df_selected_product['timestamp'] = pd.to_datetime(df_selected_product['timestamp'], errors='coerce')
    df_selected_product = df_selected_product.dropna(subset=['timestamp'])
    sentiment_trend = (
        df_selected_product
        .set_index('timestamp')
        .groupby([pd.Grouper(freq='M'), 'bert_sentiment1'])['text']
        .count()
        .unstack()
        .fillna(0)
        .reset_index()
    )
    sentiment_trend_long = sentiment_trend.melt(id_vars='timestamp', var_name='Sentiment', value_name='Review Count')
    fig = px.line(
        sentiment_trend_long,
        x='timestamp',
        y='Review Count',
        color='Sentiment',
        title=f"Sentiment Trend Over Time for Product {selected_product}",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("📊 Rating Distribution")

    # Fix: get rating distribution for selected product
    rating_counts = (
        df[df['title_y'] == selected_product]['rating']
        .value_counts()
        .sort_index()
    )

    # Convert to DataFrame with proper column names
    rating_df = pd.DataFrame({
        'Rating': rating_counts.index.astype(str),
        'Count': rating_counts.values
    })

    # Plot with Plotly
    fig_rating = px.bar(
        rating_df,
        x='Rating',
        y='Count',
        text='Count',
        color='Rating',
        title=f"Rating Distribution for {selected_product}"
    )

    fig_rating.update_traces(textposition='outside')
    fig_rating.update_layout(xaxis_title="Rating", yaxis_title="Number of Reviews")

    st.plotly_chart(fig_rating, use_container_width=True)


import ast

df['aspect_sentiment1'] = df['aspect_sentiment1'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

st.set_page_config(page_title="Aspect Sentiment Viewer", layout="wide")
st.title("User reviews")

df1 = df[df['title_y'] == selected_product]

all_aspects = sorted({aspect for item in df1['aspect_sentiment1'].dropna() for aspect in item.keys()})

st.sidebar.header("Filters")

selected_aspects = st.sidebar.multiselect("Filter by Aspects", all_aspects, default=all_aspects)

available_ratings = sorted(df1['bert_sentiment1'].unique())
selected_ratings = st.sidebar.multiselect("Filter by Ratings", available_ratings, default=available_ratings)

selected_users = st.sidebar.multiselect("Filter by Users", df1['user_id'].unique())

filtered_df = df1[
    df1['bert_sentiment1'].isin(selected_ratings) &
    df1['aspect_sentiment1'].apply(lambda d: any(a in selected_aspects for a in d.keys()) if isinstance(d, dict) else False)
]

if selected_users:
    filtered_df = filtered_df[filtered_df['user_id'].isin(selected_users)]
filtered_df = filtered_df.sort_values(by='helpful_vote', ascending=False).reset_index(drop=True)

st.write(f"### Showing {len(filtered_df)} filtered reviews")

comments_per_page = 5
total_reviews = len(filtered_df)
total_pages = (total_reviews - 1) // comments_per_page + 1

if "page_number" not in st.session_state:
    st.session_state.page_number = 0

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ Previous") and st.session_state.page_number > 0:
        st.session_state.page_number -= 1
with col3:
    if st.button("Next ➡️") and st.session_state.page_number < total_pages - 1:
        st.session_state.page_number += 1
with col2:
    st.markdown(f"**Page {st.session_state.page_number + 1} of {total_pages}**")

start_idx = st.session_state.page_number * comments_per_page
end_idx = start_idx + comments_per_page
paged_df = filtered_df.iloc[start_idx:end_idx]

for i, row in paged_df.iterrows():
    with st.expander(f"Review by {row['user_id']} (Rating: {row['rating']}) — {row['helpful_vote']}"):
        st.markdown(f"**Full Review:** {row['text']}")

        aspects = row.get('aspect_sentiment1', {})
        if isinstance(aspects, dict):
            for aspect, details in aspects.items():
                if aspect in selected_aspects:
                    sentiment_icon = details.get('sentiment', '⚪')
                    confidence = details.get('confidence', 0.0)
                    evidence_sentences = details.get('evidence', [])

                    st.markdown(f"**{aspect.title()}**: {sentiment_icon} (Confidence: {confidence:.2f})")
                    for sentence in evidence_sentences:
                        st.markdown(f"- _{sentence}_")
