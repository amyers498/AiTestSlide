# slidesocial_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load user data (make sure the CSV is in the same directory)
df = pd.read_csv("slidesocial_dummy_users.csv")

# Process traits into lists
df["Traits"] = df["Traits"].apply(lambda x: [t.strip() for t in x.split(",")])
df["Pref_Gender"] = df["Pref_Gender"].apply(lambda x: [g.strip() for g in x.split(",")])

# Embed activity + traits
@st.cache_data(show_spinner=False)
def generate_embeddings():
    return model.encode(df["Activity"] + "; " + df["Traits"].apply(lambda t: ", ".join(t)))

embeddings = generate_embeddings()

# UI
st.set_page_config(page_title="SlideSocial Matcher", layout="wide")
st.title("SlideSocial Matcher 🔮")

page = st.sidebar.selectbox("Choose a page:", ["Match Me (Page 1)", "Batch Match Test (Page 2)"])

if page == "Match Me (Page 1)":
    with st.form("user_form"):
        name = st.text_input("Your Name")
        age = st.slider("Your Age", 18, 30, 22)
        gender = st.selectbox("Your Gender", ["Male", "Female", "Nonbinary"])
        location = st.selectbox("Your Location", sorted(df["Location"].unique()))
        activity = st.text_input("What do you want to do tonight?", placeholder="Type freely...")

        st.markdown("**Pick at least 4 personality traits:**")
        all_traits = sorted(set([trait for traits in df["Traits"] for trait in traits]))
        selected_traits = st.multiselect("Your Traits", all_traits, default=all_traits[:4])

        pref_gender = st.multiselect("Preferred Genders", ["Male", "Female", "Nonbinary"], default=["Male", "Female"])
        pref_age = st.slider("Preferred Age Range", 18, 30, (20, 26))

        submitted = st.form_submit_button("Find My Top 3 Matches")

    if submitted:
        user_text = activity + "; " + ", ".join(selected_traits)
        user_vec = model.encode(user_text, convert_to_tensor=True)

        # Filter candidates
        candidates = df[
            (df["Gender"].isin(pref_gender)) &
            (df["Age"] >= pref_age[0]) & (df["Age"] <= pref_age[1]) &
            (df["Location"] == location)
        ].copy()

        if len(candidates) == 0:
            st.warning("No matches found with your filters. Try broadening your preferences.")
        else:
            candidate_vecs = model.encode(
                (candidates["Activity"] + "; " + candidates["Traits"].apply(lambda t: ", ".join(t))).tolist(),
                convert_to_tensor=True
            )
            scores = util.cos_sim(user_vec, candidate_vecs)[0].cpu().numpy()
            candidates["Score"] = scores
            top_matches = candidates.sort_values("Score", ascending=False).head(3)

            st.subheader("🔗 Your Top 3 Matches")
            for idx, row in top_matches.iterrows():
                st.markdown(f"**{row['Name']}**, {row['Age']} ({row['Gender']}) in *{row['Location']}*  ")
                st.markdown(f"**Traits:** {', '.join(row['Traits'])}")
                st.markdown(f"**Activity:** {row['Activity']}")
                st.markdown(f"**Match Score:** {row['Score']*100:.2f}%")
                st.markdown("---")

elif page == "Batch Match Test (Page 2)":
    st.subheader("🔁 Running batch match test on all users...")

    @st.cache_data(show_spinner=True)
    def compute_top_3_all():
        results = []
        for i, row in df.iterrows():
            user_vec = model.encode(row['Activity'] + "; " + ", ".join(row['Traits']), convert_to_tensor=True)

            prefs = (
                (df["Gender"].apply(lambda g: g in row["Pref_Gender"])) &
                (df["Age"] >= row["Pref_Age_Min"]) &
                (df["Age"] <= row["Pref_Age_Max"]) &
                (df["Location"] == row["Location"]) &
                (df.index != i)  # exclude self
            )
            filtered = df[prefs].copy()

            if len(filtered) == 0:
                continue

            texts = (filtered["Activity"] + "; " + filtered["Traits"].apply(lambda t: ", ".join(t))).tolist()
            filtered_vecs = model.encode(texts, convert_to_tensor=True)

            scores = util.cos_sim(user_vec, filtered_vecs)[0].cpu().numpy()
            filtered["Score"] = scores
            top_3 = filtered.sort_values("Score", ascending=False).head(3)

            for _, match in top_3.iterrows():
                results.append({
                    "User": row["Name"],
                    "Matched With": match["Name"],
                    "Match Score": float(match["Score"]),
                    "Location": row["Location"]
                })

        return pd.DataFrame(results)

    result_df = compute_top_3_all()
    st.dataframe(result_df)
    st.download_button("Download Match Results CSV", result_df.to_csv(index=False), "top_3_matches.csv")
