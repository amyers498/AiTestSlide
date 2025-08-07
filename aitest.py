import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ast import literal_eval
import torch
import altair as alt

model = SentenceTransformer('./minilm_model')
df = pd.read_csv("slidesocial_dummy_users.csv")
df["Traits"] = df["Traits"].apply(lambda x: [t.strip() for t in x.split(",")])
df["Pref_Gender"] = df["Pref_Gender"].apply(lambda x: [g.strip() for g in x.split(",")])

@st.cache_data(show_spinner=False)
def generate_embeddings():
    texts = df["Activity"] + "; " + df["Traits"].apply(lambda t: ", ".join(t))
    emb = model.encode(texts)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return emb / norms

embeddings = generate_embeddings()

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
        if len(selected_traits) < 4:
            st.error("Please select at least 4 traits.")
        else:
            user_text = activity + "; " + ", ".join(selected_traits)
            user_vec = model.encode(user_text)
            user_norm = np.linalg.norm(user_vec)
            if user_norm != 0:
                user_vec /= user_norm

            candidates = df[
                df["Gender"].isin(pref_gender) &
                (df["Age"] >= pref_age[0]) & (df["Age"] <= pref_age[1]) &
                (df["Location"] == location)
            ]

            if len(candidates) == 0:
                st.warning("No matches found with your filters. Try broadening your preferences.")
            else:
                candidate_indices = candidates.index.values
                candidate_emb = embeddings[candidate_indices]
                scores = np.dot(user_vec, candidate_emb.T)
                candidates = candidates.copy()
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
        location_groups = df.groupby('Location')
        total_users = len(df)
        progress_bar = st.progress(0)
        processed = 0

        for loc, group in location_groups:
            group_indices = group.index.values
            group_emb = embeddings[group_indices]

            for i, row in group.iterrows():
                local_i = np.where(group_indices == i)[0][0]
                user_vec = group_emb[local_i]

                mask = (
                    group['Gender'].isin(row['Pref_Gender']) &
                    (group['Age'] >= row['Pref_Age_Min']) &
                    (group['Age'] <= row['Pref_Age_Max']) &
                    (group.index != i)
                )

                filtered_indices = group_indices[mask]
                if len(filtered_indices) == 0:
                    continue

                filtered_local_indices = np.where(mask)[0]
                filtered_emb = group_emb[filtered_local_indices]
                scores = np.dot(user_vec, filtered_emb.T)

                top3_idx = np.argsort(scores)[-3:][::-1]
                for idx in top3_idx:
                    match_idx = filtered_indices[idx]
                    match_row = df.loc[match_idx]
                    results.append({
                        "User": row["Name"],
                        "Matched With": match_row["Name"],
                        "Match Score": float(scores[idx]),
                        "Location": loc,
                        "Traits": ", ".join(match_row["Traits"]),
                        "Activity": match_row["Activity"]
                    })

                processed += 1
                progress_bar.progress(processed / total_users)

        return pd.DataFrame(results)

    result_df = compute_top_3_all()

    st.dataframe(result_df)
    selected_user = st.selectbox("Select a user to inspect matches:", sorted(result_df["User"].unique()))
    user_matches = result_df[result_df["User"] == selected_user].sort_values("Match Score", ascending=False)

    st.subheader(f"🔍 Detailed matches for {selected_user}")
    for _, row in user_matches.iterrows():
        st.markdown(f"**Matched With:** {row['Matched With']}  ")
        st.markdown(f"**Location:** {row['Location']}  ")
        st.markdown(f"**Activity:** {row['Activity']}  ")
        st.markdown(f"**Traits:** {row['Traits']}")
        st.markdown(f"**Match Score:** {row['Match Score']*100:.2f}%")
        st.markdown("---")

    chart = alt.Chart(user_matches).mark_bar().encode(
        x=alt.X('Match Score:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Matched With:N', sort='-x'),
        tooltip=['Matched With', 'Match Score']
    ).properties(height=300, title=f"Top Matches for {selected_user}")
    st.altair_chart(chart, use_container_width=True)

    st.download_button("Download Match Results CSV", result_df.to_csv(index=False), "top_3_matches.csv")
