"""Run with: python -m streamlit run app.py"""
from html import escape

import pandas as pd
import streamlit as st

from dashboard.charts import coverage_chart, sentiment_donut, stacked_chart, topic_heatmap, terms_chart
from dashboard.data import ROOT, SENTIMENTS, TOPICS, export_csv, filter_corpus, fit_language, load_corpus, top_terms

st.set_page_config(page_title="Media Lens · Mamdani Coverage", page_icon="◉", layout="wide")
st.markdown("""<style>
  .block-container {max-width: 1280px; padding-top: 2.3rem; padding-bottom: 2rem;}
  [data-testid="stSidebar"] {border-right: 1px solid #E4E0D5;}
  h1 {font-family: Georgia, serif !important; font-weight: 500 !important;
      font-size: clamp(2.4rem, 4vw, 3.7rem) !important; letter-spacing: -.045em; line-height: 1.08 !important;}
  h2, h3 {letter-spacing: -.025em;}
  [data-testid="stMetric"] {background: #FFFFFF; border-color: #E4E0D5; padding: 18px;}
  [data-testid="stMetricValue"] {font-family: Georgia, serif; font-size: 2.1rem;}
  .eyebrow {font-size: .71rem; text-transform: uppercase; letter-spacing: .18em; color: #6C5BA7; font-weight: 700;}
  .intro {font-size: 1.03rem; color: #656B62; max-width: 730px; line-height: 1.65; margin-bottom: 1.6rem;}
  .finding {border-top: 2px solid #6C5BA7; padding-top: 13px; margin: 5px 0 24px;}
  .finding h4 {font-size: .93rem; margin: 0 0 7px; font-weight: 600;}
  .finding p {color: #656B62; font-size: .88rem; line-height: 1.65; margin: 0;}
  .article-title {font: 1.45rem/1.4 Georgia, serif; margin: 10px 0 12px;}
  .article-snippet {color: #656B62; line-height: 1.7;}
  .small-label {font-size: .76rem; color: #6C5BA7; letter-spacing: .03em;}
  [data-baseweb="tab-list"] {gap: 1.4rem; border-bottom: 1px solid #E4E0D5;}
  [data-baseweb="tab"] {height: 3.2rem; font-size: .88rem;}
  @media (max-width: 700px) {
    .block-container {padding-top: 1.2rem;}
    [data-baseweb="tab-list"] {gap: .4rem;}
    [data-baseweb="tab"] {font-size: .75rem; padding: 0 .25rem;}
  }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def cached_corpus():
    return load_corpus()


@st.cache_resource
def cached_language():
    return fit_language(cached_corpus())


def plot(fig, key):
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "scrollZoom": False,
                            "toImageButtonOptions": {"format": "png", "scale": 2}})


def finding(title, body):
    st.markdown(f'<div class="finding"><h4>{escape(title)}</h4><p>{escape(body)}</p></div>',
                unsafe_allow_html=True)


def reset_filters():
    for key in ["outlet_filter", "topic_filter", "sentiment_filter", "search_query"]:
        st.session_state.pop(key, None)


try:
    corpus = cached_corpus()
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Could not load the project data: {error}")
    st.stop()

outlets = sorted(corpus.outlet.unique())
with st.sidebar:
    st.markdown('<p class="eyebrow">COMP 370 · Research explorer</p>', unsafe_allow_html=True)
    st.markdown("## Media Lens")
    st.caption("How did news outlets cover Zohran Mamdani?")
    st.divider()
    st.markdown("**Explore the sample**")
    query = st.text_input("Search headlines & descriptions", placeholder="Try housing, Trump, election…", key="search_query")
    with st.expander("Outlets", expanded=False):
        selected_outlets = st.multiselect("Include outlets", outlets, default=outlets, key="outlet_filter")
    with st.expander("Topics", expanded=False):
        selected_topics = st.multiselect("Include topics", TOPICS, default=TOPICS, key="topic_filter")
    with st.expander("Sentiment", expanded=False):
        selected_sentiments = st.multiselect("Include labels", SENTIMENTS, default=SENTIMENTS, key="sentiment_filter")
    st.button("Reset filters", on_click=reset_filters, width="stretch")
    st.divider()
    st.caption(f"{len(corpus):,} annotated articles · {corpus.outlet.nunique()} outlets\n\n"
               "Archived project data. All charts describe this collected sample.")
    st.link_button("View project on GitHub ↗", "https://github.com/Abudyy/COMP_370_Final_Project", width="stretch")

df = filter_corpus(corpus, selected_outlets, selected_topics, selected_sentiments, query)
st.markdown('<p class="eyebrow">News coverage · Human annotation · Text analysis</p>', unsafe_allow_html=True)
st.title("One candidate.\nMany media narratives.")
st.markdown('<p class="intro">Explore how 15 news outlets covered Zohran Mamdani through '
            'topics, annotated sentiment, and the language of their headlines and descriptions.</p>', unsafe_allow_html=True)

metrics = st.columns(4)
metrics[0].metric("Articles in selection", f"{len(df):,}", border=True)
metrics[1].metric("Outlets represented", str(df.outlet.nunique()), border=True)
metrics[2].metric("Topic categories", str(df.topic.nunique()), border=True)
neg = (df.sentiment == "Negative").mean() * 100 if len(df) else 0
metrics[3].metric("Negative labels", f"{neg:.1f}%" if len(df) else "—", border=True,
                  help="Share of manually assigned Negative labels in the current selection.")
st.caption(f"Showing {len(df):,} of {len(corpus):,} annotated articles. Sidebar filters apply across every view.")

overview, comparisons, language, articles, method = st.tabs(
    ["Overview", "Outlet comparison", "Language", "Article explorer", "Methodology"])

if df.empty:
    with overview:
        st.info("No articles match these filters. Broaden the selection or use Reset filters in the sidebar.")
    for tab in [comparisons, language, articles]:
        with tab:
            st.info("No articles in the current selection.")
else:
    with overview:
        st.markdown("### What stands out")
        col1, col2, col3 = st.columns(3)
        leading_outlet = df.outlet.value_counts().index[0]
        leading_count = int((df.outlet == leading_outlet).sum())
        leading_topic = df.topic.value_counts().index[0]
        leading_topic_count = int((df.topic == leading_topic).sum())
        leading_label = df.sentiment.value_counts().index[0]
        leading_label_count = int((df.sentiment == leading_label).sum())
        with col1:
            finding(f"{leading_outlet} leads the selection",
                    f"{leading_count:,} articles ({leading_count / len(df):.1%}) come from this outlet. "
                    "Outlets are represented unevenly in the collected sample.")
        with col2:
            finding(f"{leading_topic} is the most common topic",
                    f"{leading_topic_count:,} articles ({leading_topic_count / len(df):.1%}) carry this label. "
                    "Each article has one recorded topic category.")
        with col3:
            finding(f"{leading_label} is the most common sentiment",
                    f"{leading_label_count:,} articles ({leading_label_count / len(df):.1%}) carry this annotation. "
                    "These labels come from the project's human coding.")
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown("### Where the coverage comes from")
            st.caption("Article counts by outlet in the current selection.")
            plot(coverage_chart(df), "overview_coverage")
        with right:
            st.markdown("### The sentiment mix")
            st.caption("Manually assigned labels, with counts on hover.")
            plot(sentiment_donut(df), "overview_sentiment")
        st.markdown("### Sentiment changes with the subject")
        st.caption("Percentages are calculated within each topic; hover to see its sample size.")
        plot(stacked_chart(df, "topic"), "overview_topic")
        st.caption("Descriptive results for this sample. A sentiment label alone does not establish political bias or causal effects.")

    with comparisons:
        st.markdown("### Compare outlets on equal terms")
        st.caption("Use shares to compare composition and counts to inspect how much evidence each outlet contributes.")
        control1, control2 = st.columns([1, 1])
        mode = control1.radio("Chart measure", ["Share (%)", "Article count"], horizontal=True)
        minimum = control2.number_input("Minimum articles per outlet", min_value=1, max_value=len(corpus), value=10,
                                       help="Applied after the sidebar filters. Small samples can produce unstable percentages.")
        eligible = df.outlet.value_counts().loc[lambda s: s >= minimum].index
        compared = df[df.outlet.isin(eligible)]
        if compared.empty:
            st.info("No outlet meets this minimum. Lower the threshold or broaden your filters.")
        else:
            st.caption(f"{len(eligible)} outlets meet the threshold · {len(compared):,} articles. "
                       f"{df.outlet.nunique() - len(eligible)} outlets are hidden from these comparison charts.")
            plot(stacked_chart(compared, "outlet", mode), "comparison_sentiment")
            st.markdown("### Which topics receive attention?")
            st.caption("Every row sums to 100%. Hover a cell for its count.")
            plot(topic_heatmap(compared), "comparison_heatmap")
            if len(eligible) >= 2:
                st.markdown("### A closer look at two outlets")
                sorted_eligible = sorted(eligible)
                first, second = st.columns(2)
                outlet_a = first.selectbox("First outlet", sorted_eligible, index=sorted_eligible.index("New York Post")
                                           if "New York Post" in sorted_eligible else 0)
                alternatives = [o for o in sorted_eligible if o != outlet_a]
                outlet_b = second.selectbox("Second outlet", alternatives,
                                            index=alternatives.index("NY Daily News") if "NY Daily News" in alternatives else 0)
                for column, outlet in [(first, outlet_a), (second, outlet_b)]:
                    with column:
                        sample = compared[compared.outlet == outlet]
                        st.markdown(f"**{outlet} · n = {len(sample)}**")
                        plot(sentiment_donut(sample), f"pair_{outlet}")
            st.caption("Differences reflect this collection and its labels. Outlet sample sizes and topic mixes differ; "
                       "no statistical significance or population-wide conclusion is implied.")

    with language:
        st.markdown("### The words behind the coverage")
        st.caption("TF-IDF highlights words and two-word phrases weighted by how distinctive they are across documents.")
        c1, c2, c3 = st.columns([1, 1.5, 1])
        grouping = c1.selectbox("Explore language by", ["Topic", "Outlet", "Entire selection"])
        group_column = "topic" if grouping == "Topic" else "outlet"
        if grouping == "Entire selection":
            selected_language = df
            c2.caption("Using all articles in the current selection.")
        else:
            counts = df[group_column].value_counts()
            chosen = c2.selectbox("Select a group", sorted(counts.index),
                                   format_func=lambda x: f"{x} · n = {counts[x]}")
            selected_language = df[df[group_column] == chosen]
        top_n = c3.slider("Terms to display", min_value=5, max_value=25, value=15)
        a, b = st.columns(2)
        hide_entities = a.toggle("Hide candidate name terms", value=True)
        hide_boilerplate = b.toggle("Hide snippet placeholder terms", value=True)
        matrix, vocabulary = cached_language()
        terms = top_terms(matrix, vocabulary, selected_language.record_id.to_numpy(), top_n,
                          hide_entities=hide_entities, hide_boilerplate=hide_boilerplate)
        st.caption(f"Mean weights for {len(selected_language):,} articles. Vocabulary and IDF are fitted once on the full "
                   f"{len(corpus):,}-article corpus so filtered selections share the same weighting.")
        if terms.empty:
            st.info("No terms remain with these display settings.")
        else:
            chart_col, table_col = st.columns([2, 1], gap="large")
            with chart_col:
                plot(terms_chart(terms), "language_terms")
            with table_col:
                st.dataframe(terms.rename(columns={"term": "Term", "score": "Mean TF-IDF"}), hide_index=True,
                             width="stretch", column_config={"Mean TF-IDF": st.column_config.NumberColumn(format="%.5f")})
                st.download_button("Download displayed terms", terms.to_csv(index=False).encode("utf-8"),
                                   file_name="media_lens_tfidf.csv", mime="text/csv", width="stretch")
        st.info("Some descriptions contain placeholder text such as 'No summary available in snippet'. "
                "The toggles hide selected terms from the chart; they do not modify the original corpus or recalculate its TF-IDF weights.")
        with st.expander("See the original saved TF-IDF results"):
            saved = (ROOT / "data/annotation/tfidf_top_words.txt").read_text(encoding="utf-8")
            st.code(saved, language="text")

    with articles:
        st.markdown("### Follow the findings back to the articles")
        st.caption("Search and filters in the sidebar update this table. Select an article below to inspect its labels and source.")
        st.dataframe(df[["outlet", "title", "topic", "sentiment", "url"]], hide_index=True, width="stretch", height=330,
                     column_config={"outlet": "Outlet", "title": st.column_config.TextColumn("Headline", width="large"),
                                    "topic": "Topic", "sentiment": "Sentiment",
                                    "url": st.column_config.LinkColumn("Original source", display_text="Open ↗")})
        st.download_button(f"Download {len(df):,} selected articles", export_csv(df),
                           file_name="media_lens_articles.csv", mime="text/csv")
        selected_id = st.selectbox("Inspect an article", df.record_id.tolist(),
                                  format_func=lambda i: f"{corpus.loc[i, 'outlet']} · {corpus.loc[i, 'title']}")
        row = df[df.record_id == selected_id].iloc[0]
        with st.container(border=True):
            st.markdown(f'<p class="small-label">{escape(row.outlet)} · {escape(row.topic)} · {escape(row.sentiment)}</p>'
                        f'<p class="article-title">{escape(row.title)}</p>'
                        f'<p class="article-snippet">{escape(row.description)}</p>', unsafe_allow_html=True)
            if row.url:
                st.link_button("Read the original source ↗", row.url)
            else:
                st.caption("No unambiguous outlet-and-title URL match was found in the stored article collection.")

with method:
    st.markdown("### From collected articles to an explorable dataset")
    st.markdown("The dashboard reads the project's committed annotations directly. It preserves the original topic and polarity "
                "labels, rebuilds the original TF-IDF calculation, and connects annotations to stored source links.")
    st.markdown("#### What the data contains")
    st.dataframe(pd.DataFrame([
        {"Component": "Annotated sample", "Details": f"{len(corpus)} rows, {corpus.outlet.nunique()} outlets, {corpus.topic.nunique()} topics, three polarity labels."},
        {"Component": "Annotation source", "Details": "data/annotation/final_full_annotated.csv; three annotator names appear in the file."},
        {"Component": "Article collection", "Details": "data/reformated_articles_final.json; 491 stored article records."},
        {"Component": "Source-link matching", "Details": f"{corpus.url.ne('').sum()} annotations have an unambiguous normalized outlet-and-title match."},
        {"Component": "TF-IDF", "Details": "Title + description; English stop words; unigrams and bigrams; default scikit-learn weighting; New_York substitution."},
        {"Component": "Time", "Details": "The annotation file has no publication-date field. This dashboard does not infer a timeline from URLs."},
    ]), hide_index=True, width="stretch")
    st.markdown("#### Reading the results carefully")
    st.markdown("""
    - **Collected sample:** these counts describe the stored articles, not all North American reporting or public opinion.
    - **Human labels:** sentiment and topics are recorded annotations. The repository does not include annotation guidelines, overlapping ratings, or an inter-rater agreement statistic.
    - **Uneven coverage:** outlets contribute very different numbers of articles. Comparison charts show counts, within-outlet shares, and a sample-size threshold.
    - **Short text:** the language analysis uses headlines and descriptions, not full article bodies. Placeholder descriptions can affect TF-IDF rankings.
    - **Descriptive analysis:** charts do not establish causation, statistical significance, or an outlet's political bias.
    """)
    st.markdown("#### The six recorded topic labels")
    st.write(", ".join(TOPICS) + ". These names are preserved from the annotation file; no original coding rubric is supplied.")
    with st.expander("Optional: inspect the project's simulated outlet scores"):
        st.warning("These are simulated scores supplied in data/media_bias.json. They are not measured article sentiment, "
                   "verified third-party ratings, or validated estimates of outlet bias.")
        outlet_meta = corpus[["outlet", "simulated_bias_score", "simulated_group"]].drop_duplicates().sort_values("outlet")
        st.dataframe(outlet_meta, hide_index=True, width="stretch")
        st.caption("The original distribution script uses Left ≤ −3, Right ≥ +3, and Center between those values. "
                   "Unmapped outlets are retained explicitly here rather than silently omitted.")
    st.markdown("#### Project credits")
    st.write("Original COMP 370 project: Abdullah, Antonin, and Kihyeok, as recorded in the annotation file. "
             "The dashboard adds interactive exploration of their collected and annotated data.")

st.divider()
st.caption("MEDIA LENS · A COMP 370 data science project · Python / pandas / scikit-learn / Streamlit / Plotly")
