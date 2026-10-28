
import streamlit as st
import pandas as pd
import plotly.express as px

from sentiment import load_classifier, analyze_headlines, summarize_sentiment

st.set_page_config(
    page_title="Financial News Sentiment & Risk Tagger",
    page_icon="📈",
    layout="wide"
)


@st.cache_resource
def get_model():
    return load_classifier()


st.title("Financial News Sentiment & Market Risk Tagger")

st.write(
    "Analyze financial headlines using FinBERT, inspect sentiment "
    "scores, and identify preliminary risk-related keywords."
)

st.warning(
    "Educational prototype only. Sentiment and keyword flags do not "
    "predict stock returns, actual financial losses, or investment risk."
)

with st.spinner("Loading financial NLP model..."):
    try:
        classifier = get_model()
    except Exception as error:
        st.error(f"Could not load FinBERT: {error}")
        st.stop()

manual_tab, upload_tab = st.tabs(
    ["Enter headlines", "Upload CSV"]
)

if "results" not in st.session_state:
    st.session_state["results"] = None

with manual_tab:
    text_input = st.text_area(
        "Enter one financial headline per line",
        height=180,
        placeholder=(
            "Company reports record profits\n"
            "Bank warns of rising loan defaults\n"
            "Company schedules its annual meeting"
        )
    )

    if st.button("Analyze headlines", key="manual"):
        headlines = [
            line.strip()
            for line in text_input.splitlines()
            if line.strip()
        ]

        if not headlines:
            st.warning("Enter at least one headline.")
        else:
            with st.spinner("Analyzing headlines..."):
                st.session_state["results"] = analyze_headlines(
                    headlines, classifier
                )

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Include a column named headline, text, or news."
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            lookup = {
                str(col).strip().lower(): col
                for col in input_df.columns
            }

            text_column = next(
                (
                    lookup[name]
                    for name in ["headline", "text", "news"]
                    if name in lookup
                ),
                None
            )

            if text_column is None:
                st.error(
                    "CSV needs a headline, text, or news column."
                )
            else:
                st.write("Input preview")
                st.dataframe(input_df.head(), use_container_width=True)

                if st.button("Analyze CSV", key="csv"):
                    with st.spinner("Analyzing uploaded headlines..."):
                        st.session_state["results"] = analyze_headlines(
                            input_df[text_column].tolist(),
                            classifier
                        )

        except Exception as error:
            st.error(f"Could not process CSV: {error}")


results = st.session_state["results"]

if results is not None:
    if results.empty:
        st.warning("No valid headlines were found.")
        st.stop()

    summary = summarize_sentiment(results)

    st.divider()
    st.header("Analysis Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Headlines", summary["total"])
    col2.metric("Bullish", summary["bullish"])
    col3.metric("Bearish", summary["bearish"])
    col4.metric("Neutral", summary["neutral"])

    st.subheader("Aggregate directional sentiment")

    score = summary["net_sentiment"]

    st.metric("Average sentiment score", f"{score:+.3f}")

    st.caption(
        "Mean positive probability minus negative probability. "
        "A directional sentiment indicator, not a market forecast."
    )

    chart_col, risk_col = st.columns(2)

    with chart_col:
        sentiment_counts = (
            results["sentiment"]
            .value_counts()
            .reindex(
                ["Bullish", "Neutral", "Bearish"],
                fill_value=0
            )
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            title="Sentiment distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with risk_col:
        risk_counts = (
            results["risk_level"]
            .value_counts()
            .rename_axis("Risk flag category")
            .reset_index(name="Count")
        )

        fig = px.bar(
            risk_counts,
            x="Risk flag category",
            y="Count",
            title="Rule-based risk flag summary"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Headline-level results")

    display_df = results.copy()

    for col in [
        "confidence", "positive", "negative",
        "neutral", "sentiment_score"
    ]:
        display_df[col] = display_df[col].round(4)

    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        "Download analysis as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="financial_sentiment_results.csv",
        mime="text/csv"
    )

    with st.expander("How the risk flags work"):
        st.write(
            "The current risk layer searches headlines for predefined "
            "terms relating to earnings deterioration, debt or liquidity, "
            "legal concerns, and operational uncertainty. One matching "
            "category produces a Review flag; two or more produce a High "
            "flag. These rules are a simple prototype, not a validated "
            "financial risk model."
        )
