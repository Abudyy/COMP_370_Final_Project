"""Shared, accessible Plotly charts with counts and explicit denominators."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dashboard.data import distribution, SENTIMENTS, TOPICS

COLORS = {"Negative": "#C76B52", "Neutral": "#A6ADA7", "Positive": "#426E62"}
TOPIC_COLORS = dict(zip(TOPICS, ["#6C5BA7", "#426E62", "#C28A4D", "#667E9C", "#B4677C", "#8A8676"]))


def style(fig, height=360):
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color="#282C28", size=12),
        margin=dict(l=0, r=10, t=12, b=12),
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#282C28"),
    )
    fig.update_xaxes(gridcolor="#E9E6DE", zeroline=False, title_font_size=12)
    fig.update_yaxes(gridcolor="#E9E6DE", zeroline=False, title_font_size=12)
    return fig


def coverage_chart(df):
    counts = df.outlet.value_counts().rename_axis("outlet").reset_index(name="count")
    counts["share"] = counts["count"] / len(df) * 100
    fig = px.bar(counts, y="outlet", x="count", orientation="h", text="count",
                 custom_data=["share"], color_discrete_sequence=["#6C5BA7"])
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}<br>%{x} articles<br>%{customdata[0]:.1f}% of selection<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=counts.outlet.tolist()[::-1], title=None)
    fig.update_xaxes(title="Annotated articles", range=[0, counts['count'].max() * 1.16])
    return style(fig, max(330, len(counts) * 28))


def sentiment_donut(df):
    counts = df.sentiment.value_counts().reindex(SENTIMENTS, fill_value=0)
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.72, sort=False,
                           marker=dict(colors=[COLORS[s] for s in SENTIMENTS], line=dict(color="#F8F7F2", width=3)),
                           textinfo="percent", textfont_size=13,
                           hovertemplate="%{label}<br>%{value} articles · %{percent}<extra></extra>"))
    fig.add_annotation(text=f"<b>{len(df):,}</b><br>articles", x=.5, y=.5, showarrow=False, font_size=22)
    fig.update_layout(showlegend=True)
    return style(fig, 340)


def stacked_chart(df, group, mode="Share (%)", values="sentiment"):
    data = distribution(df, group, values)
    axis = "share" if mode == "Share (%)" else "count"
    colors = COLORS if values == "sentiment" else TOPIC_COLORS
    order = SENTIMENTS if values == "sentiment" else TOPICS
    groups = df[group].value_counts().index.tolist()
    fig = px.bar(data, y=group, x=axis, color=values, orientation="h", barmode="stack",
                 color_discrete_map=colors, category_orders={values: order},
                 custom_data=["count", "share", "total"])
    fig.update_traces(hovertemplate="%{y}<br>%{fullData.name}: %{customdata[0]} / %{customdata[2]}"
                      "<br>%{customdata[1]:.1f}% within group<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=groups[::-1], title=None)
    fig.update_xaxes(title="Share within group (%)" if axis == "share" else "Annotated articles")
    if axis == "share":
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
    return style(fig, max(310, len(groups) * 31))


def topic_heatmap(df):
    counts = pd.crosstab(df.outlet, df.topic).reindex(columns=TOPICS, fill_value=0)
    counts = counts.reindex(df.outlet.value_counts().index)
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    fig = go.Figure(go.Heatmap(z=shares.values, x=counts.columns, y=counts.index,
                              customdata=counts.values, text=shares.round(0).astype(int).astype(str) + "%",
                              texttemplate="%{text}", colorscale=[[0, "#F5F2EC"], [1, "#6C5BA7"]],
                              zmin=0, zmax=100, colorbar=dict(title="%", thickness=10),
                              hovertemplate="%{y}<br>%{x}: %{customdata} articles<br>%{z:.1f}% within outlet<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(side="top", title=None, tickangle=-20)
    return style(fig, max(350, len(counts) * 32 + 90))


def terms_chart(terms):
    fig = px.bar(terms, x="score", y="term", orientation="h", color_discrete_sequence=["#6C5BA7"])
    fig.update_traces(hovertemplate="%{y}<br>Mean TF-IDF: %{x:.5f}<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=terms.term.tolist()[::-1], title=None)
    fig.update_xaxes(title="Mean TF-IDF weight")
    return style(fig, max(330, len(terms) * 25))
