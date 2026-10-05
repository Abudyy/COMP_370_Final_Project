# Media Lens — Exploring Mamdani News Coverage

An interactive dashboard for a COMP 370 group research project on news coverage of Zohran Mamdani. Explore **484 manually annotated articles from 15 outlets**, compare topic and sentiment distributions, inspect TF-IDF language patterns, and trace findings back to source articles.

**Stack:** Python · pandas · scikit-learn · Streamlit · Plotly

## Explore the dashboard

| View | What you can explore |
| --- | --- |
| Overview | Coverage by outlet, sentiment composition, and sentiment within topics |
| Outlet comparison | Counts or within-outlet percentages, a minimum sample size, topic heatmaps, and paired outlet views |
| Language | Original TF-IDF weighting by topic, outlet, or filtered selection; optional term hiding; downloadable term scores |
| Article explorer | Searchable annotations, original source links, article details, and CSV export |
| Methodology | Data provenance, annotation limits, URL matching, and explicitly labeled simulated outlet scores |

Sidebar filters for outlet, topic, sentiment, and literal text search apply to all analytical views. Plotly charts support hover details and image downloads.

## Run locally

Use **Python 3.11 or 3.12**. The dashboard runs entirely on committed data; it needs no API keys, external databases, or fresh scraping.

```bash
git clone https://github.com/Abudyy/COMP_370_Final_Project.git
cd COMP_370_Final_Project
# Before the dashboard PR is merged, check out its branch:
git checkout feature/media-lens-dashboard
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. Once the dashboard is merged, the branch checkout is unnecessary.

## Deploy on Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account.
2. Create an app with repository `Abudyy/COMP_370_Final_Project`, branch `feature/media-lens-dashboard` (or `main` after merging), and main file `app.py`.
3. Select Python 3.12 in advanced settings and deploy. Dependencies are installed from `requirements.txt`; no secrets are required.
4. Add the resulting public app URL to this README, your personal website, and your resume project entry.

See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) for the current deployment interface. The dashboard code is deployment-ready; this repository does not claim that a public app has already been deployed.

## Findings from the committed sample

These are descriptive results, not estimates of all media coverage:

- **New York Post contributes 182 of 484 articles (37.6%)**, making outlet representation uneven.
- Sentiment annotations are **191 Neutral (39.5%)**, **166 Negative (34.3%)**, and **127 Positive (26.2%)**.
- **Election** is the largest topic category, with 102 articles (21.1%).
- In the **Opinion** category, 42 of 84 articles (50.0%) are labeled Negative; in **Election**, 16 of 102 (15.7%) are labeled Negative. Topic composition matters when comparing outlets.

The dashboard recalculates the summaries when filters change and displays the corresponding denominators.

## Data and methods

### Authoritative annotations

`data/annotation/final_full_annotated.csv` contains 484 rows, each with an outlet ID, title, description, one topic category, and a polarity label. The original annotation file records three annotator names: Abdullah, Antonin, and Kihyeok. The dashboard preserves those labels but omits annotator names from the article explorer and its exports.

The six categories are Election, Policy, Social Response, Opinion, Political Identity, and Political Interaction. Sentiment labels are Negative, Neutral, and Positive. These are **human annotations**, not predictions from a trained sentiment model. The repository does not include a coding rubric or overlapping ratings needed to compute inter-rater agreement.

### Article links

`data/reformated_articles_final.json` stores 491 article records. Links are joined using outlet IDs and normalized titles, with explicit source-ID aliases. The join resolves **481 of 484** annotations; three unmatched records remain visible without a guessed URL. The annotation table drives every chart, so unmatched links do not change article counts.

### TF-IDF

The dashboard reproduces `data/annotation/tf-idf.py`:

- Text = title + description.
- Original `New York` / `new york` → `New_York` substitution.
- `TfidfVectorizer(stop_words="english", ngram_range=(1, 2))`, with default remaining parameters.
- Mean document TF-IDF weights within the selected group.

Vocabulary and inverse document frequency are fitted once to the complete 484-row corpus. Filters change the rows whose weights are averaged, keeping the weighting comparable across selections. Tests check all saved top-ten category scores in `data/annotation/tfidf_top_words.txt` to a tolerance of `1e-12`.

The display toggles hide a documented set of candidate-name or snippet-placeholder terms **after scoring**; they do not clean the input corpus or recalculate IDF. The original saved rankings remain available in the Language view.

### Limitations

- The collected sample is not representative of all North American media coverage; outlet sample sizes and topic mixes differ.
- The analysis uses headlines and descriptions, not full article bodies. Some descriptions contain snippet placeholders.
- The annotation file has no publication-date field, so there is no inferred time-series chart.
- Human sentiment labels do not establish an outlet's political bias, causal effects, or statistical significance.
- `data/media_bias.json` contains **simulated**, unvalidated outlet scores. They appear only in an optional methodology expander, labeled as simulated. Unmapped outlets remain explicit.

## Repository structure

```text
app.py                      Streamlit dashboard entry point
dashboard/data.py           Loading, conservative URL joins, filtering, TF-IDF, CSV export
dashboard/charts.py         Plotly charts and shared visual styling
data/annotation/            Original human annotations and saved TF-IDF results
data/                       Original collected articles and metadata
scripts/                    Original collection and preprocessing scripts
tests/                      Data correctness and Streamlit interaction tests
.streamlit/config.toml      Theme and application configuration
requirements.txt            Dashboard runtime dependencies
requirements-dev.txt        Test dependencies
```

The original scraper and preprocessing scripts are preserved. Their collection dependencies and credentials are separate from the dashboard runtime. Running the dashboard does not execute scraping or call a news API.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover annotation integrity, known counts, URL aliases and unmatched records, explicit unmapped score handling, literal search and filter composition, within-group percentage denominators, saved TF-IDF results, exports, app startup, empty selections, small samples, and reset behavior.

## Presenting this on a resume

Suggested project title: **Media Lens — News Coverage Analytics**.

Use wording that reflects your own contribution to the group project. A team-level description to adapt:

> Collaborated on a Python news analysis project covering 484 manually annotated articles across 15 outlets; used pandas and scikit-learn TF-IDF to examine topic and sentiment patterns, and presented the findings in an interactive Streamlit/Plotly dashboard.

In an interview, explain the research question, how articles were collected and deduplicated, how labels were assigned, why TF-IDF was used, and how sampling imbalance and short descriptions limit conclusions. Be ready to walk through the dashboard additions and explain the code you use.

## Credits

Original COMP 370 group project and annotations: Abdullah, Antonin, and Kihyeok. News content and source links belong to their respective publishers.
