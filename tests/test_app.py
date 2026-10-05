from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def app():
    return AppTest.from_file(str(APP), default_timeout=30).run()


def assert_clean(at):
    assert not at.exception, [error.message for error in at.exception]


def test_dashboard_starts_with_real_data():
    at = app()
    assert_clean(at)
    assert [m.value for m in at.metric] == ["484", "15", "6", "34.3%"]
    assert [tab.label for tab in at.tabs] == ["Overview", "Outlet comparison", "Language", "Article explorer", "Methodology"]


def test_search_empty_selection_and_reset():
    at = app()
    at.text_input(key="search_query").set_value("housing").run()
    assert_clean(at)
    assert 0 < int(at.metric[0].value) < 484
    at.text_input(key="search_query").set_value("there-is-no-article-with-this-string").run()
    assert_clean(at)
    assert at.metric[0].value == "0"
    at.button[0].click().run()
    assert_clean(at)
    assert at.metric[0].value == "484"


def test_outlet_selection_empty_selection_and_small_sample():
    at = app()
    at.multiselect(key="outlet_filter").set_value(["Fox News"]).run()
    assert_clean(at)
    assert at.metric[0].value == "3"
    assert any("No outlet meets" in item.value for item in at.info)
    at.multiselect(key="outlet_filter").set_value([]).run()
    assert_clean(at)
    assert at.metric[0].value == "0"


def test_language_controls_and_comparison_threshold():
    at = app()
    at.selectbox[2].select("Entire selection").run()
    assert_clean(at)
    at.toggle[0].set_value(False).run()
    assert_clean(at)
    at.number_input[0].set_value(484).run()
    assert_clean(at)
    assert any("No outlet meets" in item.value for item in at.info)
