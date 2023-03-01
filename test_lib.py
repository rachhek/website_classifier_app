from lib import ScraperTool


def test_scraper_tool():
    st = ScraperTool()
    assert (
        st.is_valid_url("https://www.google.com") == True
    ), "the url was not valid, should be valid"

    assert (
        st.is_valid_url("www.google.com") == False
    ), "url was valid, should be invalid."

    assert (
        st.get_website_name("https://www.google.com") == "google"
    ), "name was not extracted from url."
