def get_url_cat():
    from requests import get
    from json import loads
    url = "https://api.thecatapi.com/v1/images/search"
    response = get(
        url=url)
    return (response.json()[0]["url"])
