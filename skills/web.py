# skills/web.py — open websites and run web searches.

import webbrowser

from skills.registry import skill

_WEBSITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://www.github.com",
    "gmail": "https://mail.google.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "wikipedia": "https://www.wikipedia.org",
}


@skill(
    name="open_website",
    description="Open a website in the default browser.",
    parameters={
        "type": "object",
        "properties": {
            "site": {
                "type": "string",
                "description": "Site name (e.g. youtube, google, github) or a domain/URL.",
            }
        },
        "required": ["site"],
    },
    examples=["open youtube", "go to github", "open reddit.com"],
)
def open_website(args: dict) -> str:
    site = (args.get("site") or "").strip().lower()
    if not site:
        return "No site given."
    if site in _WEBSITES:
        webbrowser.open(_WEBSITES[site])
        return f"Opening {site}."
    if "." in site:  # looks like a domain/URL
        url = site if site.startswith("http") else "https://" + site
        webbrowser.open(url)
        return f"Opening {site}."
    webbrowser.open(f"https://www.google.com/search?q={site}")
    return f"I didn't recognise that site, so I searched for {site}."


@skill(
    name="web_search",
    description="Search the web for a query and open the results in the browser.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
    },
    examples=["search for pasta recipes", "google the weather in Tokyo"],
)
def web_search(args: dict) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "No search query given."
    webbrowser.open("https://www.google.com/search?q=" + query.replace(" ", "+"))
    return f"Searching the web for {query}."
