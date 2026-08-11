from urllib.parse import quote


def content_disposition(filename: str) -> str:
    """Build an attachment header the browser will honour cross-origin.

    The `download` attribute on an `<a>` is ignored for cross-origin URLs, so
    the filename has to come from the server instead of the markup.
    """
    cleaned = "".join(
        char
        for char in filename.strip().replace("/", "_").replace("\\", "_")
        if char.isprintable() and char != '"'
    )

    if not cleaned:
        cleaned = "speech.mp3"

    ascii_name = cleaned.encode("ascii", "replace").decode("ascii")

    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(cleaned)}"
