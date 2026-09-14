from __future__ import annotations

import random
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
NEXT_DATA_RE = re.compile(
    rb'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL
)


class Fetcher:
    def __init__(self, timeout: float, retries: int) -> None:
        self.timeout = timeout
        self.retries = retries

    def get(self, url: str, *, referer: str | None = None) -> bytes:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
        }
        if referer:
            headers["Referer"] = referer
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(Request(url, headers=headers), timeout=self.timeout) as response:
                    return response.read()
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2 ** attempt, 20) + random.random())
        raise RuntimeError(f"request failed after {self.retries + 1} attempts: {url}") from last_error
