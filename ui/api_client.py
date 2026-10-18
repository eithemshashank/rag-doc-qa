"""Step 5a: a small client for our own API.

The Streamlit page never touches the database or the LLM. It only calls the
API through this class, so front end and back end stay separate.
"""
from urllib.parse import quote

import requests


class ApiError(Exception):
    """Something went wrong; the message is safe to show to the user."""


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", session=None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self.timeout = timeout

    def _call(self, method: str, path: str, **kwargs):
        try:
            response = self._session.request(
                method, self.base_url + path, timeout=self.timeout, **kwargs
            )
        except requests.exceptions.RequestException:
            raise ApiError(
                f"Cannot reach the API at {self.base_url}. Is the server running?"
            ) from None
        if response.status_code >= 400:
            raise ApiError(self._detail(response))
        return response

    @staticmethod
    def _detail(response) -> str:
        """Pull the server's error message out of a failed response."""
        try:
            detail = response.json()["detail"]
        except (ValueError, KeyError, TypeError):
            return f"Request failed (HTTP {response.status_code})."
        if isinstance(detail, list):  # validation errors come as a list
            return "; ".join(str(item.get("msg", item)) for item in detail)
        return str(detail)

    def health(self) -> dict:
        return self._call("GET", "/health").json()

    def list_documents(self) -> list[dict]:
        return self._call("GET", "/documents").json()

    def upload(self, filename: str, data: bytes) -> dict:
        files = {"file": (filename, data, "application/pdf")}
        return self._call("POST", "/documents", files=files).json()

    def delete(self, filename: str) -> None:
        self._call("DELETE", f"/documents/{quote(filename, safe='')}")

    def ask(self, question: str) -> dict:
        return self._call("POST", "/ask", json={"question": question}).json()
