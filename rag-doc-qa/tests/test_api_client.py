import pytest
import requests

from ui.api_client import ApiClient, ApiError


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """Records requests and returns queued responses (or raises an error)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client_with(*responses):
    session = FakeSession(*responses)
    return ApiClient("http://api.test/", session=session), session


def test_ask_posts_the_question_and_returns_json():
    client, session = client_with(FakeResponse(200, {"answer": "14 days", "sources": []}))
    assert client.ask("How long?") == {"answer": "14 days", "sources": []}
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "http://api.test/ask")
    assert kwargs["json"] == {"question": "How long?"}


def test_upload_sends_a_multipart_pdf():
    client, session = client_with(FakeResponse(201, {"filename": "a.pdf", "pages": 1, "chunks": 1}))
    client.upload("a.pdf", b"%PDF-data")
    _, url, kwargs = session.calls[0]
    assert url == "http://api.test/documents"
    assert kwargs["files"]["file"] == ("a.pdf", b"%PDF-data", "application/pdf")


def test_delete_escapes_unusual_file_names():
    client, session = client_with(FakeResponse(204))
    client.delete("my notes #1.pdf")
    method, url, _ = session.calls[0]
    assert method == "DELETE"
    assert url == "http://api.test/documents/my%20notes%20%231.pdf"


def test_server_error_message_is_passed_on():
    client, _ = client_with(FakeResponse(400, {"detail": "Only PDF files are accepted."}))
    with pytest.raises(ApiError) as err:
        client.upload("a.txt", b"x")
    assert "Only PDF" in str(err.value)


def test_validation_errors_are_flattened():
    body = {"detail": [{"msg": "String should have at least 1 character"}]}
    client, _ = client_with(FakeResponse(422, body))
    with pytest.raises(ApiError) as err:
        client.ask("")
    assert "at least 1 character" in str(err.value)


def test_non_json_error_falls_back_to_status_code():
    client, _ = client_with(FakeResponse(500, None))
    with pytest.raises(ApiError) as err:
        client.health()
    assert "500" in str(err.value)


def test_unreachable_server_gives_friendly_message():
    client, _ = client_with(requests.exceptions.ConnectionError("refused"))
    with pytest.raises(ApiError) as err:
        client.health()
    assert "Cannot reach the API" in str(err.value)
