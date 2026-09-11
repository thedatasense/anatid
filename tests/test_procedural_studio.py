"""The visual must replay real Anatid outcomes and export without external assets."""

from http.server import HTTPServer
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from examples.procedural_studio.__main__ import handler_for, standalone
from examples.procedural_studio.data import build_demo


@pytest.fixture(scope="module")
def payload():
    return build_demo()


def test_visual_data_contains_the_retained_and_historical_database_graphs(payload):
    graphs = payload["graphs"]
    assert graphs["original"] == graphs["historical"]
    assert graphs["repaired"] == graphs["retained"]
    assert payload["accepted"]["accepted"] is True
    assert payload["rejected"]["accepted"] is False
    assert payload["negative_memories"] == [json.loads(json.dumps(payload["rejected"]))]
    assert payload["evaluation"]["original"]["passed"] == 1
    assert payload["evaluation"]["repaired"]["passed"] == 5
    assert len(payload["provenance"]) == 2
    assert all(isinstance(record["id"], str) for record in payload["provenance"])


def test_visual_cases_include_withdrawal_complete_packet_and_hold(payload):
    outdated = payload["cases"][0]["runs"]
    assert outdated["original"]["answer"] == "ready_for_review"
    assert outdated["repaired"]["answer"] == "hold_for_review"
    assert "COR-2409" in outdated["repaired"]["citations"]
    assert "withdrawn" in " ".join(outdated["repaired"]["reasons"])
    assert payload["cases"][1]["runs"]["repaired"]["answer"] == "ready_for_review"
    missing = payload["cases"][-1]["runs"]["repaired"]
    assert missing["actions"][-1] == "abstain"
    assert missing["success"] is True
    assert missing["answer"] == "hold_for_review"


def test_standalone_inlines_assets_and_escapes_script_closing_tags(payload):
    html = standalone(
        {
            **payload,
            "probe": "</script><script>bad()</script>",
            "llm": {"enabled": True, "token": "do-not-export-this-token"},
        }
    )
    assert '<script src="/app.js"' not in html
    assert '<link rel="stylesheet"' not in html
    assert "/brand/" not in html
    assert "data:image/svg+xml;base64," in html
    assert '<script id="demo-data" type="application/json">' in html
    assert "<script>bad()" not in html
    assert "do-not-export-this-token" not in html
    encoded = html.split('<script id="demo-data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(encoded)["graphs"] == payload["graphs"]
    assert json.loads(encoded)["probe"] == "</script><script>bad()</script>"
    assert json.loads(encoded)["llm"]["enabled"] is False


def test_http_serves_only_demo_assets_and_replay_data(payload):
    with HTTPServer(("127.0.0.1", 0), handler_for(payload)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/demo", timeout=5) as response:
                assert json.load(response)["graphs"] == payload["graphs"]
            for path, mime in (
                ("/", "text/html"),
                ("/app.js", "text/javascript"),
                ("/style.css", "text/css"),
                ("/brand/anatid-logo.svg", "image/svg+xml"),
                ("/brand/favicon.svg", "image/svg+xml"),
                ("/favicon.ico", "image/x-icon"),
            ):
                with urlopen(base + path, timeout=5) as response:
                    assert response.headers.get_content_type() == mime
                    assert response.read()
            with pytest.raises(HTTPError) as error:
                urlopen(base + "/../data.py", timeout=5)
            assert error.value.code == 404
        finally:
            server.shutdown()
            thread.join(timeout=5)
