"""Manufacturing interpretation and bounded OpenRouter transport use source evidence only."""

from dataclasses import replace
from http.server import ThreadingHTTPServer
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from examples.procedural_studio.__main__ import handler_for
from examples.procedural_studio.data import build_demo
from examples.procedural_studio.manufacturing import TEST, inspect_packet
from examples.procedural_studio.router import OpenRouterGuide, RouterError, guidance_context


@pytest.fixture(scope="module")
def payload():
    return build_demo()


@pytest.mark.parametrize("case", TEST, ids=lambda c: c.label)
def test_manufacturing_policy_reads_records_and_not_gold(case):
    finding = inspect_packet(case)
    assert finding.status == case.expected
    assert inspect_packet(replace(case, expected="deliberately-wrong")) == finding


def test_missing_traveler_and_wrong_lot_are_not_sufficient():
    complete = TEST[0]
    report_only = replace(complete, records=complete.records[:1])
    assert inspect_packet(report_only).status == "hold_for_review"
    unrelated = replace(complete, records=tuple(replace(r, lot="OTHER") for r in complete.records))
    assert inspect_packet(unrelated).status == "hold_for_review"


def test_model_context_excludes_scores_and_unobserved_records(payload):
    case = payload["cases"][0]
    start = guidance_context(payload, case["name"], "repaired", -1)
    assert start["observed_records"] == []
    read = guidance_context(payload, case["name"], "repaired", 2)
    assert read["active_procedure"] == "read"
    assert read["observed_records"] == case["records"]
    assert read["outgoing_neighborhood"][0]["target"] == "check"
    assert not {"expected", "runs", "evaluation", "accepted", "rejected"} & read.keys()
    assert "validation_traces" not in json.dumps(read)
    with pytest.raises(ValueError):
        guidance_context(payload, case["name"], "repaired", 99)


def response(advice, **extra):
    return io.BytesIO(
        json.dumps(
            {
                "choices": [{"message": {"content": json.dumps(advice)}}],
                "model": "test/provider-model",
                "usage": {"total_tokens": 27, "cost": 0.001},
                **extra,
            }
        ).encode()
    )


def advice():
    return {
        "next_action": "check",
        "guidance": "Reconcile the withdrawal before the packet advances.",
        "evidence_status": "hold_for_review",
        "citations": ["COR-2409"],
    }


def test_router_uses_fixed_endpoint_server_key_and_reports_real_usage(payload):
    captured = []

    def transport(request, timeout):
        captured.append(request)
        assert timeout == 60
        return response(advice())

    guide = OpenRouterGuide("test-secret", "openrouter/auto", transport=transport)
    context = guidance_context(payload, payload["cases"][0]["name"], "repaired", 2)
    result = guide.guide(context)
    assert captured[0].full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert captured[0].get_header("Authorization") == "Bearer test-secret"
    body = json.loads(captured[0].data)
    assert body["model"] == "openrouter/auto"
    assert body["max_tokens"] == 900
    assert body["reasoning"] == {"effort": "minimal", "exclude": True}
    assert body["response_format"] == {"type": "json_object"}
    assert json.loads(body["messages"][1]["content"]) == context
    assert "test-secret" not in json.dumps(result)
    assert result["usage"]["total_tokens"] == 27
    assert result["model"] == "test/provider-model"


@pytest.mark.parametrize(
    "change",
    [
        {"citations": ["NOT-OBSERVED"]},
        {"citations": [42]},
        {"next_action": "release_lot"},
        {"evidence_status": "safe"},
        {"guidance": ""},
    ],
)
def test_router_rejects_invalid_or_ungrounded_responses(payload, change):
    guide = OpenRouterGuide("secret", transport=lambda *a, **kw: response({**advice(), **change}))
    context = guidance_context(payload, payload["cases"][0]["name"], "repaired", 2)
    with pytest.raises(RouterError, match="invalid guidance"):
        guide.guide(context)


def test_transport_error_does_not_echo_secret_or_provider_body(payload):
    def fail(request, timeout):
        raise HTTPError(request.full_url, 401, "secret", {}, io.BytesIO(b"secret provider body"))

    guide = OpenRouterGuide("secret", transport=fail)
    with pytest.raises(RouterError, match="HTTP 401") as error:
        guide.guide(guidance_context(payload, payload["cases"][0]["name"], "repaired", 2))
    assert "secret" not in str(error.value)


def test_live_endpoint_checks_origin_token_and_scope_without_exposing_key(payload):
    guide = OpenRouterGuide("private-key", transport=lambda *a, **kw: response(advice()))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler_for(payload, guide)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/demo", timeout=5) as res:
                body = res.read()
            assert b"private-key" not in body
            config = json.loads(body)["llm"]
            request_data = {
                "case_id": payload["cases"][0]["name"],
                "checkpoint": "repaired",
                "step": 2,
            }

            def post(data=request_data, **headers):
                req = Request(
                    base + "/api/guide",
                    data=json.dumps(data).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "X-Demo-Token": config["token"],
                        **headers,
                    },
                    method="POST",
                )
                return urlopen(req, timeout=5)

            with post() as res:
                assert json.load(res)["next_action"] == "check"
            for headers in ({"X-Demo-Token": "wrong"}, {"Origin": "https://untrusted.example"}):
                with pytest.raises(HTTPError) as error:
                    post(**headers)
                assert error.value.code == 403
            with pytest.raises(HTTPError) as error:
                post({**request_data, "step": 200})
            assert error.value.code == 400
        finally:
            server.shutdown()
            thread.join(timeout=5)
