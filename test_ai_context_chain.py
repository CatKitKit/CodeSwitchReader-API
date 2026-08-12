import json
import os
import unittest
from unittest.mock import patch

import app as api


VALID_OBJECT = {
    "translation": "got off the bus",
    "explainHtml": "<p><strong>otobüsten</strong> uses the ablative.</p>",
    "examplesHtml": "<ol><li><strong>Evden çıktım.</strong> I left home.</li></ol>",
}
VALID_JSON = json.dumps(VALID_OBJECT, ensure_ascii=False)


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def openrouter_response(text, finish_reason="stop", reasoning=None):
    message = {"content": text}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return FakeResponse(200, {
        "choices": [{"message": message, "finish_reason": finish_reason}]
    })


def gemini_response(text=VALID_JSON, finish_reason="STOP", thought="private thought"):
    parts = []
    if thought is not None:
        parts.append({"text": thought, "thought": True})
    parts.append({"text": text})
    return FakeResponse(200, {
        "candidates": [{
            "content": {"parts": parts},
            "finishReason": finish_reason,
        }]
    })


class AiContextChainTest(unittest.TestCase):
    def setUp(self):
        self.old_app_key = api.APP_KEY
        api.APP_KEY = "test-app-key"
        with api._rate_lock:
            api._ip_hits.clear()
            api._daily.update({"day": None, "count": 0})
        self.client = api.app.test_client()
        self.headers = {"X-App-Key": "test-app-key"}
        self.prompt = "the exact existing production context prompt"
        self.payload = {
            "contents": [{"parts": [{"text": self.prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
            "purpose": "context_explanation",
        }

    def tearDown(self):
        api.APP_KEY = self.old_app_key

    def post(self, payload=None):
        return self.client.post(
            "/ai-proxy",
            json=self.payload if payload is None else payload,
            headers=self.headers,
        )

    def assert_contract(self, response):
        self.assertEqual(response.status_code, 200)
        text = response.get_json()["candidates"][0]["content"]["parts"][0]["text"]
        self.assertEqual(json.loads(text), VALID_OBJECT)

    @patch("app.requests.post")
    def test_venice_success_is_strictly_pinned_private_and_gemini_shaped(self, post):
        post.return_value = openrouter_response(
            VALID_JSON, reasoning="This must never reach the client"
        )
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "or-test",
            "GEMINI_API_KEY": "gemini-test",
        }):
            response = self.post()

        self.assert_contract(response)
        self.assertEqual(post.call_count, 1)
        call = post.call_args
        self.assertEqual(call.args[0], api.CONTEXT_OPENROUTER_URL)
        timeout = call.kwargs["timeout"]
        self.assertEqual(timeout.total, 18)
        self.assertEqual(timeout.connect_timeout, 3.05)
        self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer or-test"})
        body = call.kwargs["json"]
        self.assertEqual(body["model"], "google/gemma-4-31b-it")
        self.assertEqual(body["messages"], [{"role": "user", "content": self.prompt}])
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["provider"], {
            "only": ["venice"],
            "order": ["venice"],
            "allow_fallbacks": False,
            "zdr": True,
            "data_collection": "deny",
        })

    @patch("app.requests.post")
    def test_timeout_gives_gemini_a_fresh_18_second_attempt(self, post):
        post.side_effect = [api.requests.Timeout("synthetic"), gemini_response()]
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "or-test",
            "GEMINI_API_KEY": "gemini-test",
        }):
            response = self.post()

        self.assert_contract(response)
        self.assertEqual(post.call_count, 2)
        gemini = post.call_args_list[1]
        self.assertIn("gemini-3.5-flash-lite", gemini.args[0])
        self.assertNotIn("?key=", gemini.args[0])
        self.assertEqual(gemini.kwargs["headers"], {"x-goog-api-key": "gemini-test"})
        timeout = gemini.kwargs["timeout"]
        self.assertEqual(timeout.total, 18)
        self.assertEqual(timeout.connect_timeout, 3.05)
        self.assertNotIn("purpose", gemini.kwargs["json"])
        self.assertEqual(
            gemini.kwargs["json"]["generationConfig"]["thinkingConfig"],
            {"thinkingLevel": "minimal"},
        )
        self.assertEqual(
            gemini.kwargs["json"]["contents"][0]["parts"][0]["text"],
            self.prompt,
        )

    @patch("app.requests.post")
    def test_429_and_5xx_each_fall_through_to_gemini(self, post):
        for status in (429, 500, 502, 503):
            with self.subTest(status=status):
                post.reset_mock()
                post.side_effect = [FakeResponse(status, {}), gemini_response()]
                with patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "or-test",
                    "GEMINI_API_KEY": "gemini-test",
                }):
                    response = self.post()
                self.assert_contract(response)
                self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_unusable_venice_json_shapes_each_fall_through(self, post):
        bad_outputs = (
            "not json",
            '{"translation":"x","explainHtml":"<p>x</p>"}',
            '{"translation":" ","explainHtml":"<p>x</p>","examplesHtml":"<ol>x</ol>"}',
            '{"translation":3,"explainHtml":"<p>x</p>","examplesHtml":"<ol>x</ol>"}',
            '{"translation":"x","explainHtml":"<p></p>","examplesHtml":"<ol></ol>"}',
            '{"translation":"x","explainHtml":"<p>&nbsp;</p>","examplesHtml":"<ol>&nbsp;</ol>"}',
        )
        for raw in bad_outputs:
            with self.subTest(raw=raw):
                post.reset_mock()
                post.side_effect = [openrouter_response(raw), gemini_response()]
                with patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "or-test",
                    "GEMINI_API_KEY": "gemini-test",
                }):
                    response = self.post()
                self.assert_contract(response)
                self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_invalid_provider_json_and_truncation_fall_through(self, post):
        first_responses = (
            FakeResponse(200, ValueError("synthetic bad JSON envelope")),
            openrouter_response(VALID_JSON[:-1]),
            openrouter_response(VALID_JSON, finish_reason="length"),
        )
        for first in first_responses:
            with self.subTest(first=first):
                post.reset_mock()
                post.side_effect = [first, gemini_response()]
                with patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "or-test",
                    "GEMINI_API_KEY": "gemini-test",
                }):
                    response = self.post()
                self.assert_contract(response)
                self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_inline_thinking_and_extra_tail_brace_are_recovered_safely(self, post):
        post.return_value = openrouter_response(
            "<think>private chain of thought</think>\n```json\n" + VALID_JSON + "\n}```"
        )
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "or-test",
            "GEMINI_API_KEY": "gemini-test",
        }):
            response = self.post()
        self.assert_contract(response)
        self.assertEqual(post.call_count, 1)

    @patch("app.requests.post")
    def test_400_401_403_surface_without_gemini_fallback(self, post):
        for status in (400, 401, 403):
            with self.subTest(status=status):
                post.reset_mock()
                post.side_effect = None
                post.return_value = FakeResponse(status, {"error": "synthetic"})
                with patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "or-test",
                    "GEMINI_API_KEY": "gemini-test",
                }):
                    response = self.post()
                self.assertEqual(response.status_code, status)
                self.assertEqual(post.call_count, 1)

    @patch("app.requests.post")
    def test_missing_primary_configuration_does_not_silently_skip_to_gemini(self, post):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test"}, clear=True):
            response = self.post()
        self.assertEqual(response.status_code, 503)
        post.assert_not_called()

    @patch("app.requests.post")
    def test_both_providers_failing_returns_one_backend_error(self, post):
        post.side_effect = [FakeResponse(503, {}), FakeResponse(503, {})]
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "or-test",
            "GEMINI_API_KEY": "gemini-test",
        }):
            response = self.post()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json(), {"error": "AI context unavailable"})
        self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_unusable_gemini_fallback_is_not_returned_to_the_phone(self, post):
        post.side_effect = [
            FakeResponse(500, {}),
            gemini_response('{"translation":"x","explainHtml":"<p>x</p>"}'),
        ]
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "or-test",
            "GEMINI_API_KEY": "gemini-test",
        }):
            response = self.post()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("candidates", response.get_json())
        self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_gemini_fallback_client_errors_keep_their_status(self, post):
        for status in (400, 401, 403):
            with self.subTest(status=status):
                post.reset_mock()
                post.side_effect = [FakeResponse(500, {}), FakeResponse(status, {})]
                with patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "or-test",
                    "GEMINI_API_KEY": "gemini-test",
                }):
                    response = self.post()
                self.assertEqual(response.status_code, status)
                self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_bad_or_unknown_purpose_is_rejected_before_upstream(self, post):
        bad = {**self.payload, "purpose": "summary"}
        response = self.post(bad)
        self.assertEqual(response.status_code, 400)
        post.assert_not_called()

        marked_studio_shape = {
            **self.payload,
            "systemInstruction": {"parts": [{"text": "You are a writer."}]},
        }
        response = self.post(marked_studio_shape)
        self.assertEqual(response.status_code, 400)
        post.assert_not_called()

    @patch("app.requests.post")
    def test_unmarked_ai_proxy_calls_stay_on_existing_gemini_model(self, post):
        post.return_value = FakeResponse(200, {"candidates": []})
        unmarked = {
            "contents": [{"parts": [{"text": "ordinary summary or Studio prompt"}]}]
        }
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "or-test",
            "GEMINI_API_KEY": "gemini-test",
        }):
            response = self.post(unmarked)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 1)
        self.assertIn("gemini-3.1-flash-lite", post.call_args.args[0])
        self.assertNotIn("openrouter.ai", post.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
