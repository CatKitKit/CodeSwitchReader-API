import json
import os
import unittest
from unittest.mock import patch

import app as api


VALID_OBJECT = {
    "translation": "got off the bus",
    "explainHtml": "<p><strong>otobüsten</strong> uses the ablative.</p>",
    "examples": [{"sentence": "Evden çıktım.", "translation": "I left home."}],
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
        self.old_context_mode = os.environ.get(api.AI_CONTEXT_MODE_ENV)
        os.environ[api.AI_CONTEXT_MODE_ENV] = "openrouter"
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
        if self.old_context_mode is None:
            os.environ.pop(api.AI_CONTEXT_MODE_ENV, None)
        else:
            os.environ[api.AI_CONTEXT_MODE_ENV] = self.old_context_mode

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
    def test_default_gemini_mode_bypasses_openrouter_entirely(self, post):
        post.return_value = gemini_response()
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "gemini-test",
        }, clear=True):
            response = self.post()

        self.assert_contract(response)
        self.assertEqual(post.call_count, 1)
        call = post.call_args
        self.assertIn("gemini-3.8-flash", call.args[0])
        self.assertNotIn("openrouter.ai", call.args[0])
        self.assertEqual(call.kwargs["headers"], {"x-goog-api-key": "gemini-test"})
        self.assertNotIn("purpose", call.kwargs["json"])

    @patch("app.requests.post")
    def test_invalid_mode_fails_closed_before_any_provider_call(self, post):
        with patch.dict(os.environ, {
            "AI_CONTEXT_MODE": "surprise-provider",
            "OPENROUTER_API_KEY": "or-test",
            "GEMINI_API_KEY": "gemini-test",
        }):
            response = self.post()

        self.assertEqual(response.status_code, 503)
        post.assert_not_called()

    KEYS = {"OPENROUTER_API_KEY": "or-test", "GEMINI_API_KEY": "gemini-test", "OPENAI_API_KEY": "oa-test"}

    def post_chain(self, payload=None):
        with patch.dict(os.environ, self.KEYS):
            return self.post(payload)

    def payload_for(self, prompt):
        return {**self.payload, "contents": [{"parts": [{"text": prompt}]}]}

    @patch("app.requests.post")
    def test_luna_answers_first_with_low_effort_json_and_a_15_second_cap(self, post):
        post.return_value = openrouter_response(VALID_JSON, reasoning="never reaches the client")
        response = self.post_chain()

        self.assert_contract(response)
        self.assertEqual(post.call_count, 1)
        call = post.call_args
        self.assertEqual(call.args[0], api.OPENAI_CHAT_URL)
        self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer oa-test"})
        body = call.kwargs["json"]
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["messages"], [{"role": "user", "content": self.prompt}])
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(call.kwargs["timeout"].total, 15)
        self.assertEqual(call.kwargs["timeout"].connect_timeout, 3.05)

    @patch("app.requests.post")
    def test_hard_languages_ask_gemini_first_then_luna(self, post):
        hard = (
            'clicked on the word/phrase: "x" (Source Language: ka-GE).\nProvide all explanations and translations in English.',
            'clicked on the word/phrase: "x" (Source Language: kk).\nProvide all explanations and translations in English.',
            'clicked on the word/phrase: "x" (Source Language: zh-HK).\nProvide all explanations and translations in English.',
            'clicked on the word/phrase: "x" (Source Language: yue-HK).\nProvide all explanations and translations in English.',
            # The learner reads the explanation in a hard language.
            'clicked on the word/phrase: "x" (Source Language: fr-FR).\nProvide all explanations and translations in Bangla.',
        )
        for prompt in hard:
            with self.subTest(prompt=prompt):
                post.reset_mock()
                post.side_effect = [FakeResponse(503, {}), openrouter_response(VALID_JSON)]
                response = self.post_chain(self.payload_for(prompt))
                self.assert_contract(response)
                self.assertIn("gemini-3.8-flash", post.call_args_list[0].args[0])
                self.assertEqual(post.call_args_list[1].args[0], api.OPENAI_CHAT_URL)

    def test_ordinary_and_unknown_languages_are_not_gemini_first(self):
        for prompt in (
            'clicked on the word/phrase: "x" (Source Language: sw-KE).\nProvide all explanations and translations in English.',
            'clicked on the word/phrase: "x" (Source Language: zh-CN).\nProvide all explanations and translations in Mandarin.',
            'clicked on the word/phrase: "x" (Source Language: Unknown).\nProvide all explanations and translations in English.',
            self.prompt,
        ):
            with self.subTest(prompt=prompt):
                self.assertFalse(api._context_gemini_first(prompt))

    @patch("app.requests.post")
    def test_gemma_is_last_and_strictly_pinned_private(self, post):
        post.side_effect = [FakeResponse(503, {}), FakeResponse(503, {}), openrouter_response(VALID_JSON)]
        response = self.post_chain()

        self.assert_contract(response)
        self.assertEqual(response.get_json()["modelVersion"], "coreweave")
        self.assertEqual(post.call_count, 3)
        self.assertEqual(post.call_args_list[0].args[0], api.OPENAI_CHAT_URL)
        self.assertIn("gemini-3.8-flash", post.call_args_list[1].args[0])
        call = post.call_args_list[2]
        self.assertEqual(call.args[0], api.CONTEXT_OPENROUTER_URL)
        self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer or-test"})
        body = call.kwargs["json"]
        self.assertEqual(body["model"], "google/gemma-4-31b-it")
        self.assertEqual(body["messages"], [{"role": "user", "content": self.prompt}])
        self.assertEqual(body["provider"], {
            "only": ["coreweave"],
            "order": ["coreweave"],
            "allow_fallbacks": False,
            "zdr": True,
            "data_collection": "deny",
        })

    @patch("app.requests.post")
    def test_luna_timeout_gives_gemini_a_fresh_attempt(self, post):
        post.side_effect = [api.requests.Timeout("synthetic"), gemini_response()]
        response = self.post_chain()

        self.assert_contract(response)
        self.assertEqual(post.call_count, 2)
        gemini = post.call_args_list[1]
        self.assertIn("gemini-3.8-flash", gemini.args[0])
        self.assertNotIn("?key=", gemini.args[0])
        self.assertEqual(gemini.kwargs["headers"], {"x-goog-api-key": "gemini-test"})
        self.assertEqual(gemini.kwargs["timeout"].total, 15)
        self.assertNotIn("purpose", gemini.kwargs["json"])
        self.assertEqual(
            gemini.kwargs["json"]["generationConfig"]["thinkingConfig"],
            {"thinkingLevel": "low"},
        )
        self.assertEqual(gemini.kwargs["json"]["contents"][0]["parts"][0]["text"], self.prompt)

    @patch("app.requests.post")
    def test_busy_gone_down_or_unreachable_each_fall_through(self, post):
        for first in (FakeResponse(429, {}), FakeResponse(404, {}), FakeResponse(500, {}),
                      FakeResponse(503, {}), api.requests.ConnectionError("synthetic")):
            with self.subTest(first=first):
                post.reset_mock()
                post.side_effect = [first, gemini_response()]
                self.assert_contract(self.post_chain())
                self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_unusable_luna_json_shapes_each_fall_through(self, post):
        bad_outputs = (
            "not json",
            '{"translation":"x","explainHtml":"<p>x</p>"}',
            '{"translation":" ","explainHtml":"<p>x</p>","examples":[{"sentence":"S.","translation":"T."}]}',
            '{"translation":3,"explainHtml":"<p>x</p>","examples":[{"sentence":"S.","translation":"T."}]}',
            '{"translation":"x","explainHtml":"<p></p>","examples":[{"sentence":"S.","translation":"T."}]}',
            '{"translation":"x","explainHtml":"<p>&nbsp;</p>","examples":[{"sentence":"S.","translation":"T."}]}',
            # The pre-10/1 phone shape is gone: a provider answering in it falls through.
            '{"translation":"x","explainHtml":"<p>x</p>","examplesHtml":"<ol><li>S.</li></ol>"}',
        )
        for raw in bad_outputs:
            with self.subTest(raw=raw):
                post.reset_mock()
                post.side_effect = [openrouter_response(raw), gemini_response()]
                self.assert_contract(self.post_chain())
                self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_invalid_provider_json_truncation_and_refusal_fall_through(self, post):
        firsts = (
            FakeResponse(200, ValueError("synthetic bad JSON envelope")),
            openrouter_response(VALID_JSON[:-1]),
            openrouter_response(VALID_JSON, finish_reason="length"),
            openrouter_response(None, finish_reason="content_filter"),
        )
        for first in firsts:
            with self.subTest(first=first):
                post.reset_mock()
                post.side_effect = [first, gemini_response()]
                self.assert_contract(self.post_chain())
                self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_inline_thinking_and_extra_tail_brace_are_recovered_safely(self, post):
        post.return_value = openrouter_response(
            "<think>private chain of thought</think>\n```json\n" + VALID_JSON + "\n}```"
        )
        self.assert_contract(self.post_chain())
        self.assertEqual(post.call_count, 1)

    @patch("app.requests.post")
    def test_one_providers_key_or_request_error_moves_on_to_the_next(self, post):
        for status in (400, 401, 403):
            with self.subTest(status=status):
                post.reset_mock()
                post.side_effect = [FakeResponse(status, {"error": "synthetic"}), gemini_response()]
                self.assert_contract(self.post_chain())
                self.assertEqual(post.call_count, 2)

    @patch("app.requests.post")
    def test_any_missing_key_fails_closed_before_any_call(self, post):
        for missing in self.KEYS:
            with self.subTest(missing=missing):
                env = {k: v for k, v in self.KEYS.items() if k != missing}
                with patch.dict(os.environ, {**env, "AI_CONTEXT_MODE": "openrouter"}, clear=True):
                    response = self.post()
                self.assertEqual(response.status_code, 503)
                post.assert_not_called()

    @patch("app.requests.post")
    def test_all_three_failing_returns_one_backend_error(self, post):
        post.side_effect = [FakeResponse(503, {}), FakeResponse(503, {}), FakeResponse(503, {})]
        response = self.post_chain()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json(), {"error": "AI context unavailable"})
        self.assertEqual(post.call_count, 3)

    @patch("app.requests.post")
    def test_unusable_answers_are_never_returned_to_the_phone(self, post):
        post.side_effect = [
            FakeResponse(500, {}),
            gemini_response('{"translation":"x","explainHtml":"<p>x</p>"}'),
            openrouter_response("not json"),
        ]
        response = self.post_chain()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("candidates", response.get_json())
        self.assertEqual(post.call_count, 3)

    @patch("app.time.monotonic")
    @patch("app.requests.post")
    def test_chain_fits_the_phones_40_second_wait(self, post, clock):
        # Luna times out at 15 s, Gemini at 15 s more: Gemma gets the 6 s left, not 15.
        clock.side_effect = [0, 0, 15, 30]
        post.side_effect = [api.requests.Timeout("a"), api.requests.Timeout("b"), openrouter_response(VALID_JSON)]
        self.assert_contract(self.post_chain())
        self.assertEqual([c.kwargs["timeout"].total for c in post.call_args_list], [15, 15, 6])

        # Less than 4 s left: give up rather than start a call the phone won't wait for.
        post.reset_mock()
        clock.side_effect = [0, 0, 15, 33]
        post.side_effect = [api.requests.Timeout("a"), api.requests.Timeout("b")]
        self.assertEqual(self.post_chain().status_code, 502)
        self.assertEqual(post.call_count, 2)

    def test_example_pairs_are_cleaned_and_replace_the_html_list(self):
        raw = json.dumps({
            "translation": "left",
            "explainHtml": "<p>x</p>",
            "examples": [
                {"sentence": "<strong>Evden</strong>  çıktım.", "translation": "I left home &amp; ran."},
                {"sentence": "Gato.", "translation": ""},
                {"sentence": "  ", "translation": "dropped: no sentence"},
                {"sentence": "Dropped.", "translation": 3},
                "not an object",
            ] + [{"sentence": f"S{n}.", "translation": f"T{n}."} for n in range(6)],
            "examplesHtml": "<ol><li>ignored</li></ol>",
        }, ensure_ascii=False)
        clean = json.loads(api._context_contract_json(raw))
        self.assertNotIn("examplesHtml", clean)
        self.assertEqual(clean["examples"][0], {"sentence": "Evden çıktım.", "translation": "I left home & ran."})
        self.assertEqual(clean["examples"][1], {"sentence": "Gato.", "translation": ""})
        self.assertEqual(len(clean["examples"]), api.CONTEXT_MAX_EXAMPLES)

    def test_no_usable_example_is_rejected(self):
        for examples in ([], [{"sentence": "", "translation": "x"}], "not a list"):
            with self.subTest(examples=examples):
                raw = json.dumps({"translation": "x", "explainHtml": "<p>x</p>", "examples": examples})
                self.assertIsNone(api._context_contract_json(raw))

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
        self.assertIn("gemini-3.8-flash", post.call_args.args[0])
        self.assertNotIn("openrouter.ai", post.call_args.args[0])

    UNMARKED = {"contents": [{"parts": [{"text": "ordinary Studio prompt"}]}]}

    def post_unmarked(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test"}):
            return self.post(self.UNMARKED)

    @patch.object(api.requests, "post")
    def test_unmarked_busy_or_down_35_falls_to_31(self, post):
        for failure in (FakeResponse(503, {"error": {}}), FakeResponse(429, {}),
                        FakeResponse(404, {}), api.requests.Timeout("synthetic"),
                        api.requests.ConnectionError("synthetic")):
            post.reset_mock()
            post.side_effect = [failure, FakeResponse(200, {"candidates": ["ok"]})]
            response = self.post_unmarked()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {"candidates": ["ok"]})
            self.assertEqual(post.call_count, 2)
            self.assertIn("gemini-3.8-flash", post.call_args_list[0].args[0])
            self.assertIn("gemini-3.5-flash-lite", post.call_args_list[1].args[0])

    @patch.object(api.requests, "post")
    def test_unmarked_client_error_moves_on_instead_of_stopping(self, post):
        # A rejected Google key (400/401/403) tries 3.5, then the other companies (none here).
        post.return_value = FakeResponse(400, {"error": {"code": 400}})
        response = self.post_unmarked()
        self.assertEqual(post.call_count, 2)
        self.assertEqual((response.status_code, response.get_json()), (502, {"error": "Upstream error"}))

    @patch.object(api.requests, "post")
    def test_unmarked_both_models_failing_without_backup_key_is_one_502(self, post):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test"}):
            os.environ.pop("OPENROUTER_API_KEY", None)
            post.side_effect = [FakeResponse(503, {}), FakeResponse(503, {"error": {"code": 503}})]
            response = self.post(self.UNMARKED)
            self.assertEqual((response.status_code, response.get_json()), (502, {"error": "Upstream error"}))
            post.side_effect = [api.requests.Timeout("a"), api.requests.Timeout("b")]
            self.assertEqual(self.post(self.UNMARKED).status_code, 502)
            self.assertEqual(post.call_count, 4)


if __name__ == "__main__":
    unittest.main()
