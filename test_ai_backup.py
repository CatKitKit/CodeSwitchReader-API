"""Gemma (ModelRun) backup behind ordinary /ai-proxy calls, and the translation route."""
import os
import unittest
from unittest.mock import patch

import app as api
from test_ai_context_chain import FakeResponse, openrouter_response

KEYS = {"GEMINI_API_KEY": "gemini-test", "OPENROUTER_API_KEY": "or-test"}
STORY = {
    "contents": [{"parts": [{"text": "write a story"}]}],
    "systemInstruction": {"parts": [{"text": "You are an expert story writer."}]},
}
LESSON = dict(STORY, generationConfig={"responseMimeType": "application/json"})
TRANSLATION = dict(STORY, purpose="translation")
GEMINI_OK = {"candidates": [{"content": {"parts": [{"text": "gemini text"}]}, "finishReason": "STOP"}]}


class AiBackupTest(unittest.TestCase):
    def setUp(self):
        self.old_app_key = api.APP_KEY
        api.APP_KEY = "test-app-key"
        with api._rate_lock:
            api._ip_hits.clear()
            api._daily.update({"day": None, "count": 0})
        self.client = api.app.test_client()

    def tearDown(self):
        api.APP_KEY = self.old_app_key

    def post(self, payload, env=KEYS):
        with patch.dict(os.environ, env, clear=False):
            for name in set(KEYS) - set(env):
                os.environ.pop(name, None)
            return self.client.post("/ai-proxy", json=payload, headers={"X-App-Key": "test-app-key"})

    def assert_gemma_answer(self, response, text):
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["candidates"][0]["content"]["parts"][0]["text"], text)
        self.assertIn("modelrun", data["modelVersion"])

    @patch.object(api.requests, "post")
    def test_google_down_falls_to_pinned_private_modelrun(self, post):
        post.side_effect = [FakeResponse(503, {}), api.requests.ConnectionError("x"), openrouter_response("gemma story")]
        response = self.post(STORY)
        self.assert_gemma_answer(response, "gemma story")
        url, body = post.call_args_list[2].args[0], post.call_args_list[2].kwargs["json"]
        self.assertIn("openrouter.ai", url)
        self.assertEqual(body["model"], "google/gemma-4-31b-it")
        self.assertEqual(body["provider"], {"only": ["modelrun"], "order": ["modelrun"],
                                            "allow_fallbacks": False, "zdr": True, "data_collection": "deny"})
        self.assertEqual(body["messages"], [
            {"role": "system", "content": "You are an expert story writer."},
            {"role": "user", "content": "write a story"}])

    @patch.object(api.requests, "post")
    def test_gemini_answers_and_blocks_never_reach_the_backup(self, post):
        for gemini in (FakeResponse(200, GEMINI_OK), FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}}),
                       FakeResponse(400, {"error": {"code": 400}})):
            post.reset_mock()
            post.side_effect = [gemini]
            response = self.post(STORY)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(response.get_json(), gemini.json())

    @patch.object(api.requests, "post")
    def test_json_requests_skip_json_mode_and_need_an_object(self, post):
        post.side_effect = [FakeResponse(503, {}), FakeResponse(503, {}), openrouter_response('```json\n{"intro": "hi"}\n```')]
        response = self.post(LESSON)
        self.assert_gemma_answer(response, '```json\n{"intro": "hi"}\n```')
        self.assertNotIn("response_format", post.call_args_list[2].kwargs["json"])
        post.side_effect = [FakeResponse(503, {}), FakeResponse(503, {}), openrouter_response("Sorry, no JSON")]
        self.assertEqual(self.post(LESSON).status_code, 502)

    @patch.object(api.requests, "post")
    def test_cut_off_or_failed_backup_is_an_honest_502(self, post):
        for backup in (openrouter_response("half a sto", finish_reason="length"), FakeResponse(429, {}),
                       api.requests.Timeout("x"), openrouter_response("   ")):
            post.reset_mock()
            post.side_effect = [FakeResponse(503, {}), FakeResponse(503, {}), backup]
            response = self.post(STORY)
            self.assertEqual((response.status_code, response.get_json()), (502, {"error": "Upstream error"}))

    @patch.object(api.requests, "post")
    def test_translations_go_to_gemma_first_without_the_marker(self, post):
        post.side_effect = [openrouter_response("traducción")]
        self.assert_gemma_answer(self.post(TRANSLATION), "traducción")
        self.assertEqual(post.call_count, 1)

    @patch.object(api.requests, "post")
    def test_translation_falls_to_gemini_without_the_marker(self, post):
        post.side_effect = [FakeResponse(503, {}), FakeResponse(200, GEMINI_OK)]
        response = self.post(TRANSLATION)
        self.assertEqual(response.get_json(), GEMINI_OK)
        gemini = post.call_args_list[1]
        self.assertIn("gemini-3.5-flash-lite", gemini.args[0])
        self.assertNotIn("purpose", gemini.kwargs["json"])

    @patch.object(api.requests, "post")
    def test_translation_without_openrouter_key_uses_gemini(self, post):
        post.side_effect = [FakeResponse(200, GEMINI_OK)]
        response = self.post(TRANSLATION, env={"GEMINI_API_KEY": "gemini-test"})
        self.assertEqual(response.get_json(), GEMINI_OK)
        self.assertIn("generativelanguage", post.call_args.args[0])

    @patch.object(api.requests, "post")
    def test_translation_shape_is_strict(self, post):
        self.assertEqual(self.post(dict(TRANSLATION, generationConfig={})).status_code, 400)
        self.assertEqual(self.post(dict(STORY, purpose="surprise")).status_code, 400)
        # Nested shape too (Sol): this route sends text somewhere other than Gemini first.
        for bad in ({"contents": True}, {"contents": []}, {"contents": [{"parts": "x"}]},
                    {"contents": [{"parts": [{"text": 5}]}]},
                    {"contents": STORY["contents"], "systemInstruction": "malformed"}):
            self.assertEqual(self.post(dict(bad, purpose="translation")).status_code, 400, bad)
        post.assert_not_called()

    @patch.object(api.requests, "post")
    def test_a_backup_content_block_is_final_never_retried_on_gemini(self, post):
        policy = {"code": 403, "metadata": {"error_type": "content_policy_violation"}}
        for blocked in (FakeResponse(403, {"error": {"code": 403}}),
                        openrouter_response("", finish_reason="content_filter"),
                        FakeResponse(200, {"error": policy}),
                        FakeResponse(200, {"error": {"code": 400, "metadata": {"error_type": "moderation"}}}),
                        FakeResponse(200, {"choices": [{"error": policy, "finish_reason": "error"}]})):
            post.reset_mock()
            post.side_effect = [blocked]
            response = self.post(TRANSLATION)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(response.get_json(), api.BACKUP_BLOCKED)
            self.assertNotIn("candidates", response.get_json())

    @patch.object(api.requests, "post")
    def test_every_gemini_5xx_moves_to_the_next_model(self, post):
        for status in (501, 505, 599):
            post.reset_mock()
            post.side_effect = [FakeResponse(status, {}), FakeResponse(200, GEMINI_OK)]
            self.assertEqual(self.post(STORY).get_json(), GEMINI_OK)
            self.assertIn("gemini-3.1-flash-lite", post.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
