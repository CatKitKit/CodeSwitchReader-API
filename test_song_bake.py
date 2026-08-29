import base64
import json
import os
import unittest
from unittest.mock import patch

import app as api


class StubResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload if payload is not None else {}
        self.status_code = status_code

    def json(self):
        return self.payload


class SongBakeTest(unittest.TestCase):
    def setUp(self):
        self.old_app_key = api.APP_KEY
        self.old_enabled = api.SONG_GENERATION_ENABLED
        api.APP_KEY = "test-app-key"
        api.SONG_GENERATION_ENABLED = True
        with api._song_rate_lock:
            api._song_ip_hits.clear()
            api._song_daily.update({"day": None, "count": 0})
        self.client = api.app.test_client()
        self.headers = {"X-App-Key": "test-app-key"}
        self.payload = {
            "lyrics": "[Verse 1]\nEsta es una canción.\n\n[Chorus]\nVuelve a cantar.",
            "targetLanguage": "Spanish",
            "style": "acoustic",
            "mood": "playful",
        }
        self.audio = b"ID3\x03\x00\x00test-song"
        self.upstream = {
            "steps": [{
                "type": "model_output",
                "content": [{
                    "type": "audio",
                    "mime_type": "audio/mpeg",
                    "data": base64.b64encode(self.audio).decode("ascii"),
                }],
            }],
        }

    def tearDown(self):
        api.APP_KEY = self.old_app_key
        api.SONG_GENERATION_ENABLED = self.old_enabled

    def post(self, payload=None, headers=None):
        return self.client.post(
            "/song-bake",
            json=self.payload if payload is None else payload,
            headers=self.headers if headers is None else headers,
        )

    def test_requires_app_key_and_explicit_server_switch(self):
        self.assertEqual(self.post(headers={}).status_code, 401)
        api.SONG_GENERATION_ENABLED = False
        self.assertEqual(self.post().status_code, 503)

    def test_accepts_only_the_fixed_song_shape_and_choices(self):
        self.assertEqual(
            self.post({**self.payload, "prompt": "arbitrary proxy"}).status_code,
            400,
        )
        self.assertEqual(
            self.post({**self.payload, "style": "sound exactly like an artist"}).status_code,
            400,
        )
        self.assertEqual(
            self.post({**self.payload, "lyrics": ""}).status_code,
            400,
        )

    def test_rejects_oversized_utf8_before_parsing(self):
        response = self.client.post(
            "/song-bake",
            data=json.dumps(
                {**self.payload, "lyrics": "猫" * 20000},
                ensure_ascii=False,
            ).encode("utf-8"),
            content_type="application/json",
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 413)

    def test_missing_provider_key_fails_before_spending_rate_limit(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            self.assertEqual(self.post().status_code, 503)
        self.assertEqual(api._song_daily["count"], 0)

    def test_success_uses_lyria_and_returns_only_the_audio_contract(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "provider-key"}), patch.object(
            api.requests, "post", return_value=StubResponse(self.upstream)
        ) as send:
            response = self.post()

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(base64.b64decode(body["audioBase64"]), self.audio)
        self.assertEqual(body["mimeType"], "audio/mpeg")
        self.assertEqual(body["bytes"], len(self.audio))
        self.assertEqual(body["model"], "lyria-3-pro-preview")
        args, kwargs = send.call_args
        self.assertEqual(args[0], api.SONG_INTERACTIONS_URL)
        self.assertEqual(kwargs["json"]["model"], "lyria-3-pro-preview")
        prompt = kwargs["json"]["input"]
        self.assertIn(self.payload["lyrics"], prompt)
        self.assertIn("Spanish", prompt)
        self.assertIn("playful acoustic", prompt)
        self.assertNotIn("audioBase64", prompt)

    def test_paid_rate_limit_is_separate_and_tight(self):
        with patch.object(api, "SONG_RATE_MAX_PER_WINDOW", 1), patch.dict(
            os.environ, {"GEMINI_API_KEY": "provider-key"}
        ), patch.object(api.requests, "post", return_value=StubResponse(self.upstream)):
            self.assertEqual(self.post().status_code, 200)
            self.assertEqual(self.post().status_code, 429)

    def test_global_daily_cap_survives_ip_rotation(self):
        with patch.object(api, "SONG_DAILY_MAX", 1), patch.dict(
            os.environ, {"GEMINI_API_KEY": "provider-key"}
        ), patch.object(api.requests, "post", return_value=StubResponse(self.upstream)):
            first = self.client.post(
                "/song-bake",
                json=self.payload,
                headers={**self.headers, "X-Forwarded-For": "198.51.100.1"},
            )
            second = self.client.post(
                "/song-bake",
                json=self.payload,
                headers={**self.headers, "X-Forwarded-For": "198.51.100.2"},
            )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)

    def test_rejects_missing_invalid_and_oversized_audio(self):
        bad_payloads = [
            {"steps": []},
            {"steps": [{"content": [{"type": "audio", "data": "%%%"}]}]},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), patch.dict(
                os.environ, {"GEMINI_API_KEY": "provider-key"}
            ), patch.object(api.requests, "post", return_value=StubResponse(payload)):
                with api._song_rate_lock:
                    api._song_ip_hits.clear()
                    api._song_daily.update({"day": None, "count": 0})
                self.assertEqual(self.post().status_code, 502)

        large = base64.b64encode(b"x" * 20).decode("ascii")
        payload = {"steps": [{"content": [{
            "type": "audio", "mime_type": "audio/mpeg", "data": large,
        }]}]}
        with patch.object(api, "SONG_MAX_AUDIO_BYTES", 10), patch.dict(
            os.environ, {"GEMINI_API_KEY": "provider-key"}
        ), patch.object(api.requests, "post", return_value=StubResponse(payload)):
            with api._song_rate_lock:
                api._song_ip_hits.clear()
                api._song_daily.update({"day": None, "count": 0})
            self.assertEqual(self.post().status_code, 502)

    def test_upstream_refusal_and_timeout_are_plain_failures(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "provider-key"}), patch.object(
            api.requests, "post", return_value=StubResponse(status_code=400)
        ):
            self.assertEqual(self.post().status_code, 502)

        with api._song_rate_lock:
            api._song_ip_hits.clear()
            api._song_daily.update({"day": None, "count": 0})
        with patch.dict(os.environ, {"GEMINI_API_KEY": "provider-key"}), patch.object(
            api.requests, "post", side_effect=api.requests.Timeout()
        ):
            self.assertEqual(self.post().status_code, 504)


if __name__ == "__main__":
    unittest.main()
