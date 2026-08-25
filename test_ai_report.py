import json
import unittest
from unittest.mock import patch

import app as api


class AiOutputReportTest(unittest.TestCase):
    def setUp(self):
        self.old_app_key = api.APP_KEY
        api.APP_KEY = "test-app-key"
        with api._report_rate_lock:
            api._report_ip_hits.clear()
        self.client = api.app.test_client()
        self.headers = {"X-App-Key": "test-app-key"}
        self.payload = {
            "category": "incorrect",
            "feature": "Context explanation",
            "output": "The reported answer",
            "note": "The example contradicts the translation.",
            "context": "Word: 猫\nSentence: 猫が寝ている。",
            "appVersion": "1.0",
        }

    def tearDown(self):
        api.APP_KEY = self.old_app_key

    def post(self, payload=None, headers=None):
        return self.client.post(
            "/report-ai-output",
            json=self.payload if payload is None else payload,
            headers=self.headers if headers is None else headers,
        )

    def test_requires_app_key(self):
        self.assertEqual(self.post(headers={}).status_code, 401)

    def test_accepts_report_and_writes_one_structured_record(self):
        with patch("builtins.print") as write_log:
            response = self.post()

        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["reportId"])
        write_log.assert_called_once()
        record = json.loads(write_log.call_args.args[0])
        self.assertEqual(record["eventType"], "ai_output_report")
        self.assertEqual(record["reportId"], body["reportId"])
        self.assertEqual(record["output"], self.payload["output"])
        self.assertEqual(record["context"], self.payload["context"])
        self.assertNotIn("ip", record)
        self.assertNotIn("account", record)
        self.assertNotIn("deviceId", record)

    def test_context_and_note_may_be_empty(self):
        payload = {**self.payload, "context": "", "note": ""}
        with patch("builtins.print"):
            response = self.post(payload)
        self.assertEqual(response.status_code, 201)

    def test_rejects_unknown_category_and_extra_fields(self):
        self.assertEqual(
            self.post({**self.payload, "category": "spam"}).status_code,
            400,
        )
        with api._report_rate_lock:
            api._report_ip_hits.clear()
        self.assertEqual(
            self.post({**self.payload, "deviceId": "should-not-exist"}).status_code,
            400,
        )

    def test_rejects_missing_or_oversized_output(self):
        self.assertEqual(self.post({**self.payload, "output": ""}).status_code, 400)
        with api._report_rate_lock:
            api._report_ip_hits.clear()
        self.assertEqual(
            self.post({**self.payload, "output": "x" * 12001}).status_code,
            400,
        )

    def test_character_caps_do_not_reject_valid_cjk_utf8(self):
        payload = {
            **self.payload,
            "output": "猫" * 12000,
            "context": "犬" * 6000,
        }
        with patch("builtins.print"):
            response = self.client.post(
                "/report-ai-output",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                content_type="application/json",
                headers=self.headers,
            )
        self.assertEqual(response.status_code, 201)

    def test_report_rate_limit_is_separate(self):
        for _ in range(api.REPORT_RATE_MAX_PER_WINDOW):
            with patch("builtins.print"):
                self.assertEqual(self.post().status_code, 201)
        self.assertEqual(self.post().status_code, 429)


if __name__ == "__main__":
    unittest.main()
