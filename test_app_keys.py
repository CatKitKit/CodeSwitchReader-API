"""Paid-update key swap: current + next keys pass, the retired key gets 426, others 401."""
import unittest

import app as api

ROUTES = ["/ai-proxy", "/song-bake", "/dict-fallback", "/report-ai-output"]


class AppKeyTests(unittest.TestCase):
    def setUp(self):
        self.saved = (api.APP_KEY, api.APP_KEY_NEXT, api.RETIRED_APP_KEY)
        api.APP_KEY, api.APP_KEY_NEXT, api.RETIRED_APP_KEY = "current-key", "next-key", "old-key"
        self.client = api.app.test_client()

    def tearDown(self):
        api.APP_KEY, api.APP_KEY_NEXT, api.RETIRED_APP_KEY = self.saved

    def status(self, route, key):
        headers = {} if key is None else {"X-App-Key": key}
        # An empty body fails AFTER auth (400/413/503), so any non-401/426 means "let in".
        return self.client.post(route, json={}, headers=headers).status_code

    def test_current_and_next_keys_pass_auth(self):
        for route in ROUTES:
            for key in ("current-key", "next-key"):
                self.assertNotIn(self.status(route, key), (401, 426), (route, key))

    def test_retired_key_is_told_to_update(self):
        for route in ROUTES:
            self.assertEqual(self.status(route, "old-key"), 426, route)

    def test_unknown_missing_and_junk_keys_are_unauthorized(self):
        for route in ROUTES:
            for key in ("wrong", None, "", "kéy-ü"):
                self.assertEqual(self.status(route, key), 401, (route, key))

    def test_unset_next_and_retired_never_match_an_empty_header(self):
        api.APP_KEY_NEXT = api.RETIRED_APP_KEY = ""
        self.assertEqual(self.status("/ai-proxy", ""), 401)
        self.assertNotIn(self.status("/ai-proxy", "current-key"), (401, 426))

    def test_unconfigured_server_still_fails_closed(self):
        api.APP_KEY = ""
        self.assertEqual(self.status("/ai-proxy", "next-key"), 503)


if __name__ == "__main__":
    unittest.main()
