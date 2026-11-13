"""Web accounts (mobile SIGNIN_PLAN_TECH.md): Google sign-in -> our session -> the AI routes.

Google's check is faked for the route tests and run for real (the real library, our own
signing key) in GoogleTokenTest. RevenueCat is faked as in test_song_allowance. A route
"lets a caller in" when an empty body gets past the door and fails on its shape (400), so
no AI provider is ever called here.
"""
import base64
import datetime
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

import app as api

SECRET = "test-account-secret-0123456789abcdef"
GOOGLE_SUB = "110169484474386276334"
# Pinned on purpose: this id is where RevenueCat keeps a signed-in person's Plus. If this
# line has to change, every signed-in purchase in production is stranded.
ACCOUNT = "g_c5fd3ba40dbc8b1c65ddf25de31011a6e2d5be9da6b93943a0e47b06f125314d"
WEB_ROUTES = ["/ai-proxy", "/dict-fallback", "/report-ai-output"]


class StubResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload if payload is not None else {}
        self.status_code = status_code

    def json(self):
        return self.payload


class AccountTestCase(unittest.TestCase):
    SAVED = ("APP_KEY", "APP_KEY_NEXT", "RETIRED_APP_KEY", "WEB_ACCOUNTS_ENABLED",
             "SESSION_KEY_VERSION", "ACCOUNT_RATE_PER_MINUTE", "ACCOUNT_AI_DAILY",
             "ACCOUNT_DICT_DAILY", "ACCOUNT_SIGNIN_MAX_PER_WINDOW", "RATE_MAX_PER_WINDOW",
             "SONG_GENERATION_ENABLED", "GOOGLE_WEB_CLIENT_ID")

    def setUp(self):
        self.saved = {name: getattr(api, name) for name in self.SAVED}
        api.APP_KEY, api.APP_KEY_NEXT, api.RETIRED_APP_KEY = "app-key", "", "old-key"
        api.WEB_ACCOUNTS_ENABLED = True
        api.RATE_MAX_PER_WINDOW = 1000  # the IP limit has its own tests
        with api._account_lock:
            api._account_plus_cache.clear()
            api._account_minute_hits.clear()
            api._account_daily.clear()
            api._signin_ip_hits.clear()
        with api._rate_lock:
            api._ip_hits.clear()
            api._daily.update({"day": None, "count": 0})
        with api._song_rate_lock:
            api._song_lookup_global.clear()
        self.env = patch.dict("os.environ", {
            "ACCOUNT_ID_SECRET": SECRET, "REVENUECAT_SECRET_KEY": "sk_test"})
        self.env.start()
        self.plus_accounts = set()   # who RevenueCat says has Plus
        self.rc_down = False
        self.rc_calls = []
        self.google_calls = []
        self.google_down = False
        self.deletion_record = Mock()
        self.deletion_record.get.return_value.to_dict.return_value = None
        self.patches = [
            patch.object(api.requests, "get", self.fake_revenuecat),
            patch.object(api, "_google_account_sub", self.fake_google),
            patch.object(api, "_account_deletion_record", return_value=self.deletion_record),
        ]
        for item in self.patches:
            item.start()
        self.client = api.app.test_client()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.env.stop()
        for name, value in self.saved.items():
            setattr(api, name, value)

    def fake_revenuecat(self, url, headers=None, params=None, timeout=None):
        self.rc_calls.append(url)
        if self.rc_down:
            raise api.requests.ConnectionError("down")
        customer = url.split("/customers/", 1)[1].split("/", 1)[0]
        if customer not in self.plus_accounts:
            return StubResponse({}, 404)  # RevenueCat has never seen a free web account
        return StubResponse({"items": [{"id": "sub_1", "status": "active", "gives_access": True,
                                        "environment": "production"}]})

    def fake_google(self, token):
        """"good:<sub>" is a token Google vouches for; anything else is not."""
        self.google_calls.append(token)
        if self.google_down:
            raise api.AccountSignInUnavailable("down")
        return token[5:] if token.startswith("good:") else None

    def sign_in(self, token="good:" + GOOGLE_SUB, headers=None, body=None):
        return self.client.post("/account/session", headers=headers or {},
                                json=body if body is not None else {"idToken": token})

    def session(self, sub=GOOGLE_SUB):
        return self.sign_in("good:" + sub).get_json()["session"]

    def call(self, route, session=None, key=None, ip=None):
        headers = {}
        if session is not None:
            headers["X-Account-Session"] = session
        if key is not None:
            headers["X-App-Key"] = key
        if ip:
            headers["X-Forwarded-For"] = ip
        return self.client.post(route, json={}, headers=headers)

    def forge(self, body, secret=SECRET):
        """A session carrying `body`, signed the way the server signs."""
        signed = "v1." + api._b64url(json.dumps(body).encode("utf-8"))
        return signed + "." + api._session_signature(signed, secret.encode("utf-8"))


class SwitchedOffTest(AccountTestCase):
    def setUp(self):
        super().setUp()
        self.good_session = api._issue_session(ACCOUNT, SECRET.encode("utf-8"))
        api.WEB_ACCOUNTS_ENABLED = False

    def test_no_sign_in_and_google_is_never_asked(self):
        self.assertEqual(self.sign_in().status_code, 503)
        self.assertEqual(self.client.get(
            "/account", headers={"X-Account-Session": self.good_session}).status_code, 503)
        self.assertEqual(self.google_calls, [])
        self.assertEqual(self.rc_calls, [])

    def test_a_real_session_opens_nothing(self):
        for route in WEB_ROUTES + ["/song-bake"]:
            response = self.call(route, session=self.good_session)
            self.assertEqual(response.status_code, 401, route)
            self.assertEqual(response.get_json(), {"error": "Unauthorized"}, route)
        self.assertEqual(self.rc_calls, [])

    def test_the_app_key_works_as_before(self):
        for route in WEB_ROUTES:
            self.assertEqual(self.call(route, key="app-key").status_code, 400, route)
            self.assertEqual(self.call(route, key="old-key").status_code, 426, route)


class SignInTest(AccountTestCase):
    def test_a_good_token_gets_a_session_and_the_free_tier(self):
        response = self.sign_in()
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(set(body), {"session", "plus", "rcId"})
        self.assertIs(body["plus"], False)
        self.assertEqual(body["rcId"], ACCOUNT)
        self.assertEqual(self.call("/dict-fallback", session=body["session"]).status_code, 400)

    def test_the_account_id_recipe_never_changes(self):
        body = self.sign_in(headers={"X-App-Key": "app-key"}).get_json()
        self.assertEqual(body["rcId"], ACCOUNT)
        self.assertTrue(api.SONG_RC_USER_RE.match(ACCOUNT), "songs must accept the id")
        again = self.sign_in(headers={"X-App-Key": "app-key"}).get_json()
        self.assertEqual(again["rcId"], ACCOUNT, "one Google account, one id, every time")
        other = self.sign_in("good:999", headers={"X-App-Key": "app-key"}).get_json()
        self.assertNotEqual(other["rcId"], ACCOUNT)
        with patch.dict("os.environ", {"ACCOUNT_ID_SECRET": "another-secret-0123456789abcdefgh"}):
            moved = self.sign_in(headers={"X-App-Key": "app-key"}).get_json()
        self.assertNotEqual(moved["rcId"], ACCOUNT, "the id depends on the secret")

    def test_plus_is_reported_from_revenuecat(self):
        self.plus_accounts.add(ACCOUNT)
        self.assertIs(self.sign_in().get_json()["plus"], True)
        self.assertIn(f"/customers/{ACCOUNT}/subscriptions", self.rc_calls[0])

    def test_revenuecat_down_never_fails_a_sign_in(self):
        self.rc_down = True
        response = self.sign_in()
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.get_json()["plus"])
        self.assertTrue(response.get_json()["session"])

    def test_a_token_google_does_not_vouch_for_is_refused(self):
        response = self.sign_in("forged")
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("session", response.get_json())
        self.assertEqual(self.rc_calls, [])

    def test_google_not_answering_is_a_try_again_not_a_no(self):
        self.google_down = True
        response = self.sign_in()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("session", response.get_json())

    def test_bad_shapes_never_reach_google(self):
        api.ACCOUNT_SIGNIN_MAX_PER_WINDOW = 100  # the tries limit has its own test
        bodies = [{}, {"idToken": ""}, {"idToken": 5}, {"idToken": ["x"]}, {"idToken": None},
                  {"idToken": "good:1", "extra": 1}, {"token": "good:1"},
                  {"idToken": "x" * (api.ACCOUNT_ID_TOKEN_MAX_CHARS + 1)}, ["good:1"], "good:1"]
        for body in bodies:
            self.assertEqual(self.sign_in(body=body).status_code, 400, body)
        big = self.client.post("/account/session", data=b"x" * (api.ACCOUNT_MAX_BODY_BYTES + 1),
                               content_type="application/json")
        self.assertEqual(big.status_code, 413)
        self.assertEqual(self.google_calls, [])

    def test_no_secret_or_a_short_one_fails_closed(self):
        session = self.session()
        for secret in ("", "short", "x" * (api.ACCOUNT_SECRET_MIN_CHARS - 1)):
            with patch.dict("os.environ", {"ACCOUNT_ID_SECRET": secret}):
                self.google_calls.clear()
                self.assertEqual(self.sign_in().status_code, 503, secret)
                self.assertEqual(self.google_calls, [], secret)
                self.assertEqual(self.call("/dict-fallback", session=session).status_code, 503)
                self.assertEqual(self.client.get(
                    "/account", headers={"X-Account-Session": session}).status_code, 503)

    def test_no_client_id_fails_closed(self):
        api.GOOGLE_WEB_CLIENT_ID = ""  # an empty audience would mean "any site's token"
        self.assertEqual(self.sign_in().status_code, 503)
        self.assertEqual(self.google_calls, [])

    def test_sign_in_tries_are_limited_per_address(self):
        api.ACCOUNT_SIGNIN_MAX_PER_WINDOW = 3
        ip = {"X-Forwarded-For": "203.0.113.5"}
        self.assertEqual([self.sign_in("forged", headers=ip).status_code for _ in range(4)],
                         [401, 401, 401, 429])
        self.assertEqual(len(self.google_calls), 3, "a refused try never reaches Google")
        self.assertEqual(self.sign_in(headers={"X-Forwarded-For": "203.0.113.6"}).status_code, 200)


class AccountDeletionTest(AccountTestCase):
    MISSING_CUSTOMER = StubResponse({"type": "resource_missing"}, status_code=404)

    def delete(self, payload=None):
        if payload is None:
            payload = {"idToken": "good:" + GOOGLE_SUB}
        return self.client.delete("/account", json=payload)

    @patch.dict("os.environ", {"REVENUECAT_DELETE_KEY": "sk_delete"})
    def test_requires_google_proof_for_the_account_being_deleted(self):
        with patch.object(api, "_revoke_account_sessions") as revoke, \
                patch.object(api.requests, "delete") as delete:
            self.assertEqual(self.delete({"idToken": "forged"}).status_code, 401)
            self.assertEqual(self.delete({"idToken": "good:999", "expectedAccount": ACCOUNT}).status_code, 409)
            self.assertEqual(self.delete({"idToken": "good:999", "rcId": ACCOUNT}).status_code, 400)
            revoke.assert_not_called()
            delete.assert_not_called()

    @patch.dict("os.environ", {"REVENUECAT_DELETE_KEY": "sk_delete"})
    def test_deletion_revokes_existing_sessions_and_uses_only_the_verified_id(self):
        old, other = self.session(), self.session("999")
        cutoff = datetime.datetime.now(datetime.timezone.utc)

        def revoke(account, secret):
            self.assertEqual(account, ACCOUNT)
            self.deletion_record.get.return_value.to_dict.return_value = {"revokedBefore": cutoff}

        self.plus_accounts.add(ACCOUNT)
        with patch.object(api, "_revoke_account_sessions", side_effect=revoke), \
                patch.object(api.requests, "delete", return_value=StubResponse(status_code=200)) as delete, \
                patch.object(api.requests, "get", return_value=self.MISSING_CUSTOMER) as get:
            response = self.delete({"idToken": "good:" + GOOGLE_SUB, "expectedAccount": ACCOUNT})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {"status": "deleted", "rcId": ACCOUNT})
            url = f"{api.REVENUECAT_API}/projects/{api.REVENUECAT_PROJECT_ID}/customers/{ACCOUNT}"
            delete.assert_called_once_with(url, headers={"Authorization": "Bearer sk_delete"},
                                           timeout=(5, 10), allow_redirects=False)
            get.assert_called_once_with(url, headers={"Authorization": "Bearer sk_delete"},
                                        timeout=(5, 10), allow_redirects=False)
        self.assertEqual(self.call("/dict-fallback", session=old).status_code, 401)
        self.assertEqual(self.client.get("/account", headers={api.SESSION_HEADER: old}).status_code, 401)
        with patch.object(api, "_account_deletion_record") as record:
            record.return_value.get.return_value.to_dict.return_value = None
            self.assertEqual(self.call("/dict-fallback", session=other).status_code, 400)
        fresh = api._issue_session(ACCOUNT, api._account_secret(), cutoff.timestamp() + 2)
        self.assertEqual(self.call("/dict-fallback", session=fresh).status_code, 400)

    @patch.dict("os.environ", {"REVENUECAT_DELETE_KEY": "sk_delete"})
    def test_distinguishes_queued_confirmed_and_failed_deletion(self):
        with patch.object(api, "_revoke_account_sessions"), \
                patch.object(api.requests, "delete") as delete, patch.object(api.requests, "get") as get:
            delete.return_value = StubResponse(status_code=200)
            get.return_value = StubResponse(status_code=200)
            self.assertEqual(self.delete().status_code, 202)
            get.side_effect = api.requests.ConnectionError("unavailable")
            self.assertEqual(self.delete().status_code, 202)
            delete.return_value = self.MISSING_CUSTOMER
            self.assertEqual(self.delete().get_json()["status"], "deleted")
            delete.return_value = StubResponse(status_code=404)
            self.assertEqual(self.delete().status_code, 503)
            delete.return_value = StubResponse(status_code=403)
            self.assertEqual(self.delete().status_code, 503)

    @patch.dict("os.environ", {"REVENUECAT_DELETE_KEY": "sk_delete"})
    def test_store_failure_never_sends_the_delete_to_revenuecat(self):
        with patch.object(api, "_revoke_account_sessions", side_effect=RuntimeError("store down")), \
                patch.object(api.requests, "delete") as delete:
            self.assertEqual(self.delete().status_code, 503)
            delete.assert_not_called()

    def check_link(self, proof):
        return self.client.get("/account/phone-link", headers={api.SESSION_HEADER: proof})

    def test_phone_link_can_check_for_deletion_only(self):
        response = self.sign_in(body={"idToken": "good:" + GOOGLE_SUB, "phoneLink": True})
        proof = response.get_json()["phoneLink"]
        self.assertEqual(self.check_link(proof).get_json(), {"revoked": False})
        self.assertEqual(self.check_link(response.get_json()["session"]).status_code, 401)
        self.assertEqual(self.call("/dict-fallback", session=proof).status_code, 401)
        self.deletion_record.get.return_value.to_dict.return_value = {
            "revokedBefore": datetime.datetime.now(datetime.timezone.utc)}
        self.assertEqual(self.check_link(proof).get_json(), {"revoked": True})

    @patch.dict("os.environ", {"REVENUECAT_DELETE_KEY": "sk_delete"})
    def test_an_older_phone_link_uses_a_lookup_that_cannot_create_a_customer(self):
        first_issued = time.time() - 32 * 24 * 60 * 60
        proof = api._issue_session(ACCOUNT, api._account_secret(), first_issued, version="p1")
        with patch.object(api.requests, "get") as get:
            get.return_value = self.MISSING_CUSTOMER
            self.assertEqual(self.check_link(proof).get_json(), {"revoked": True})
            self.assertTrue(get.call_args.args[0].endswith(f"/customers/{ACCOUNT}"))
            get.return_value = StubResponse({"first_seen_at": int(time.time() * 1000)})
            self.assertEqual(self.check_link(proof).get_json(), {"revoked": True})
            get.return_value = StubResponse({"first_seen_at": int((first_issued + 5) * 1000)})
            renewed = self.check_link(proof).get_json()
            self.assertFalse(renewed["revoked"])
            self.assertNotEqual(renewed["phoneLink"], proof)
            self.assertEqual(self.call("/dict-fallback", session=renewed["phoneLink"]).status_code, 401)


class SessionTest(AccountTestCase):
    def assertSignInAgain(self, session, why):
        for route in WEB_ROUTES:
            response = self.call(route, session=session)
            self.assertEqual(response.status_code, 401, (why, route))
            self.assertTrue(response.get_json().get("signIn"), (why, route))
        self.assertEqual(self.client.get(
            "/account", headers={"X-Account-Session": session}).status_code, 401, why)

    def test_a_session_is_what_the_server_signed_and_nothing_else(self):
        good = self.session()
        self.assertEqual(self.call("/dict-fallback", session=good).status_code, 400)
        version, body, signature = good.split(".")
        other = self.session("999").split(".")
        self.assertSignInAgain(f"{version}.{other[1]}.{signature}", "another account's body")
        self.assertSignInAgain(f"{version}.{body}.{other[2]}", "another account's signature")
        self.assertSignInAgain(f"{version}.{body}.", "no signature")
        self.assertSignInAgain(f"v2.{body}.{signature}", "unknown version")
        self.assertSignInAgain(f"{version}.{body}.{signature}.x", "extra part")
        self.assertSignInAgain(body, "body alone")
        self.assertSignInAgain("x" * (api.SESSION_MAX_CHARS + 1), "too long")
        self.assertSignInAgain("v1.\xe9\xfc.\xe9", "junk bytes are a no, not a crash")

    def test_a_session_signed_with_another_secret_is_refused(self):
        self.assertSignInAgain(self.forge(
            {"rc": ACCOUNT, "exp": int(time.time()) + 600}, "someone-elses-secret-0123456789abcdef"),
            "wrong secret")

    def test_a_session_ends(self):
        self.assertSignInAgain(self.forge({"rc": ACCOUNT, "exp": int(time.time()) - 1}), "expired")
        fresh = self.forge({"rc": ACCOUNT, "exp": int(time.time()) + 600})
        self.assertEqual(self.call("/dict-fallback", session=fresh).status_code, 400)
        issued = json.loads(base64.urlsafe_b64decode(self.session().split(".")[1] + "=="))
        self.assertAlmostEqual(issued["exp"], time.time() + 30 * 24 * 3600, delta=60)

    def test_even_a_signed_body_must_be_a_sane_one(self):
        later = int(time.time()) + 600
        for body in ({"rc": ACCOUNT}, {"exp": later}, {"rc": "$RCAnonymousID:abc", "exp": later},
                     {"rc": ACCOUNT, "exp": str(later)}, {"rc": ACCOUNT, "exp": True},
                     {"rc": ACCOUNT, "exp": later + 0.5}, {"rc": [ACCOUNT], "exp": later},
                     [ACCOUNT, later]):
            self.assertSignInAgain(self.forge(body), body)

    def test_raising_the_key_version_signs_everyone_out(self):
        session = self.session()
        api.SESSION_KEY_VERSION = "2"
        self.assertSignInAgain(session, "old key version")
        self.assertEqual(self.call("/dict-fallback", session=self.session()).status_code, 400)
        self.assertEqual(self.sign_in(headers={"X-App-Key": "app-key"}).get_json()["rcId"], ACCOUNT,
                         "accounts are untouched")


class TierTest(AccountTestCase):
    def test_a_free_account_gets_the_dictionary_and_reports_only(self):
        session = self.session()
        self.assertEqual(self.call("/dict-fallback", session=session).status_code, 400)
        self.assertEqual(self.call("/report-ai-output", session=session).status_code, 400)
        refused = self.call("/ai-proxy", session=session)
        self.assertEqual(refused.status_code, 403)
        self.assertTrue(refused.get_json()["plusRequired"])

    def test_a_plus_account_gets_the_ai_too(self):
        self.plus_accounts.add(ACCOUNT)
        session = self.session()
        for route in WEB_ROUTES:
            self.assertEqual(self.call(route, session=session).status_code, 400, route)

    def test_songs_stay_app_only(self):
        api.SONG_GENERATION_ENABLED = True
        self.plus_accounts.add(ACCOUNT)
        self.assertEqual(self.call("/song-bake", session=self.session()).status_code, 401)
        self.assertEqual(self.call("/song-bake", key="app-key").status_code, 400)

    def test_the_app_is_never_asked_about_plus(self):
        self.assertEqual(self.call("/ai-proxy", key="app-key").status_code, 400)
        self.assertEqual(self.call("/ai-proxy", key="app-key", session="junk").status_code, 400,
                         "a good key needs no session")
        self.assertEqual(self.rc_calls, [])

    def test_a_retired_key_is_told_to_update_whatever_else_it_sends(self):
        self.assertEqual(self.call("/ai-proxy", key="old-key", session=self.session()).status_code, 426)

    def test_no_credentials_is_the_same_plain_no(self):
        for route in WEB_ROUTES:
            response = self.call(route)
            self.assertEqual((response.status_code, response.get_json()),
                             (401, {"error": "Unauthorized"}), route)

    def test_revenuecat_down_is_a_try_again_never_a_free_pass(self):
        session = self.session()
        with api._account_lock:
            api._account_plus_cache.clear()
        self.rc_down = True
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 503)
        self.assertEqual(self.call("/dict-fallback", session=session).status_code, 400,
                         "the free step does not need RevenueCat")

    def test_a_recent_yes_carries_a_plus_account_through_an_outage(self):
        self.rc_down = True
        with api._account_lock:
            api._account_plus_cache[ACCOUNT] = (True, time.time() - api.ACCOUNT_PLUS_YES_SECS - 1)
        session = api._issue_session(ACCOUNT, SECRET.encode("utf-8"))
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 400)
        with api._account_lock:
            api._account_plus_cache[ACCOUNT] = (True, time.time() - api.ACCOUNT_PLUS_GRACE_SECS - 1)
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 503, "but not a stale one")
        with api._account_lock:
            api._account_plus_cache[ACCOUNT] = (False, time.time() - api.ACCOUNT_PLUS_NO_SECS - 1)
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 503, "and never a no")

    def test_the_plus_answer_is_reused_then_asked_again(self):
        session = api._issue_session(ACCOUNT, SECRET.encode("utf-8"))
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 403)
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 403)
        self.assertEqual(len(self.rc_calls), 1, "a no is reused for a minute")
        self.plus_accounts.add(ACCOUNT)  # they just bought Plus on the phone
        with api._account_lock:
            api._account_plus_cache[ACCOUNT] = (False, time.time() - api.ACCOUNT_PLUS_NO_SECS - 1)
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 400)
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 400)
        self.assertEqual(len(self.rc_calls), 2, "a yes is reused too")
        self.plus_accounts.clear()       # and later it ends
        with api._account_lock:
            api._account_plus_cache[ACCOUNT] = (True, time.time() - api.ACCOUNT_PLUS_YES_SECS - 1)
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 403)

    def test_requests_arriving_together_share_one_ask(self):
        self.plus_accounts.add(ACCOUNT)
        self.assertEqual(self.together(8, self.fake_revenuecat), [True] * 8)
        self.assertEqual(len(self.rc_calls), 1, "eight requests, one ask")

    def together(self, count, fake):
        """`count` requests for one account while RevenueCat (`fake`) is slow to answer."""
        started, release = threading.Event(), threading.Event()

        def slow(url, **kwargs):
            started.set()
            release.wait(5)
            return fake(url, **kwargs)

        results = []

        def ask():
            try:
                results.append(api._account_has_plus(ACCOUNT))
            except api.SongAllowanceUnavailable:
                results.append("unavailable")

        with patch.object(api.requests, "get", slow):
            threads = [threading.Thread(target=ask) for _ in range(count)]
            for thread in threads:
                thread.start()
            self.assertTrue(started.wait(5))
            time.sleep(0.2)  # let the others reach the wait
            release.set()
            for thread in threads:
                thread.join(5)
        self.assertEqual(api._account_lookups, {}, "nothing is left waiting")
        return results

    def test_a_slow_failure_is_shared_not_queued(self):
        self.rc_down = True
        self.assertEqual(self.together(8, self.fake_revenuecat), ["unavailable"] * 8)
        self.assertEqual(len(self.rc_calls), 1, "eight requests wait for ONE failing ask")
        with self.assertRaises(api.SongAllowanceUnavailable):
            api._account_has_plus(ACCOUNT)
        self.assertEqual(len(self.rc_calls), 2, "a request arriving afterwards asks for itself")
        with api._account_lock:
            api._account_plus_cache[ACCOUNT] = (True, time.time() - api.ACCOUNT_PLUS_YES_SECS - 1)
        self.assertEqual(self.together(8, self.fake_revenuecat), [True] * 8,
                         "and a recent yes carries all of them")
        self.assertEqual(len(self.rc_calls), 3)

    def test_a_failed_ask_leaves_nothing_behind(self):
        self.rc_down = True
        with self.assertRaises(api.SongAllowanceUnavailable):
            api._account_has_plus(ACCOUNT)
        self.assertEqual(api._account_lookups, {})
        self.assertEqual(api._account_plus_cache, {}, "a failure is never remembered as an answer")
        self.rc_down = False
        self.assertIs(api._account_has_plus(ACCOUNT), False)

    def test_check_again_asks_sooner_than_the_ai_routes(self):
        session = api._issue_session(ACCOUNT, SECRET.encode("utf-8"))
        headers = {"X-Account-Session": session}
        self.assertEqual(self.client.get("/account", headers=headers).get_json(), {"plus": False})
        self.plus_accounts.add(ACCOUNT)
        self.assertEqual(self.client.get("/account", headers=headers).get_json(), {"plus": False},
                         "seconds later the answer is reused")
        with api._account_lock:
            api._account_plus_cache[ACCOUNT] = (False, time.time() - api.ACCOUNT_PLUS_FRESH_SECS - 1)
        self.assertEqual(self.client.get("/account", headers=headers).get_json(), {"plus": True})
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 400,
                         "and the AI routes see it at once")
        self.rc_down = True
        with api._account_lock:
            api._account_plus_cache.clear()
        self.assertEqual(self.client.get("/account", headers=headers).get_json(), {"plus": None})


class AccountLimitTest(AccountTestCase):
    def test_an_account_is_limited_per_minute_from_any_address(self):
        api.ACCOUNT_RATE_PER_MINUTE = 3
        session = self.session()
        statuses = [self.call("/dict-fallback", session=session, ip=f"203.0.113.{n}").status_code
                    for n in range(4)]
        self.assertEqual(statuses, [400, 400, 400, 429])
        self.assertEqual(self.call("/report-ai-output", session=session).status_code, 429,
                         "one minute window for every route")
        self.assertEqual(self.call("/dict-fallback", session=self.session("999")).status_code, 400,
                         "another account is not affected")
        self.assertEqual(self.call("/dict-fallback", key="app-key").status_code, 400,
                         "nor is the app")

    def test_daily_counts_are_per_account_and_per_kind(self):
        api.ACCOUNT_DICT_DAILY, api.ACCOUNT_AI_DAILY = 2, 3
        self.plus_accounts.add(ACCOUNT)
        session = self.session()
        self.assertEqual([self.call("/dict-fallback", session=session).status_code for _ in range(3)],
                         [400, 400, 429])
        self.assertEqual([self.call("/ai-proxy", session=session).status_code for _ in range(4)],
                         [400, 400, 400, 429])
        self.assertEqual(self.call("/report-ai-output", session=session).status_code, 400,
                         "reports have no daily count")
        self.assertEqual(self.call("/dict-fallback", session=self.session("999")).status_code, 400)

    def test_a_new_day_starts_a_new_count(self):
        api.ACCOUNT_DICT_DAILY = 1
        session = self.session()
        self.assertEqual(self.call("/dict-fallback", session=session).status_code, 400)
        self.assertEqual(self.call("/dict-fallback", session=session).status_code, 429)
        with api._account_lock:
            api._account_daily[(ACCOUNT, "dict")][0] = "2000-01-01"
        self.assertEqual(self.call("/dict-fallback", session=session).status_code, 400)

    def test_a_refused_account_takes_nothing_from_the_shared_limits(self):
        api.ACCOUNT_DICT_DAILY = 1
        session = self.session()
        self.call("/dict-fallback", session=session)
        before = api._daily["count"]
        self.assertEqual(self.call("/dict-fallback", session=session).status_code, 429)
        self.assertEqual(self.call("/ai-proxy", session=session).status_code, 403)
        self.assertEqual(self.call("/ai-proxy", session="v1.x.y").status_code, 401)
        self.assertEqual(api._daily["count"], before)


CORS_PROBE = """
import json
import app
client = app.app.test_client()
seen = {}
for origin in json.loads(%r):
    for route in ["/ai-proxy", "/dict-fallback", "/report-ai-output", "/account/session",
                  "/account", "/generate-ipa", "/song-bake"]:
        headers = client.options(route, headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-account-session"}).headers
        seen[origin + " " + route] = [headers.get("Access-Control-Allow-Origin"),
                                      (headers.get("Access-Control-Allow-Headers") or "").lower()]
print("PROBE" + json.dumps(seen))
"""
APP_ORIGINS = ["https://localhost", "capacitor://localhost", "https://codeswitchreader.org"]
WEB_ORIGINS = ["https://app.meowti-lingual.com", "http://localhost:8455"]
STRANGERS = ["https://evil.example", "http://localhost:9999", "https://app.meowti-lingual.com.evil.example"]


class CorsTest(unittest.TestCase):
    """The allowed origins are fixed when the server starts, so each case starts its own."""

    def probe(self, switch):
        script = CORS_PROBE % json.dumps(APP_ORIGINS + WEB_ORIGINS + STRANGERS)
        done = subprocess.run(
            [sys.executable, "-c", script], cwd=os.path.dirname(os.path.abspath(__file__)),
            env={**os.environ, "WEB_ACCOUNTS_ENABLED": switch}, capture_output=True, text=True)
        line = [row for row in done.stdout.splitlines() if row.startswith("PROBE")]
        self.assertTrue(line, done.stderr[-500:])
        return json.loads(line[0][5:])

    def test_switched_on_the_web_origins_may_send_a_session(self):
        seen = self.probe("1")
        for key, (allowed, headers) in seen.items():
            origin = key.split(" ")[0]
            if origin in STRANGERS:
                self.assertIsNone(allowed, key)
            else:
                self.assertEqual(allowed, origin, key)
                self.assertIn("x-account-session", headers, key)

    def test_switched_off_only_the_apps_own_origins_are_allowed(self):
        seen = self.probe("")
        for key, (allowed, _) in seen.items():
            origin = key.split(" ")[0]
            self.assertEqual(allowed, origin if origin in APP_ORIGINS else None, key)

    def test_account_delete_accepts_json_preflight(self):
        response = api.app.test_client().options("/account", headers={
            "Origin": "https://localhost",
            "Access-Control-Request-Method": "DELETE",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertIn("DELETE", response.headers["Allow"])
        self.assertIn("DELETE", response.headers["Access-Control-Allow-Methods"])
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "https://localhost")
        self.assertIn("content-type", response.headers["Access-Control-Allow-Headers"].lower())


try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    from google.auth import crypt as google_crypt, jwt as google_jwt
    REAL_GOOGLE_LIBRARY = True
except ImportError:
    REAL_GOOGLE_LIBRARY = False


@unittest.skipUnless(REAL_GOOGLE_LIBRARY, "google-auth and cryptography are not installed here")
class GoogleTokenTest(unittest.TestCase):
    """_google_account_sub against the REAL google-auth library. Only Google's key fetch is
    replaced: the keys are ours, so we can sign tokens that are right and wrong in each way."""
    CLIENT_ID = "our-client.apps.googleusercontent.com"

    @classmethod
    def setUpClass(cls):
        cls.signer, cls.cert = cls.key_and_cert("kid-google")
        cls.stranger, _ = cls.key_and_cert("kid-google")  # same key id, a different key

    @staticmethod
    def key_and_cert(key_id):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
                .public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=1))
                .sign(key, hashes.SHA256()))
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                serialization.NoEncryption())
        return (google_crypt.RSASigner.from_string(pem, key_id),
                cert.public_bytes(serialization.Encoding.PEM).decode("ascii"))

    def setUp(self):
        self.saved = api.GOOGLE_WEB_CLIENT_ID
        api.GOOGLE_WEB_CLIENT_ID = self.CLIENT_ID
        self.certs_status = 200
        self.fetched = []
        self.patch = patch.object(api, "_google_certs_transport", lambda: self.fake_transport)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        api.GOOGLE_WEB_CLIENT_ID = self.saved

    def fake_transport(self, url, method="GET", **kwargs):
        self.fetched.append(url)

        class Response:
            status = self.certs_status
            data = json.dumps({"kid-google": self.cert}).encode("utf-8")
        return Response()

    def test_a_garbled_key_answer_is_unavailable_too(self):
        """Through the REAL transport wrapper, with only the network call replaced."""
        self.patch.stop()
        try:
            for data in (b"<html>oops</html>", b"", b"[]", b"{}", b"\xff\xfe"):
                class Response:
                    status = 200
                with patch("google.auth.transport.requests.Request.__call__",
                           lambda *args, **kwargs: Response()):
                    Response.data = data
                    with self.assertRaises(api.AccountSignInUnavailable, msg=data):
                        api._google_account_sub(self.token())

            class Good:
                status = 200
                data = json.dumps({"kid-google": self.cert}).encode("utf-8")
            seen = {}

            def fetch(self_, url, method="GET", **kwargs):
                seen.update(kwargs)
                return Good()
            with patch("google.auth.transport.requests.Request.__call__", fetch):
                self.assertEqual(api._google_account_sub(self.token()), GOOGLE_SUB)
            self.assertEqual(seen.get("timeout"), 10, "Google's own wait is two minutes")
        finally:
            self.patch.start()

    def token(self, signer=None, **changes):
        now = int(time.time())
        claims = {"iss": "https://accounts.google.com", "aud": self.CLIENT_ID, "sub": GOOGLE_SUB,
                  "email": "someone@example.com", "iat": now - 5, "exp": now + 3600}
        claims.update(changes)
        claims = {name: value for name, value in claims.items() if value is not None}
        return google_jwt.encode(signer or self.signer, claims).decode("ascii")

    def test_a_real_google_token_for_this_app_names_the_account(self):
        self.assertEqual(api._google_account_sub(self.token()), GOOGLE_SUB)
        self.assertEqual(self.fetched, ["https://www.googleapis.com/oauth2/v1/certs"])
        self.assertEqual(api._google_account_sub(self.token(iss="accounts.google.com")), GOOGLE_SUB)

    def test_a_token_made_for_another_site_is_refused(self):
        self.assertIsNone(api._google_account_sub(self.token(aud="someone-else.apps.googleusercontent.com")))
        self.assertIsNone(api._google_account_sub(self.token(aud=None)))

    def test_an_expired_or_not_yet_valid_token_is_refused(self):
        now = int(time.time())
        self.assertIsNone(api._google_account_sub(self.token(iat=now - 7200, exp=now - 3600)))
        self.assertIsNone(api._google_account_sub(self.token(iat=now + 3600, exp=now + 7200)))

    def test_a_token_google_did_not_issue_is_refused(self):
        self.assertIsNone(api._google_account_sub(self.token(iss="https://accounts.evil.example")))
        self.assertIsNone(api._google_account_sub(self.token(iss=None)))

    def test_a_token_signed_by_someone_else_is_refused(self):
        self.assertIsNone(api._google_account_sub(self.token(signer=self.stranger)))
        head, body, signature = self.token().split(".")
        swapped = self.token(sub="999").split(".")[1]
        self.assertIsNone(api._google_account_sub(f"{head}.{swapped}.{signature}"))
        unsigned = api._b64url(json.dumps({"alg": "none", "typ": "JWT", "kid": "kid-google"}).encode())
        self.assertIsNone(api._google_account_sub(f"{unsigned}.{body}."))
        for junk in ("", "not-a-token", "a.b.c", "é.ü.ñ"):
            self.assertIsNone(api._google_account_sub(junk), junk)

    def test_a_token_without_a_usable_account_id_is_refused(self):
        for sub in (None, "", 12345, "x" * 256):
            self.assertIsNone(api._google_account_sub(self.token(sub=sub)), sub)

    def test_googles_keys_not_answering_is_unavailable_not_a_no(self):
        self.certs_status = 500
        with self.assertRaises(api.AccountSignInUnavailable):
            api._google_account_sub(self.token())


if __name__ == "__main__":
    unittest.main()
