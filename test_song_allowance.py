"""Plus song allowance on /song-bake (mobile PAYWALL_PLAN.md): RevenueCat says who has Plus,
Firestore counts each purchase's songs. Both are faked here; Lyria is faked as in
test_song_bake."""
import base64
import sys
import types
import unittest
from unittest.mock import patch

import app as api


class StubResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload if payload is not None else {}
        self.status_code = status_code

    def json(self):
        return self.payload


def fake_firestore_module(store):
    """Just enough of google.cloud.firestore: documents in `store`, transactions applied
    immediately (one thread here, so that is what a transaction would do)."""
    class Snap:
        def __init__(self, data):
            self._data = data

        def to_dict(self):
            return dict(self._data) if self._data is not None else None

    class Ref:
        def __init__(self, key):
            self.key = key

        def get(self, transaction=None):
            return Snap(store.get(self.key))

    class Collection:
        def __init__(self, name):
            self.name = name

        def document(self, doc_id):
            return Ref((self.name, doc_id))

    class Transaction:
        def set(self, ref, data):
            store[ref.key] = {k: v for k, v in data.items() if k != "updated"}

        def update(self, ref, data):
            store[ref.key].update(data)

    class Client:
        def collection(self, name):
            return Collection(name)

        def transaction(self):
            return Transaction()

    module = types.ModuleType("google.cloud.firestore")
    module.Client = Client
    module.SERVER_TIMESTAMP = object()
    module.transactional = lambda fn: (lambda transaction: fn(transaction))
    return module


class SongAllowanceTest(unittest.TestCase):
    def setUp(self):
        self.saved = {name: getattr(api, name) for name in (
            "APP_KEY", "SONG_GENERATION_ENABLED", "SONG_PLUS_REQUIRED",
            "SONG_RATE_MAX_PER_WINDOW", "SONG_LOOKUP_MAX_PER_WINDOW",
            "SONG_LOOKUP_GLOBAL_PER_MINUTE", "_firestore_client")}
        api.APP_KEY = "test-app-key"
        api.SONG_GENERATION_ENABLED = True
        api.SONG_PLUS_REQUIRED = True
        api.SONG_RATE_MAX_PER_WINDOW = 100  # the IP limit has its own tests
        api.SONG_LOOKUP_MAX_PER_WINDOW = 100
        api._firestore_client = None
        with api._song_rate_lock:
            api._song_ip_hits.clear()
            api._song_daily.update({"day": None, "count": 0})
            api._song_lookup_hits.clear()
            api._song_lookup_global.clear()
        self.store = {}
        firestore = fake_firestore_module(self.store)
        cloud = types.ModuleType("google.cloud")
        cloud.firestore = firestore
        google = types.ModuleType("google")
        google.cloud = cloud
        self.modules = patch.dict(sys.modules, {
            "google": google, "google.cloud": cloud, "google.cloud.firestore": firestore})
        self.modules.start()
        self.env = patch.dict("os.environ", {
            "LYRIA_API_KEY": "lyria-key", "REVENUECAT_SECRET_KEY": "sk_test"})
        self.env.start()
        self.client = api.app.test_client()
        self.subscriptions = []        # what RevenueCat says this customer has
        self.rc_status = 200
        self.rc_body = None
        self.pages = None
        self.rc_calls = []
        self.lyria_calls = 0
        self.lyria_response = StubResponse({"steps": [{"type": "model_output", "content": [{
            "type": "audio", "mime_type": "audio/mpeg",
            "data": base64.b64encode(b"ID3\x03\x00\x00song").decode("ascii")}]}]})

    def tearDown(self):
        self.modules.stop()
        self.env.stop()
        for name, value in self.saved.items():
            setattr(api, name, value)

    def fake_get(self, url, headers=None, params=None, timeout=None):
        self.rc_calls.append((url, headers))
        if self.rc_status == "down":
            raise api.requests.ConnectionError("down")
        if self.rc_body is not None:
            return StubResponse(self.rc_body, self.rc_status)
        page = int(url.rsplit("starting_after=p", 1)[1]) if "starting_after=p" in url else 0
        pages = self.pages or [self.subscriptions]
        body = {"items": pages[page]}
        if page + 1 < len(pages):
            body["next_page"] = f"/v2/projects/proj83fdf9e6/customers/x/subscriptions?starting_after=p{page + 1}"
        return StubResponse(body, self.rc_status)

    def fake_post(self, *args, **kwargs):
        self.lyria_calls += 1
        return self.lyria_response

    def bake(self, rc_user_id="$RCAnonymousID:abc123"):
        payload = {"lyrics": "[Verse]\nLa la la.", "targetLanguage": "Spanish",
                   "style": "pop", "mood": "playful"}
        if rc_user_id is not None:
            payload["rcUserId"] = rc_user_id
        with patch.object(api.requests, "get", self.fake_get), \
                patch.object(api.requests, "post", self.fake_post):
            return self.client.post("/song-bake", json=payload,
                                    headers={"X-App-Key": "test-app-key"})

    def plus(self, status="active", sub_id="sub_1", environment="production"):
        self.subscriptions = [{"id": sub_id, "status": status, "gives_access": True,
                               "environment": environment}]

    def count(self, sub_id="sub_1", period=None):
        period = period or api._song_allowance_period(False)
        return self.store.get(("song_allowance", f"{sub_id}__{period}"), {}).get("count", 0)

    def test_switch_off_ignores_the_id_and_never_asks_revenuecat(self):
        api.SONG_PLUS_REQUIRED = False
        response = self.bake()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("songsLeft", response.get_json())
        self.assertEqual(self.rc_calls, [])
        self.assertEqual(self.store, {})

    def test_an_old_phone_without_an_id_is_told_to_update(self):
        response = self.bake(rc_user_id=None)
        self.assertEqual(response.status_code, 403)
        self.assertIn("update", response.get_json()["error"])
        self.assertEqual(self.lyria_calls, 0)

    def test_a_malformed_id_is_a_bad_request(self):
        self.assertEqual(self.bake(rc_user_id="../../etc").status_code, 400)
        self.assertEqual(self.bake(rc_user_id="x" * 200).status_code, 400)
        self.assertEqual(self.rc_calls, [])

    def test_no_plus_no_song_and_no_rate_slot_used(self):
        self.subscriptions = [{"id": "sub_old", "status": "expired", "gives_access": False}]
        response = self.bake()
        self.assertEqual(response.status_code, 403)
        self.assertTrue(response.get_json()["plusRequired"])
        self.assertEqual(self.lyria_calls, 0)
        self.assertEqual(api._song_daily["count"], 0, "a refused song takes no rate slot")
        self.rc_status = 404  # a phone RevenueCat never saw
        self.assertEqual(self.bake().status_code, 403)

    def test_revenuecat_is_asked_with_the_secret_key_and_a_quoted_id(self):
        self.plus()
        self.bake("$RCAnonymousID:abc123")
        url, headers = self.rc_calls[0]
        self.assertTrue(url.endswith("/projects/proj83fdf9e6/customers/%24RCAnonymousID%3Aabc123/subscriptions"))
        self.assertEqual(headers["Authorization"], "Bearer sk_test")

    def test_five_songs_a_month_then_a_plain_no(self):
        self.plus()
        lefts = [self.bake().get_json()["songsLeft"] for _ in range(5)]
        self.assertEqual(lefts, [4, 3, 2, 1, 0])
        sixth = self.bake()
        self.assertEqual(sixth.status_code, 429)
        self.assertIn("all 5 songs for this month", sixth.get_json()["error"])
        self.assertEqual(self.lyria_calls, 5, "the sixth never reaches Lyria")

    def test_the_free_week_has_two_then_the_paid_month_starts_fresh(self):
        self.plus(status="trialing")
        self.assertEqual(self.bake().get_json()["songsLeft"], 1)
        self.assertEqual(self.bake().get_json()["songsLeft"], 0)
        third = self.bake()
        self.assertEqual(third.status_code, 429)
        self.assertIn("free week", third.get_json()["error"])
        self.plus(status="active")
        self.assertEqual(self.bake().get_json()["songsLeft"], 4)

    def test_a_new_month_starts_fresh(self):
        self.plus()
        with patch.object(api, "_song_allowance_period", return_value="2026-11"):
            for _ in range(5):
                self.bake()
            self.assertEqual(self.bake().status_code, 429)
        with patch.object(api, "_song_allowance_period", return_value="2026-12"):
            self.assertEqual(self.bake().get_json()["songsLeft"], 4)

    def test_counted_by_purchase_so_a_reinstall_does_not_reset_it(self):
        self.plus(sub_id="sub_same")
        for _ in range(3):
            self.bake("$RCAnonymousID:firstinstall")
        # Restore purchases on a new install: new anonymous id, same subscription.
        self.assertEqual(self.bake("$RCAnonymousID:newinstall").get_json()["songsLeft"], 1)

    def test_a_song_that_never_came_back_gives_its_song_back(self):
        self.plus()
        self.lyria_response = StubResponse({"error": {"code": "prohibited_content"}}, 400)
        self.assertEqual(self.bake().status_code, 422)
        self.lyria_response = StubResponse({"error": {"code": "internal"}}, 500)
        self.assertEqual(self.bake().status_code, 502)
        self.lyria_response = StubResponse({"steps": []})
        self.assertEqual(self.bake().status_code, 502)
        self.assertEqual(self.count(), 0)

    def test_revenuecat_or_firestore_down_refuses_without_charging_anything(self):
        self.plus()
        self.rc_status = "down"
        down = self.bake()
        self.assertEqual(down.status_code, 503)
        self.rc_status = 500
        self.assertEqual(self.bake().status_code, 503)
        self.assertEqual(self.lyria_calls, 0)
        self.assertEqual(self.store, {})
        self.assertEqual(api._song_daily["count"], 0)

    def test_no_secret_key_on_the_server_refuses_rather_than_giving_songs_away(self):
        self.plus()
        with patch.dict("os.environ", {"REVENUECAT_SECRET_KEY": ""}):
            self.assertEqual(self.bake().status_code, 503)
        self.assertEqual(self.lyria_calls, 0)


    def test_a_delayed_request_from_an_old_period_never_resets_the_new_one(self):
        self.plus()
        with patch.object(api, "_song_allowance_period", return_value="2026-12"):
            for _ in range(5):
                self.bake()
        with patch.object(api, "_song_allowance_period", return_value="2026-11"):
            self.bake()  # a November request landing late
        with patch.object(api, "_song_allowance_period", return_value="2026-12"):
            self.assertEqual(self.bake().status_code, 429, "December stays used up")
        self.plus(status="trialing")  # a late "still in the free week" answer
        self.bake()
        self.plus(status="active")
        with patch.object(api, "_song_allowance_period", return_value="2026-12"):
            self.assertEqual(self.bake().status_code, 429)

    def test_refused_songs_still_use_up_plus_checks(self):
        api.SONG_LOOKUP_MAX_PER_WINDOW = 3
        self.subscriptions = []
        for _ in range(3):
            self.assertEqual(self.bake().status_code, 403)
        self.assertEqual(self.bake().status_code, 429)
        self.assertEqual(len(self.rc_calls), 3, "the fourth never reaches RevenueCat")

    def test_a_used_up_purchase_does_not_hide_another_with_songs_left(self):
        self.subscriptions = [
            {"id": "sub_sandbox", "status": "active", "gives_access": True, "environment": "sandbox"},
            {"id": "sub_prod", "status": "active", "gives_access": True, "environment": "production"},
        ]
        for _ in range(5):
            self.bake()
        self.assertEqual(self.count("sub_prod"), 5, "production is used first")
        self.assertEqual(self.bake().status_code, 200)
        self.assertEqual(self.count("sub_sandbox"), 1)

    def test_every_page_of_purchases_is_read(self):
        self.pages = [
            [{"id": "sub_old", "status": "expired", "gives_access": False}],
            [{"id": "sub_live", "status": "active", "gives_access": True, "environment": "production"}],
        ]
        self.assertEqual(self.bake().status_code, 200)
        self.assertEqual(self.count("sub_live"), 1)

    def test_odd_answers_and_a_broken_database_are_a_503_not_a_crash(self):
        self.plus()
        self.rc_body = []
        self.assertEqual(self.bake().status_code, 503)
        self.rc_body = None
        with patch.object(api, "_firestore", side_effect=RuntimeError("no credentials")):
            self.assertEqual(self.bake().status_code, 503)
        self.assertEqual(self.lyria_calls, 0)


    def test_the_server_wide_budget_counts_every_call_not_every_lookup(self):
        api.SONG_LOOKUP_GLOBAL_PER_MINUTE = 3
        expired = [{"id": "sub_x", "status": "expired", "gives_access": False}]
        self.pages = [expired, expired]  # two calls per lookup
        self.assertEqual(self.bake().status_code, 403)
        self.assertEqual(self.bake().status_code, 503, "the 4th call is over budget")
        self.assertEqual(len(self.rc_calls), 3)

    def test_a_history_longer_than_five_pages_is_try_again_not_no_plus(self):
        old = [{"id": "sub_old", "status": "expired", "gives_access": False}]
        live = [{"id": "sub_live", "status": "active", "gives_access": True, "environment": "production"}]
        self.pages = [old] * 5 + [live]
        response = self.bake()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("plusRequired", response.get_json())
        self.assertEqual(len(self.rc_calls), 5)


if __name__ == "__main__":
    unittest.main()
