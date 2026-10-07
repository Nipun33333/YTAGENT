import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import types

# The production project depends on google-genai. The test environment does not
# need the SDK because every network generation call is mocked.
if "google" not in sys.modules:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeTypes:
        class GenerateContentConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class HttpOptions:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Part:
            @staticmethod
            def from_bytes(data, mime_type):
                return {"data": data, "mime_type": mime_type}

            @staticmethod
            def from_text(text):
                return {"text": text}

    genai.Client = _FakeClient
    genai.types = _FakeTypes
    genai_types.GenerateContentConfig = _FakeTypes.GenerateContentConfig
    genai_types.HttpOptions = _FakeTypes.HttpOptions
    genai_types.Part = _FakeTypes.Part
    google.genai = genai
    sys.modules["google"] = google
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = genai_types

# Optional runtime dependencies that are irrelevant to unit tests.
for name in ["edge_tts", "flask", "google_auth_oauthlib", "google_auth_oauthlib.flow", "google.auth", "google.auth.transport", "google.auth.transport.requests", "googleapiclient", "googleapiclient.discovery", "googleapiclient.http"]:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

sys.modules["edge_tts"].Communicate = object
sys.modules["google_auth_oauthlib.flow"].InstalledAppFlow = object
sys.modules["google.auth.transport.requests"].Request = object
sys.modules["googleapiclient.discovery"].build = lambda *a, **k: None
sys.modules["googleapiclient.http"].MediaFileUpload = object

flask = sys.modules["flask"]
class _FakeFlask:
    def __init__(self, *a, **k): pass
    def route(self, *a, **k):
        return lambda fn: fn
flask.Flask = _FakeFlask
flask.Response = object
flask.jsonify = lambda x=None, **k: x
flask.request = types.SimpleNamespace(get_json=lambda **k: {}, args={})
flask.send_file = lambda *a, **k: None
flask.abort = lambda *a, **k: None

if "yt_dlp" not in sys.modules:
    sys.modules["yt_dlp"] = types.ModuleType("yt_dlp")
