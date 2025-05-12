import pytest
from run import app as flask_app

@pytest.fixture
def client():
    # configure Flask for testing
    flask_app.config["TESTING"] = True
    return flask_app.test_client()

def test_app_imports():
    # simply ensure your app object exists
    assert flask_app is not None

def test_root_endpoint(client):
    # hit “/” (or whichever route you do have) and expect a 200 or 404,
    # but not a 500
    resp = client.get("/")
    assert resp.status_code < 500

# if you have a health or docs endpoint, you can test that too:
def test_docs_available(client):
    resp = client.get("/docs")
    # swagger-ui should return HTML
    assert resp.status_code == 200
    assert b"swagger-ui" in resp.data or b"OpenAPI" in resp.data