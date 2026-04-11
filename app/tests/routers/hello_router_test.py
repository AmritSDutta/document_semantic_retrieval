from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers.hello import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


def test_read_main():
    response = client.get("/api/hello")
    assert response.status_code == 200
    assert response.json() == {"message": "hi"}
