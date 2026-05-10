# intro-to-fastapi

A minimal FastAPI app — my first hands-on with building a web API in Python.

## What This Is

Three simple GET endpoints that demonstrate how Python functions get exposed over HTTP. The goal was to understand how ML models are typically served to users.

## Endpoints

| Route | Description |
|-------|-------------|
| `GET /` | Returns a hello world message |
| `GET /greet?name=` | Returns a personalised greeting |
| `GET /stats?text=` | Returns word count and character count |

## Key Learnings

- The local machine acts as a server that responds to requests triggered by URLs
- Any Python function can be made accessible over the internet through an API
- This is the standard pattern for serving ML models to users

## How to Run

```bash
pip install fastapi uvicorn
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.
