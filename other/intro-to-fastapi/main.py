from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Hello World"}

@app.get("/greet")
def greet(name: str):
    return {"message": f"Hello {name}"}

@app.get("/stats")
def bruh(text: str):
    chr_count = len(set(text))
    word_count = len(text.split())
    return {"word count": word_count,"character count":chr_count}