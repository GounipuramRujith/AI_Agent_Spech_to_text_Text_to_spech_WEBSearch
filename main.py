# main.py
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import agent_core  # your working file

app = FastAPI()

# Serve static files (like response.mp3 and index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


# --- Text chat ---
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    if not query:
        return JSONResponse({"response": "❌ Empty query"})
    answer = await agent_core.run_agent(query)
    return JSONResponse({"response": answer})


# --- Voice chat (Speech → Text → Answer) ---
@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    filename = "voice_input.wav"
    with open(filename, "wb") as f:
        f.write(await file.read())

    query_text = agent_core.speech_to_text(filename)
    answer = await agent_core.run_agent(query_text)
    return JSONResponse({"query_text": query_text, "response": answer})


# --- Entry point ---
if __name__ == "__main__":
    import uvicorn, os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("🚀 FastAPI running locally at: http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)







