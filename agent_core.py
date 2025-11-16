import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import asyncio
import edge_tts
import os
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import spacy
import networkx as nx
import torch
import time
import hashlib

# ======================================================
# ⚙️ Device setup
# ======================================================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ======================================================
# 🎙️ Load Whisper model
# ======================================================
print("🎙️ Loading Whisper...")
stt_model = whisper.load_model("tiny.en")

# ======================================================
# 🤖 Load TinyLlama (local LLM)
# ======================================================
print("🤖 Loading TinyLlama (local LLM)...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": device},
    torch_dtype=torch.float16 if device == "mps" else torch.float32
)
pipeline_model = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=120,
    temperature=0.3,
    device_map={"": device}
)

# ======================================================
# 🔊 Text-to-Speech (TTS)
# ======================================================
_last_play = {"hash": None, "ts": 0.0}

async def speak(text):
    """
    Generate response.mp3 and return path (no backend playback).
    """
    print("🟢 speak() called")

    # Avoid duplicate playback (same text within 1.5s)
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    now = time.time()
    if _last_play["hash"] == text_hash and (now - _last_play["ts"]) < 1.5:
        print("⏭️ Duplicate speak() detected — skipping generation.")
        return os.path.join("static", "response.mp3")

    # Fix 1: Ensure 'static' directory exists before saving the file
    os.makedirs("static", exist_ok=True)

    output_path = os.path.join("static", "response.mp3")
    communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural", rate="+0%")
    await communicate.save(output_path)
    print(f"🔊 Voice saved to: {output_path}")

    _last_play["hash"] = text_hash
    _last_play["ts"] = now

    # ❌ No local playback here
    return output_path

# ======================================================
# 🧠 Memory + Knowledge Graph
# ======================================================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=ChatMessageHistory()
)

nlp = spacy.load("en_core_web_sm")
G = nx.DiGraph()

# ======================================================
# 🌐 Google Search + Embedding (RAG)
# ======================================================
def google_search_and_embed(query):
    # NOTE: The Serper API key is hardcoded and will not work without a valid key.
    header = {
        "X-API-KEY": "ddaf66551a261a0f83258c4a47cddeec509e2e57",  # Replace with your key
        "Content-Type": "application/json"
    }
    response = requests.post("https://google.serper.dev/search", headers=header, json={"q": query})
    if response.status_code != 200:
        print(f"⚠️ Serper API failed: {response.status_code}")
        return None

    data = response.json()
    links = data.get("organic", [])
    all_text = ""
    for link in links[:3]:
        try:
            page = requests.get(link["link"], headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            soup = BeautifulSoup(page.text, "html.parser")
            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
            all_text += " ".join(paragraphs) + "\n"
        except Exception as e:
            print(f"⚠️ Error fetching {link['link']}: {e}")
            continue

    if not all_text:
        return None

    chunking = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = chunking.create_documents([all_text])
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embedding)

# ======================================================
# 🧩 Core AI Logic (RAG + LLM + TTS)
# ======================================================
async def run_agent(query):
    print("🟢 run_agent() called")
    print(f"🔍 Query: {query}")

    # 1️⃣ Search Web + Create Context
    vector_db = google_search_and_embed(query)
    context = ""

    if vector_db:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query.strip())
        context = "\n".join(doc.page_content for doc in retrieved_docs)

        # 2️⃣ Build Knowledge Graph
        for doc in retrieved_docs:
            doc_nlp = nlp(doc.page_content)
            for sent in doc_nlp.sents:
                entities = [ent.text for ent in sent.ents]
                if len(entities) >= 2:
                    for i in range(len(entities) - 1):
                        G.add_edge(entities[i], entities[i + 1], relation="related_to")
                for ent in entities:
                    G.add_node(ent)

    # 3️⃣ Use Memory + Knowledge Graph
    kg_info = ""
    if G.number_of_edges() > 0:
        kg_info = "\nKnowledge Graph Triples:"
        for u, v, data in list(G.edges(data=True))[:5]:
            kg_info += f"\n({u}, {data['relation']}, {v})"

    chat_history = memory.load_memory_variables({})["chat_history"]

    prompt = f"""
You are a highly intelligent assistant. Use the given web context, past chat memory, 
and knowledge graph to answer the user's question precisely.

Chat History:
{chat_history}

Context:
{context}

{kg_info}

Question:
{query}

Answer:
"""

    # 4️⃣ Generate Answer
    result = pipeline_model(prompt)[0]["generated_text"]
    answer = result.split("Answer:")[-1].strip() if "Answer:" in result else result.strip()
    memory.save_context({"input": query}, {"output": answer})
    print(f"🤖 Answer: {answer}")

    # 5️⃣ Convert Answer to Speech
    await speak(answer)
    return answer

# ======================================================
# 🎤 Record & Transcribe (Speech-to-Text)
# ======================================================
def record_voice(filename="voice_input.wav", duration=7, fs=16000):
    print("👂🏽 Speak now (7 seconds)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("✅ Voice recorded.")
    return filename

def speech_to_text(filename):
    print("🧠 Transcribing speech...")
    result = stt_model.transcribe(filename)
    text = result["text"]
    print(f"🗣 You said: {text}")
    return text

# ======================================================
# 🚀 Main Loop (Speech → Text → RAG → Speech)
# ======================================================
async def async_main():
    print("\n🎯 AI Voice RAG Agent")
    print("Speak a question, and it will search + reason + reply aloud.\n")

    while True:
        input("🎤 Press ENTER and start speaking:")
        print("🟢 Entering new main loop iteration...")
        
        # NOTE: The following two lines require a microphone and will not work in the sandbox.
        # The user must test this part locally.
        # audio_file = record_voice(duration=7)
        # query = speech_to_text(audio_file)
        
        # For testing the async structure fix, we will use a dummy query.
        query = "What is the capital of France?"
        print(f"🗣 Using dummy query: {query}")
        
        await run_agent(query)

if __name__ == "__main__":
    # Fix 2: Run the main loop using asyncio.run() only once
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n👋 Agent stopped.")
    except RuntimeError as e:
        # This catches the "RuntimeError: Event loop is closed" that can happen
        # when asyncio.run is called repeatedly, which was the original structure.
        # The new structure should prevent this, but it's good practice to handle it.
        if "Event loop is closed" in str(e):
            print("⚠️ Event loop error caught. The agent is stopping.")
        else:
            raise





