from fastapi import FastAPI, UploadFile, HTTPException
from parsing import chunking
from pypdf import PdfReader
from embedding import embedding
from vector import Vector, Get_Answer
from groq import Groq
from dotenv import load_dotenv
from main import Api
import json
import io
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = Groq()

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the uploaded PDF document for relevant information to answer the user's question. Use this whenever you need facts or details from the document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, what to look for in the document"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


class askRequest(BaseModel):
    question: str
    hash: str


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5502",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def search_docs(query: str, collection_hash: str):
    embed = embedding(query)
    response = Get_Answer(embed, collection_hash)
    return response["documents"][0]


def run_agent(question: str, collection_hash: str):
    messages = [{"role": "user", "content": question}]
    for i in range(5):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "search_docs":
                    print("tool called:", args["query"])
                    result = search_docs(
                        query=args["query"], collection_hash=collection_hash
                    )
                    result_text = "\n".join(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })
        else:
            return message.content
    return "Sorry, I couldn't find a complete answer."


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    FileObject = await file.read()
    reader = PdfReader(io.BytesIO(FileObject))
    chunks = chunking(reader)
    Embed = embedding(chunks)
    Response = Vector(Embed, chunks, FileObject)
    return {"File_hash": Response}


@app.post("/ask")
async def ask(payload: askRequest):
    try:
        embed = embedding(payload.question)
        Response = Get_Answer(embed, payload.hash)
        Messages = "\n".join(Response["documents"][0])
        Final = Api(Messages)
        return {"Response": Final}
    except:
        raise HTTPException(status_code=404, detail="Document not found")


if __name__ == "__main__":
    print(run_agent("a question your PDF can answer", "PASTE_HASH_HERE"))