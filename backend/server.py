from fastapi import FastAPI, UploadFile, HTTPException
from parsing import chunking
from pypdf import PdfReader
from embedding import embedding
from vector import Vector, Get_Answer
from groq import Groq
from dotenv import load_dotenv
from main import Api
from tavily import TavilyClient
import json
import io
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = Groq()
tavily_client = TavilyClient()

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
    },
   {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "If the question is out of the pdf and you needed a little bit of help to search online, you can use this tool to get online info",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for the web"
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
    "http://127.0.0.1:5500",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def Web_search(query:str):
    try:
        print("hit")
        result=""

        response=  tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True
        )
        for i in response.get("results",[]):
            result += i.get('content', '') + "\n"

        return result
    except Exception as e:
        print(f"An error occurred: {e}")
async def get_answer(question: str, collection_hash: str):
       result = await run_agent(question, collection_hash)
       verdict=check_faithfulness(question,result["answer"],result["context"])
       if verdict.lower().startswith("faithful"):
          return result["answer"]
       stricter_question = question + "\n\nImportant: only use information explicitly present in the retrieved document content. Do not add any facts, numbers, or details not directly stated there."
       retry_result =await run_agent(stricter_question, collection_hash)
       retry_verdict = check_faithfulness(question, retry_result["answer"], retry_result["context"])

       if retry_verdict.lower().startswith("faithful"):
           return retry_result["answer"]
       return {"answer": "Sorry, I couldn't find a complete answer."}
           
     
def check_faithfulness(question:str,answer:str,context:list[str])->str:
    context="\n".join(context)
    judge_prompt = f"""
    question:{question}
    Context from document:
    {context}
    
    Answer given:
    {answer}

    Does the answer only use information that is actually present in the context above? ...
    Reply with exactly one word first: "faithful" or "unfaithful", then a one-line reason."""
    response=client.chat.completions.create(
         model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": judge_prompt}]
    )
    return  response.choices[0].message.content
      
    
def search_docs(query: str, collection_hash: str):
    embed = embedding(query)
    response = Get_Answer(embed, collection_hash)
    return response["documents"][0]

async def  run_agent(question: str, collection_hash: str):
    messages = [{"role": "user", "content": question}]
    retrieved_chunk=[]
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
                    print("tool called:", repr(args["query"]))
                    result = search_docs(
                        query=args["query"], collection_hash=collection_hash
                    )
                    retrieved_chunk.extend(result)
                    result_text = "\n".join(result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })
                elif tool_call.function.name=="web_search":
                    result= Web_search(args["query"])
                    result_text=result
                    messages.append({
                     "role":"tool",
                     "tool_call_id":tool_call.id,
                    "content":result_text
                    }
                       
                    )
        else:
            return ({"answer":message.content,"context":retrieved_chunk})
            
    return {"answer": "Sorry, I couldn't find a complete answer.", "context": retrieved_chunk}


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
        answer=await get_answer(payload.question,payload.hash)
        return {"Response": answer}
        
       
    except Exception as e:
        print("REAL ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


