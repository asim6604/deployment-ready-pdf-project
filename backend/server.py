from fastapi import FastAPI, UploadFile,Body,HTTPException
from parsing import chunking
from pypdf import PdfReader
from embedding import embedding
from vector import Vector
from main import Api
from typing import Annotated
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import io
from vector import Get_Answer

class askRequest(BaseModel):
    question:str
    hash:str

app=FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5502"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/uploadfile/")
async def create_upload_file(file:UploadFile):
    FileObject=await file.read();
   
    reader = PdfReader(io.BytesIO(FileObject))
    chunks=chunking(reader)
    Embed=embedding(chunks)
    Response=Vector(Embed,chunks,FileObject)
   
    return{"File_hash":Response}

@app.post("/ask")
async def ask(payload:askRequest):
   try:
        Question=payload;
    
        embed=embedding(Question.question)
        Response=Get_Answer(embed,Question.hash);
        Messages="\n".join(Response['documents'][0])
        Final=Api(Messages)
        return{"Response":Final}
   except:
      
        raise HTTPException(status_code=404,detail="Document not found")




    



 
