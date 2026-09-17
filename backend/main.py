from groq import Groq
from dotenv import load_dotenv
from vector import Vector
load_dotenv()


client=Groq()
def Api(messages):

    chat_completion=client.chat.completions.create(
    messages=[
    {
    "role":"system",
    "content":"You are assistant your job is to read the question of user and then relevant chunks  of answers would be given to you and you have to answer the question of user using that chunks and if u get the answer as empty array and return answer as error no answer  "
    },{
        "role":"user",
        "content":"what is the shape of the earth"

    },{
        "role":"assistant",
        "content":messages
    }
    ],




    model="openai/gpt-oss-120b",
    )
    return(chat_completion.choices[0].message.content)