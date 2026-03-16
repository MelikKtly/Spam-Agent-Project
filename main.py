from graph import build_graph
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

graph= build_graph()

class Message(BaseModel):
    text: str


@app.post("/classify")
def detect_spam(message: Message):
    result= graph.invoke({
        "text":message.text
    })
    return {
        "input": msg.text,
        "classification": result["result"],
        "validation": result["validation"]
    }