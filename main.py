from fastapi import FastAPI
from pydantic import BaseModel
from graph import build_graph

app = FastAPI()

graph = build_graph()


class Message(BaseModel):
    text: str


@app.post("/detect-spam")
def detect_spam(msg: Message):

    result = graph.invoke({
        "text": msg.text,
        "result": "",
        "validation": ""
    })

    return {
        "input": msg.text,
        "classification": result["result"],
        "validation": result["validation"]
    }