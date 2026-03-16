from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


llm=ChatOllama(    
    model="qwen3:1.7b", 
    temperature=0
    )

main_prompt = PromptTemplate.from_template(
    """You are spam detection agent. You will be given a message and you need to determine if it is spam or not.
    Message: {text}
    Is this message spam? Answer with "Yes" or "No" and explain your reasoning.
    """
)

def main_agent(text):
    response = llm(main_prompt.format(text=text))
    return response


test_prompt = PromptTemplate.from_template("""
You are a validation agent.

Original message:
{text}

Main agent classification:
{result}

Check whether this classification is correct.

Return:

Correct
or
Incorrect

Explain shortly.
""")


def test_agent(text, result):
    response = llm(test_prompt.format(text=text, result=result))
    return response

