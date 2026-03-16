from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0
)

# MAIN AGENT
main_prompt = PromptTemplate.from_template("""
You are a spam detection assistant.

Your task is to determine whether the following message is spam.

Message:
{text}

Return ONLY one of the following:

Spam
Not Spam
""")

def main_agent(text: str):

    prompt = main_prompt.format(text=text)

    response = llm.invoke(prompt)

    return response.content


# TEST AGENT
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

def test_agent(text: str, result: str):

    prompt = test_prompt.format(
        text=text,
        result=result
    )

    response = llm.invoke(prompt)

    return response.content