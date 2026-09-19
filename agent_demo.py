"""
ReAct Agent：把 RAG 检索封装成工具，由 Agent 自主决定何时调用
对应简历：研究 ReAct 智能体框架，实践 RAG 作为工具被 Agent 调用
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
    temperature=0,
)

rag = RAGPipeline(doc_path="doc.txt", persist_dir="./chroma_db")


@tool
def search_knowledge_base(query: str) -> str:
    """当用户询问与项目文档相关的问题时，用这个工具检索本地知识库。
    参数 query：要检索的问题，例如"AI应用开发实习需要哪些技术"。
    """
    contexts = rag.hybrid_retrieve(query, top_k=3)
    return "\n---\n".join(contexts)


tools = [search_knowledge_base]

prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


if __name__ == "__main__":
    print("Agent 已启动（RAG 作为工具被调用）：")
    q = "AI应用开发实习需要掌握哪些技术？"
    print("问题：", q)
    out = agent_executor.invoke({"input": q})
    print("回答：", out["output"])
