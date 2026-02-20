import os
import chainlit as cl
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = os.path.join("vectorstore", "db_faiss")

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template,
                          input_variables=['context', 'question'])

print("⏳ System Starting...")

if os.path.exists(DB_FAISS_PATH):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ Database Loaded.")
else:
    db = None
    print("⚠️ Database missing. Run ingest.py.")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)


@cl.on_chat_start
async def start():
    if db is None:
        await cl.Message(content="⚠️ **Error:** Database not found. Please run `ingest.py` first.").send()
        return

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )
    
    await cl.Message(content="🧑‍⚕️ **Dr. AI is online.** Ask me about medical conditions in your database.").send()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    
    msg = cl.Message(content="")
    await msg.send()
    
    res = await chain.ainvoke(message.content)
    answer = res["result"]
    sources = res.get("source_documents", [])

    source_text = ""
    if sources:
        source_names = []
        for doc in sources:
            page_num = doc.metadata.get('page', 'Unknown')
            if isinstance(page_num, int):
                page_num += 1
            source_names.append(f"Page {page_num}")
        
        unique_sources = sorted(list(set(source_names)))
        source_text = f"\n\n**Sources:** {', '.join(unique_sources)}"

    msg.content = answer + source_text
    await msg.update()