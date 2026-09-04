from dotenv import load_dotenv
from rag import ask_question

import os

load_dotenv()

#print(os.getenv("OPENAI_API_KEY"))


q = input("Ask: ")
print(ask_question(q))