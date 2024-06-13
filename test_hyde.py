"""
Test hyde
"""
from factory_hyde import hyde_step1_2

QUERY = "What is Oracle AI Vector Search? How can it be used for RAG?"

answer = hyde_step1_2(QUERY)

print("")
print("Query:", QUERY)
print("")
print(answer)
print("")

