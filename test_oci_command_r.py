"""
Test for custom implementation of CommandR
"""

from oci_command_r_oo import OCICommandR

from config import ENDPOINT
from config_private import COMPARTMENT_ID

MODEL_ID = "cohere.command-r-plus"

chat = OCICommandR(
    model=MODEL_ID,
    service_endpoint=ENDPOINT,
    compartment_id=COMPARTMENT_ID,
    max_tokens=1024,
    is_streaming=True,
)

# msg = HumanMessage(content="What are the side effects of metformin?")
TASK = """
Please write a documentation passage to answer the question
Question: what are the most relevant features regarding JSON we find in Oracle Database 23c?
Passage:
"""


response = chat.invoke(query=TASK, chat_history=[], documents=[])

chat.print_response(response)
