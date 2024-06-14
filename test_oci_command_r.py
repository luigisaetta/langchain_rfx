"""
Test for langChain implementation of CommandR
"""

from oci_command_r_oo import OCICommandR

from config_private import COMPARTMENT_ID

chat = OCICommandR(
    model="cohere.command-r-plus",
    service_endpoint="https://ppe.inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=COMPARTMENT_ID,
    max_tokens=1024,
    is_streaming=True,
)

# msg = HumanMessage(content="What are the side effects of metformin?")
task = """
Please write a documentation passage to answer the question
Question: what are the most relevant features regarding JSON we find in Oracle Database 23c?
Passage:
"""


response = chat.invoke(query=task, chat_history=[], documents=[])

chat.print_response(response)
