"""
Test for langChain implementation of CommandR
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from oci_command_r_oo_lc import OCICommandR

from config import ENDPOINT
from config_private import COMPARTMENT_ID

chat = OCICommandR(
    model="cohere.command-r-16k",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=COMPARTMENT_ID,
    max_tokens=1024,
)

msg = HumanMessage(content="What are the side effects of metformin?")

response = chat.invoke(input=[msg])

print(response.content)
