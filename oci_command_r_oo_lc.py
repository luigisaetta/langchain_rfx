"""
Cohere command-r client OO
now fully compatible with LangChain

This is an OO version based on OCI Python SDK
last update: 12/06/2024
"""

from typing import Any, Dict, List, Optional
import logging

import json

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from oci.generative_ai_inference.models import CohereChatRequest, ChatDetails
from oci.generative_ai_inference.models import OnDemandServingMode
from oci.generative_ai_inference.models.cohere_message import CohereMessage

from oci_chat_utils import get_generative_ai_dp_client

logger = logging.getLogger("oci_command_r")


# additional
OCI_CONFIG_DIR = "~/.oci/config"
TIMEOUT = (10, 240)


class OCICommandR(BaseChatModel):
    """
    This class wraps the code to use command r in OCI

    Usage:
        chat = OCICommandR(
            model="cohere.command-r-16k",
            service_endpoint="endpoint",
            compartment_id="ocid",
            max_tokens=512
        )
    """

    client: Any
    """ the client for OCI genai"""

    model_name = "command-r"

    model: str
    """the model_id"""
    temperature: Optional[float] = 0.1
    """the temperature"""
    top_k: Optional[int] = 1
    """top_k"""
    top_p: Optional[float] = 0.1
    max_tokens: Optional[int] = 512
    """max num of tokens in output"""
    service_endpoint: str = None
    """service endpoint in OCI"""
    compartment_id: str = None
    auth_type: Optional[str] = "API_KEY"
    auth_profile: Optional[str] = "DEFAULT"
    preamble_override: Optional[str] = None
    is_streaming: Optional[bool] = False
    is_search_queries_only: Optional[bool] = (False,)
    """if true generate only a condensed query using chat_history"""

    def __init__(
        self,
        model: str,
        service_endpoint: str = None,
        compartment_id: str = None,
        temperature: Optional[float] = 0.1,
        top_k: Optional[int] = 1,
        top_p: Optional[float] = 0.1,
        max_tokens: Optional[int] = 512,
        auth_type: Optional[str] = "API_KEY",
        auth_profile: Optional[str] = "DEFAULT",
        preamble_override: Optional[str] = None,
        is_search_queries_only: Optional[bool] = False,
        is_streaming: Optional[bool] = False,
    ):
        """
        model: the id of the model
        temperature: the temperature
        top_k
        top_p
        max_tokens
        """
        super().__init__(
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            service_endpoint=service_endpoint,
            compartment_id=compartment_id,
            auth_type=auth_type,
            auth_profile=auth_profile,
            preamble_override=preamble_override,
            is_search_queries_only=is_search_queries_only,
            is_streaming=is_streaming,
        )

        # init the client to OCI
        self.client = get_generative_ai_dp_client(
            self.service_endpoint,
            self.auth_profile,
            use_session_token=False,
        )

    def invoke_custom(self, query: str, chat_history: List, documents: List):
        """
        query: user request
        chat_history: list of previous messages
        documents: list of documents to use as Context
        """

        chat_request = CohereChatRequest(
            message=query,
            chat_history=chat_history,
            # documents to use for answering
            documents=documents,
            is_search_queries_only=self.is_search_queries_only,
            preamble_override=self.preamble_override,
            max_tokens=self.max_tokens,
            is_stream=self.is_streaming,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        chat_detail = ChatDetails(
            serving_mode=OnDemandServingMode(
                # here we set the model
                model_id=self.model
            ),
            compartment_id=self.compartment_id,
            # here we insert the chat request
            chat_request=chat_request,
        )

        #
        # here we call the LLM
        #
        try:
            chat_response = self.client.chat(chat_detail)
        except Exception as e:
            logger.error("Error in invoke: %s", e)
            chat_response = None

        return chat_response

    def print_response(self, chat_response):
        """
        helper function to print LLm output
        handling streaming/no_streaming
        """
        print("")

        if self.is_streaming:
            for event in chat_response.data.events():
                res = json.loads(event.data)
                if "text" in res.keys():
                    # 9/06 (added to remove duplication of text)
                    if "finishReason" not in res.keys():
                        print(res["text"], end="", flush=True)

            print("\n")
        else:
            # no streaming
            print(chat_response.data.chat_response.text)
            print("")

    # for LangChain compatibility
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        base_params = {
            "model": self.model,
            "temperature": self.temperature,
            "preamble_override": self.preamble_override,
        }
        return {k: v for k, v in base_params.items() if v is not None}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    def _handle_request(
        self,
        messages: List[BaseMessage],
        is_streaming: Optional[bool] = False,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        # transform Messages in CohereMessages
        role_mapping = {
            HumanMessage: "USER",
            AIMessage: "CHATBOT",
            # dirty fix for now
            SystemMessage: "CHATBOT",
        }

        cohere_msgs = [
            CohereMessage(
                role=role_mapping.get(type(msg), "SYSTEM"), message=msg.content
            )
            for msg in messages
        ]

        chat_request = CohereChatRequest(
            message=cohere_msgs[-1].message,
            chat_history=cohere_msgs[:-1],
            # documents to use for answering for now in chat_history
            documents=[],
            is_search_queries_only=self.is_search_queries_only,
            preamble_override=self.preamble_override,
            max_tokens=self.max_tokens,
            is_stream=self.is_streaming,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        chat_detail = ChatDetails(
            serving_mode=OnDemandServingMode(
                # here we set the model
                model_id=self.model
            ),
            compartment_id=self.compartment_id,
            # here we insert the chat request
            chat_request=chat_request,
        )

        #
        # here we call the LLM
        #
        try:
            chat_response = self.client.chat(chat_detail)
        except Exception as e:
            logger.error("Error in invoke: %s", e)
            chat_response = None

        return chat_response

    # We need this, for LangChain compatibility... complete !
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self._handle_request(messages, is_streaming=False)

        # prepare the output
        out_message = AIMessage(
            content=response.data.chat_response.text,
            response_metadata={},
        )

        generation = ChatGeneration(message=out_message)
        return ChatResult(generations=[generation])
