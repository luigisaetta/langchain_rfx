""""
Experimental client for Llama3 in OCI
compatible with LangChain
inspired by: 
    https://python.langchain.com/v0.1/docs/modules/model_io/chat/custom_chat_model/
    
last update: 10/06/2024
"""

from typing import Any, Dict, List, Optional, Iterator
import logging
import json

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.outputs import ChatGenerationChunk

from oci.generative_ai_inference.models import GenericChatRequest, ChatDetails
from oci.generative_ai_inference.models import BaseChatRequest, TextContent, Message
from oci.generative_ai_inference.models import OnDemandServingMode

from oci_chat_utils import get_generative_ai_dp_client

logger = logging.getLogger("oci_llama3")

# additional
OCI_CONFIG_DIR = "~/.oci/config"
TIMEOUT = (10, 240)


class OCILlama3(BaseChatModel):
    """
    This class wraps the code to use llama3 r in OCI

    Usage:
        chat = OCILlama3(
            model="meta.llama3-3-70b-instruct",
            service_endpoint="endpoint",
            compartment_id="ocid",
            max_tokens=512
        )
    """

    client: Any
    """ the client for OCI genai"""

    model_name = "meta.llama-3-70b-instruct"

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
    is_streaming: Optional[bool] = False

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
            is_streaming=is_streaming,
        )
        # init the client to OCI
        self.client = get_generative_ai_dp_client(
            self.service_endpoint,
            self.auth_profile,
            use_session_token=False,
        )

    #
    # This is called by streaming and non-streaming
    #
    def _handle_request(
        self,
        messages: List[BaseMessage],
        is_streaming: Optional[bool] = False,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """
        This is called by streaming and non-streaming

        is_streaming: to discriminate if the request comes from stream or generate
        """
        # prepare the request for OCI Python SDK

        # process the list of messages and translate to OCI format
        role_map = {HumanMessage: "USER", AIMessage: "ASSISTANT"}

        oci_msgs = []
        for in_msg in messages:
            role = role_map.get(type(in_msg), "SYSTEM")

            content = TextContent(text=in_msg.content)
            message = Message(role=role, content=[content])

            oci_msgs.append(message)

        chat_request = GenericChatRequest(
            api_format=BaseChatRequest.API_FORMAT_GENERIC,
            messages=oci_msgs,
            is_stream=is_streaming,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        chat_detail = ChatDetails(
            serving_mode=OnDemandServingMode(model_id=self.model),
            chat_request=chat_request,
            compartment_id=self.compartment_id,
        )

        #
        # here we call the LLM
        #
        try:
            response = self.client.chat(chat_detail)
        except Exception as e:
            logger.error("Error in invoke: %s", e)
            response = None

        return response

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
            content=response.data.chat_response.choices[0].message.content[0].text,
            response_metadata={},
        )

        generation = ChatGeneration(message=out_message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        response = self._handle_request(messages, is_streaming=True)

        for event in response.data.events():
            res = json.loads(event.data)
            if "message" in res.keys():
                content = res["message"]["content"][0]["text"]
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                yield chunk

        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={})
        )
        yield chunk

    def print_response(self, chat_response):
        """
        helper function to print handling streaming/no_streaming
        """
        print("")

        if self.is_streaming:
            # 9/06/2024: updated to support streaming
            for chunk in chat_response:
                print(chunk.content, end="", flush=True)

            print("\n")
        else:
            # no streaming
            print(chat_response.content)
            print("")

    # for LangChain compatibility
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling."""
        base_params = {
            "model": self.model,
            "temperature": self.temperature,
        }
        return {k: v for k, v in base_params.items() if v is not None}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params
