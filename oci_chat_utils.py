""""
oci_chat_utils

Code common to oci_command_r_oo and oci_llama3_oo
"""

import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.retry import NoneRetryStrategy

OCI_CONFIG_DIR = "~/.oci/config"
TIMEOUT = (10, 240)


def make_security_token_signer(oci_config):
    """
    if needed create the security token signer
    """
    pk = oci.signer.load_private_key_from_file(oci_config.get("key_file"), None)

    with open(oci_config.get("security_token_file")) as f:
        st_string = f.read()

    return oci.auth.signers.SecurityTokenSigner(st_string, pk)


def get_generative_ai_dp_client(endpoint, profile, use_session_token):
    """
    create the client for OCI GenAI
    """
    config = oci.config.from_file(OCI_CONFIG_DIR, profile)

    if use_session_token:
        signer = make_security_token_signer(oci_config=config)

        client = GenerativeAiInferenceClient(
            config=config,
            signer=signer,
            service_endpoint=endpoint,
            retry_strategy=NoneRetryStrategy(),
            timeout=TIMEOUT,
        )
    else:
        client = GenerativeAiInferenceClient(
            config=config,
            service_endpoint=endpoint,
            retry_strategy=NoneRetryStrategy(),
            timeout=TIMEOUT,
        )

    return client
