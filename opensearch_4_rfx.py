"""
Add extension for using OpenSearch as Vector Store
"""

from opensearchpy import OpenSearch

from config import OPENSEARCH_SHARED_PARAMS, OPENSEARCH_URL
from config_private import OPENSEARCH_USER, OPENSEARCH_PWD


def is_embedding(field_mapping):
    """
    function to recognize if the index contains vector
    works with the default LangChain implmentation for OpenSearch
    """

    # maybe there could be the need to customize
    if field_mapping.get("type") == "knn_vector":
        return True
    return False


class OpenSearchRFX:
    """
    extension to get collection list  and books_list
    from OpenSearch
    """

    @classmethod
    def _get_client(cls):
        client = OpenSearch(
            OPENSEARCH_URL,
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PWD),
            use_ssl=OPENSEARCH_SHARED_PARAMS["use_ssl"],
            verify_certs=OPENSEARCH_SHARED_PARAMS["verify_certs"],
            ssl_show_warn=OPENSEARCH_SHARED_PARAMS["ssl_show_warn"],
        )
        return client

    @classmethod
    def list_collections(cls):
        """
        get the list of all the collections
        """
        client = cls._get_client()

        collections = []

        indices = client.cat.indices(format="json")

        for index in indices:
            index_name = index["index"]
            mappings = client.indices.get_mapping(index=index_name)

            for index_name, mapping in mappings.items():
                properties = mapping["mappings"].get("properties", {})

                for field_name, field_mapping in properties.items():
                    if is_embedding(field_mapping):
                        # print(f"Index '{index_name}'
                        # contains embeddings in field '{field_name}'.")

                        collections.append(index_name)

        return collections

    @classmethod
    def list_books(cls, collection_name):
        """
        get the list of all the books
        """
        client = cls._get_client()

        # this query works if in metadata we're loading
        # source with book name
        query = {"query": f"SELECT DISTINCT metadata.source FROM {collection_name}"}

        response = client.transport.perform_request(
            method="POST", url="/_plugins/_sql", body=query
        )

        books_list = []

        for row in response["datarows"]:
            books_list.append(row[0])

        return books_list
