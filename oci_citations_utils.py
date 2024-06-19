"""
This file contains utilities to extract citations
from Cohere command-r/r-plus response

last update: 07/06/2024
"""

# to extract all the info regarding citations
# Extract start, end, and document_ids
from oci.response import Response


def extract_document_list(response: Response):
    """
    This function extract form the Cohere response
    in the section for citations the list of documents
    """
    extracted_docs = []

    # we must handle the case of no citations
    if response.data.chat_response.documents is not None:
        for doc in response.data.chat_response.documents:
            extracted_docs.append(
                {"id": doc["id"], "source": doc["source"], "page": doc["page"]}
            )

    # sort in order of increasing id
    sorted_docs = sorted(extracted_docs, key=lambda x: x["id"])

    return sorted_docs


def extract_citations_from_response(response: Response):
    """
    This function extract form the Cohere response
    in the section for citations the list of citations
    """
    extracted_data = []

    # we must handle the case of no citations
    if response.data.chat_response.citations is not None:
        for item in response.data.chat_response.citations:
            extracted_info = {
                "start": item.start,
                "end": item.end,
                "text": item.text,
                "document_ids": item.document_ids,
            }
            extracted_data.append(extracted_info)

    return extracted_data


def find_source_page_by_id(data: dict, search_id) -> tuple:
    """
    find source page of a document from the id
    """
    for item in data:
        if item["id"] == search_id:
            return item["source"], item["page"]
    # if id not found
    return None, None


# this functions complete citations with source (name of doc) and page


def extract_complete_citations(response: Response) -> list:
    """
    This function extract from the Cohere response
    documents and citations and complete citations with source, page
    """
    extracted_doc = extract_document_list(response)
    extracted_citations = extract_citations_from_response(response)

    complete_citations = []
    for citation in extracted_citations:
        document_ids = citation["document_ids"]

        documents = []
        for doc_id in document_ids:
            source, page = find_source_page_by_id(extracted_doc, doc_id)
            documents.append({"id": doc_id, "source": source, "page": page})

        new_citation = {
            "interval": (citation["start"], citation["end"]),
            "text": citation["text"],
            "documents": documents,
        }
        complete_citations.append(new_citation)

    return complete_citations
