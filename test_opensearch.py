"""
test OpenSearch
"""

from opensearch_4_rfx import OpenSearchRFX


#
# Main
#

print("")

collection_list = OpenSearchRFX.list_collections()

print(collection_list)
