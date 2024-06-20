"""
test OpenSearch
"""
from opensearchpy import OpenSearch

from config_private import OPENSEARCH_USER, OPENSEARCH_PWD

def is_embedding(field_mapping):
    """
    function to recognize is the index contains vector
    works with the default LangChain implmentation for OpenSearch
    """
    if field_mapping.get('type') == 'float' and field_mapping.get('dims'):
        return True
    if field_mapping.get('type') == 'knn_vector':
        return True
    return False

# Configura la connessione al tuo cluster OpenSearch
host = 'localhost'
port = 9200

client = OpenSearch(
    hosts=[{'host': "localhost", 'port': 9200}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PWD),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)

# Ottieni la lista degli indici
indices = client.cat.indices(format='json')

# Stampa la lista degli indici
# for index in indices:
#    print(index['index'])

print("")

for index in indices:
    index_name = index['index']
    mappings = client.indices.get_mapping(index=index_name)
    
    for index_name, mapping in mappings.items():
        properties = mapping['mappings'].get('properties', {})
        
        for field_name, field_mapping in properties.items():
            if is_embedding(field_mapping):
                print(f"Index '{index_name}' contains embeddings in field '{field_name}'.")

