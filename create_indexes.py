import json
import logging
import os
import subprocess

from functools import partial
from dotenv import load_dotenv

# Azure packages
from azure.core.exceptions import ResourceExistsError
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIParameters,
    AzureOpenAIVectorizer,
    FieldMapping,
    HnswAlgorithmConfiguration,
    HnswParameters,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataSourceType,
    SearchIndexerIndexProjections,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkill,
    SearchIndexerSkillset,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    SplitSkill,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from rich.logging import RichHandler

# -----------------------------------------------------------------------------
# Logger setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Load environment variables from local .env
# -----------------------------------------------------------------------------
def load_azd_env():
    """
    Load environment variables from a local .env file using python-dotenv.
    """
    env_file_path = ".env"  # Specify the local .env file
    if not os.path.exists(env_file_path):
        raise Exception(f"Error: {env_file_path} file not found")

    logger.info(f"Loading environment variables from {env_file_path}")
    load_dotenv(env_file_path, override=True)

# -----------------------------------------------------------------------------
# Create or update Data Source, Search Index, Skillset, and Indexer
# -----------------------------------------------------------------------------
def setup_index(
    azure_credential,
    index_name,
    azure_search_endpoint,
    azure_storage_connection_string,
    azure_storage_container,
    azure_openai_embedding_endpoint,
    azure_openai_embedding_deployment,
    azure_openai_embedding_model,
    azure_openai_embeddings_dimensions
):
    """
    Creates or updates:
    1) Data Source
    2) Search Index
    3) Skillset (Split → Embeddings)
    4) Indexer
    """

    index_client = SearchIndexClient(azure_search_endpoint, azure_credential)
    indexer_client = SearchIndexerClient(azure_search_endpoint, azure_credential)

    # -------------------------------------------------------------------------
    # 1) DATA SOURCE
    # -------------------------------------------------------------------------
    data_source_connections = indexer_client.get_data_source_connections()
    if index_name in [ds.name for ds in data_source_connections]:
        logger.info(f"Data source connection '{index_name}' already exists. Not re-creating.")
    else:
        logger.info(f"Creating data source connection: {index_name}")
        data_source = SearchIndexerDataSourceConnection(
            name=index_name,
            type=SearchIndexerDataSourceType.AZURE_BLOB,
            connection_string=azure_storage_connection_string,
            container=SearchIndexerDataContainer(name=azure_storage_container),
            data_to_extract="contentAndMetadata",      
            parsing_mode="default"                     
        )
        indexer_client.create_data_source_connection(data_source)

    # -------------------------------------------------------------------------
    # 2) SEARCH INDEX (with vector field)
    # -------------------------------------------------------------------------
    existing_indexes = [idx.name for idx in index_client.list_indexes()]
    if index_name in existing_indexes:
        logger.info(f"Search Index '{index_name}' already exists. Not re-creating.")
    else:
        logger.info(f"Creating Search Index: {index_name}")
        index_client.create_index(
            SearchIndex(
                name=index_name,
                fields=[
                    # The primary key
                    SearchableField(
                        name="chunk_id", 
                        key=True,
                        analyzer_name="keyword",
                        sortable=True
                    ),
                    # Parent ID field
                    SimpleField(
                        name="parent_id",
                        type=SearchFieldDataType.String,
                        filterable=True
                    ),
                    # Title field
                    SearchableField(name="title"),
                    # Actual text chunk
                    SearchableField(name="chunk"),
                    # The vector field to store embeddings
                    SearchField(
                        name="text_vector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        vector_search_dimensions=azure_openai_embeddings_dimensions,
                        vector_search_profile_name="vp",
                        stored=True,
                        hidden=False
                    )
                ],
                vector_search=VectorSearch(
                    algorithms=[
                        HnswAlgorithmConfiguration(
                            name="algo",
                            parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE)
                        )
                    ],
                    vectorizers=[
                        AzureOpenAIVectorizer(
                            name="openai_vectorizer",
                            azure_open_ai_parameters=AzureOpenAIParameters(
                                resource_uri=azure_openai_embedding_endpoint,
                                deployment_id=azure_openai_embedding_deployment,
                                model_name=azure_openai_embedding_model
                            )
                        )
                    ],
                    profiles=[
                        VectorSearchProfile(
                            name="vp",
                            algorithm_configuration_name="algo",
                            vectorizer="openai_vectorizer"
                        )
                    ]
                ),
                semantic_search=SemanticSearch(
                    configurations=[
                        SemanticConfiguration(
                            name="default",
                            prioritized_fields=SemanticPrioritizedFields(
                                title_field=SemanticField(field_name="title"),
                                content_fields=[SemanticField(field_name="chunk")]
                            )
                        )
                    ],
                    default_configuration_name="default"
                )
            )
        )

    # -------------------------------------------------------------------------
    # 3) SKILLSET (Split → Embedding)
    # -------------------------------------------------------------------------
    skillsets = indexer_client.get_skillsets()
    if index_name in [skillset.name for skillset in skillsets]:
        logger.info(f"Skillset {index_name} already exists, not re-creating")
    else:
        logger.info(f"Creating skillset: {index_name}")
        indexer_client.create_skillset(
            skillset=SearchIndexerSkillset(
                name=index_name,
                skills=[
                    SplitSkill(
                        text_split_mode="pages",
                        context="/document",
                        maximum_page_length=2000,
                        page_overlap_length=500,
                        inputs=[InputFieldMappingEntry(name="text", source="/document/content")],  # Ensure text source
                        outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")]  # Ensure output is mapped
                    ),
                    AzureOpenAIEmbeddingSkill(
                        context="/document/pages/*",
                        resource_uri=azure_openai_embedding_endpoint,
                        api_key=None,
                        deployment_id=azure_openai_embedding_deployment,
                        model_name=azure_openai_embedding_model,
                        dimensions=azure_openai_embeddings_dimensions,
                        inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],  # Ensure input exists
                        outputs=[OutputFieldMappingEntry(name="embedding", target_name="text_vector")]
                    )
                ],
                index_projections=SearchIndexerIndexProjections(
                    selectors=[
                        SearchIndexerIndexProjectionSelector(
                            target_index_name=index_name,
                            parent_key_field_name="parent_id",
                            source_context="/document/pages/*",
                            mappings=[
                                InputFieldMappingEntry(name="chunk", source="/document/pages/*"),
                                InputFieldMappingEntry(name="text_vector", source="/document/pages/*/text_vector"),
                                InputFieldMappingEntry(name="title", source="/document/metadata_storage_name")
                            ]
                        )
                    ],
                    parameters=SearchIndexerIndexProjectionsParameters(
                        projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
                    )
                )))

    # -------------------------------------------------------------------------
    # 4) INDEXER (data source → skillset → index)
    # -------------------------------------------------------------------------
    indexers = indexer_client.get_indexers()
    if index_name in [indexer.name for indexer in indexers]:
        logger.info(f"Indexer {index_name} already exists, not re-creating")
    else:
        indexer_client.create_indexer(
            indexer=SearchIndexer(
                name=index_name,
                data_source_name=index_name,
                skillset_name=index_name,
                target_index_name=index_name,
                field_mappings=[
                    FieldMapping(
                        source_field_name="metadata_storage_name", 
                        target_field_name="title"
                    )
                ]
            )
        )


# -----------------------------------------------------------------------------
# Upload local documents to Azure Blob & run the indexer
# -----------------------------------------------------------------------------
def upload_documents(
    azure_credential,
    indexer_name,
    azure_search_endpoint,
    azure_storage_endpoint,
    azure_storage_container
):
    """
    1) Upload files from enterprise-data/ folder to the specified Blob Storage container.
    2) Run the indexer to process (extract → split → embed) your documents.
    """
    indexer_client = SearchIndexerClient(azure_search_endpoint, azure_credential)

    # 1) Upload local files -> Blob
    blob_client = BlobServiceClient.from_connection_string(
        conn_str=AZURE_STORAGE_CONNECTION_STRING,
        max_single_put_size=4 * 1024 * 1024
    )

    container_client = blob_client.get_container_client(azure_storage_container)
    if not container_client.exists():
        container_client.create_container()

    existing_blobs = [blob.name for blob in container_client.list_blobs()]

    for file_entry in os.scandir("enterprise-data"):
        if not file_entry.is_file():
            continue
        with open(file_entry.path, "rb") as opened_file:
            filename = os.path.basename(file_entry.path)
            if filename in existing_blobs:
                logger.info("Blob already exists, skipping file: %s", filename)
            else:
                logger.info("Uploading blob for file: %s", filename)
                container_client.upload_blob(filename, opened_file, overwrite=True)

    # 2) Trigger the indexer
    try:
        logger.info(f"Running indexer: {indexer_name}")
        indexer_client.run_indexer(indexer_name)
        logger.info(
            "Indexer started. Newly uploaded documents will be indexed in a few minutes. "
            "Check the Azure Portal for progress."
        )
    except ResourceExistsError:
        logger.info("Indexer is already running. Not starting again.")


# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_azd_env()

    logger.info("Checking if we need to set up Azure AI Search index...")

    if os.environ.get("AZURE_SEARCH_REUSE_EXISTING") == "true":
        logger.info(
            "AZURE_SEARCH_REUSE_EXISTING=true -> using existing Azure AI Search index. "
            "No changes made to the index, skillset, or indexer."
        )
        exit()
    else:
        logger.info("Setting up Azure AI Search index with integrated vectorization...")

    # 1) Gather required environment variables
    AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
    AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    AZURE_OPENAI_EMBEDDING_MODEL = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
    EMBEDDINGS_DIMENSIONS = 3072 

    AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
    AZURE_STORAGE_ENDPOINT = os.environ["AZURE_STORAGE_ENDPOINT"]
    AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    AZURE_STORAGE_CONTAINER = os.environ["AZURE_STORAGE_CONTAINER"]
    AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

    azure_credential = AzureDeveloperCliCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        process_timeout=60
    )

    # 2) Create/Update Data Source, Index, Skillset, Indexer
    setup_index(
        azure_credential,
        index_name=AZURE_SEARCH_INDEX,
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_storage_connection_string=AZURE_STORAGE_CONNECTION_STRING,
        azure_storage_container=AZURE_STORAGE_CONTAINER,
        azure_openai_embedding_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        azure_openai_embedding_model=AZURE_OPENAI_EMBEDDING_MODEL,
        azure_openai_embeddings_dimensions=EMBEDDINGS_DIMENSIONS
    )

    # 3) Upload Documents & Run the Indexer
    upload_documents(
        azure_credential,
        indexer_name=AZURE_SEARCH_INDEX,
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_storage_endpoint=AZURE_STORAGE_ENDPOINT,
        azure_storage_container=AZURE_STORAGE_CONTAINER
    )