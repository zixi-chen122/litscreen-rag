"""
Central configuration: loads environment variables and constructs the
chat model + embeddings client, switching between providers based on
LLM_PROVIDER.

Supported values:
  - "azure_v1" : Azure AI Foundry's newer OpenAI-compatible v1 surface.
                 Endpoint looks like https://<resource>.services.ai.azure.com/openai/v1
                 Uses plain API-key auth, NO api-version needed. This is what
                 newly-created Azure OpenAI/Foundry resources show by default
                 in the "Call model" code sample (client.responses.create(...)).
  - "azure"    : The older/classic Azure OpenAI REST surface, which requires
                 an api-version query parameter (e.g. endpoint looks like
                 https://<resource>.openai.azure.com/ with no /openai/v1 suffix).
  - "openai"   : Plain OpenAI (api.openai.com), for local testing without Azure.
"""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure_v1").lower()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = "litscreen_papers"


def get_chat_model(temperature: float = 0.0):
    """Return a LangChain chat model configured for the chosen provider."""
    if LLM_PROVIDER == "azure_v1":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],  # e.g. https://<resource>.services.ai.azure.com/openai/v1
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            temperature=temperature,
        )
    elif LLM_PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=temperature,
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"),
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Use 'azure_v1', 'azure', or 'openai'.")


def get_embeddings():
    """Return a LangChain embeddings client configured for the chosen provider."""
    if LLM_PROVIDER == "azure_v1":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        )
    elif LLM_PROVIDER == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Use 'azure_v1', 'azure', or 'openai'.")
