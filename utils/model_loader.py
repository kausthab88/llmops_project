from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from logger.custom_logger import CustomLogger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os
import sys
from exception.custom_exception import DocumentPortalException


log = CustomLogger().get_logger(__name__)

class Model_Loader:

    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        log.info("config loaded successfully", config_keys = list(self.config.keys()))

    def _validate_env(self):
        required_vars=["GROQ_API_KEY","GOOGLE_API_KEY","CLAUDE_API_KEY","OPENAI_API_KEY"]
        self.api_keys= {key:os.getenv(key) for key in required_vars}
        missing = [k for k,v in self.api_keys.items() if not v]
        if missing:
            log.error("Missing environment variables", missing_vars=missing)
            raise DocumentPortalException("Missing environment variables", sys)

    def load_embeddings(self):
        
        try:
            log.info("loading embedding model...")
            provider = self.config["embedding_model"]["provider"]
            model_name = self.config["embedding_model"]["model_name"]

            if provider == "openai":
                return OpenAIEmbeddings(
                    model=model_name,
                    api_key=self.api_keys["OPENAI_API_KEY"]
                )
            
            elif provider == "google":  
                return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_keys["GOOGLE_API_KEY"]
            )
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
                #return ChatGoogleGenerativeAI(model=model_name)
        except Exception as e:
            log.error("Error loading embedding model", error = str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        llm_block = self.config["llm"]
        log.info("Loading LLM...")
        provider_key = os.getenv("LLM_PROVIDER", "groq")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider_key=provider_key)
            raise ValueError(f"Provider '{provider_key}' not found in config")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature =llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider = provider, model= model_name, temperature = temperature, max_tokens = max_tokens)

        if provider == "google":
            llm = ChatGoogleGenerativeAI(
                  model=model_name,
                  google_api_key = self.api_keys["GOOGLE_API_KEY"],
                  temperature = temperature,
                  max_output_tokens = max_tokens               
            )
            return llm
        
        elif provider == "groq":
            llm = ChatGroq(
                model=model_name,
                api_key = self.api_keys["GROQ_API_KEY"],
                temperature = temperature,
                max_tokens = max_tokens
            )
            return llm
        
        elif provider == "openai":
            llm = ChatOpenAI(
                model=model_name,
                api_key = self.api_keys["OPENAI_API_KEY"],
                temperature = temperature,
                max_tokens = max_tokens
            )
            return llm
        
if __name__ == "__main__":
    loader = Model_Loader()

    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")

    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")

    result= llm.invoke("Helloo, what day is it today?")
    print(f"LLM result: {result.content}")

    embed_result = embeddings.embed_query("Helloo, what day is it today?")
    print(f"Embedding result: {embed_result}")