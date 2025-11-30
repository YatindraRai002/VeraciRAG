"""
Configuration settings for the Self-Correcting RAG system
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration
RETRIEVAL_MODEL = "gpt-3.5-turbo"
GUARDIAN_MODEL = "gpt-4"  # More capable model for evaluation
GENERATOR_MODEL = "gpt-4"
EVALUATOR_MODEL = "gpt-4"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
TOP_K_DOCUMENTS = 5
RELEVANCE_THRESHOLD = 0.6  # Guardian's minimum relevance score

# Evaluation Configuration
FACTUAL_CONSISTENCY_THRESHOLD = 0.7  # Evaluator's minimum consistency score
