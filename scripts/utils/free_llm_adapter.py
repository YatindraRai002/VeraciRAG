"""
Free LLM Integration - Use Open Source Models
Integrate with HuggingFace, Ollama, or other free LLM providers
"""
import os
from typing import List, Dict, Optional

# Configuration for different free LLM providers
FREE_LLM_CONFIGS = {
    "ollama": {
        "description": "Run models locally with Ollama",
        "install": "Download from https://ollama.ai",
        "models": ["llama2", "mistral", "phi", "neural-chat"],
        "cost": "FREE - runs on your machine"
    },
    "huggingface": {
        "description": "Use HuggingFace models",
        "install": "pip install transformers torch",
        "models": ["google/flan-t5-large", "mistralai/Mistral-7B", "meta-llama/Llama-2-7b"],
        "cost": "FREE with API key or local"
    },
    "groq": {
        "description": "Fast inference with Groq",
        "install": "pip install groq",
        "models": ["llama2-70b", "mixtral-8x7b"],
        "cost": "FREE tier available"
    },
    "together": {
        "description": "Together AI platform",
        "install": "pip install together",
        "models": ["togethercomputer/llama-2-7b", "mistralai/Mixtral-8x7B"],
        "cost": "FREE credits available"
    }
}


class FreeLLMAdapter:
    """
    Adapter for free LLM providers
    Supports: Ollama, HuggingFace, Groq, Together AI
    """
    
    def __init__(self, provider: str = "ollama", model_name: str = "llama2"):
        self.provider = provider
        self.model_name = model_name
        self.client = None
        
        print(f"\n🤖 Initializing Free LLM: {provider}/{model_name}")
        
        if provider == "ollama":
            self._init_ollama()
        elif provider == "huggingface":
            self._init_huggingface()
        elif provider == "groq":
            self._init_groq()
        elif provider == "together":
            self._init_together()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            self.client = ollama
            print(f"✓ Ollama initialized with {self.model_name}")
            print("  Run: ollama pull {self.model_name} (if not already downloaded)")
        except ImportError:
            print("❌ Ollama not installed")
            print("   Install: pip install ollama")
            print("   Download: https://ollama.ai")
            self.client = None
    
    def _init_huggingface(self):
        """Initialize HuggingFace client"""
        try:
            from transformers import pipeline
            print(f"✓ Loading HuggingFace model: {self.model_name}")
            print("  (First run may take time to download)")
            self.client = pipeline("text-generation", model=self.model_name)
            print("✓ Model loaded successfully")
        except ImportError:
            print("❌ Transformers not installed")
            print("   Install: pip install transformers torch")
            self.client = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Tip: Use smaller models like 'google/flan-t5-base'")
            self.client = None
    
    def _init_groq(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("❌ GROQ_API_KEY not found in .env")
                print("   Get free key: https://console.groq.com")
                self.client = None
            else:
                self.client = Groq(api_key=api_key)
                print(f"✓ Groq initialized with {self.model_name}")
        except ImportError:
            print("❌ Groq not installed")
            print("   Install: pip install groq")
            self.client = None
    
    def _init_together(self):
        """Initialize Together AI client"""
        try:
            import together
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                print("❌ TOGETHER_API_KEY not found in .env")
                print("   Get free key: https://api.together.xyz")
                self.client = None
            else:
                together.api_key = api_key
                self.client = together
                print(f"✓ Together AI initialized with {self.model_name}")
        except ImportError:
            print("❌ Together not installed")
            print("   Install: pip install together")
            self.client = None
    
    def invoke(self, prompt: str, max_tokens: int = 500) -> str:
        """Invoke the LLM"""
        if self.client is None:
            return "ERROR: LLM client not initialized"
        
        try:
            if self.provider == "ollama":
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt
                )
                return response['response']
            
            elif self.provider == "huggingface":
                response = self.client(
                    prompt,
                    max_length=max_tokens,
                    num_return_sequences=1
                )
                return response[0]['generated_text']
            
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider == "together":
                response = self.client.Complete.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens
                )
                return response['output']['choices'][0]['text']
            
        except Exception as e:
            return f"ERROR: {e}"


def setup_free_llm_guide():
    """Interactive guide to setup free LLMs"""
    print("\n" + "="*70)
    print("  🆓 FREE LLM SETUP GUIDE")
    print("="*70 + "\n")
    
    print("Choose a free LLM provider:\n")
    
    for i, (name, config) in enumerate(FREE_LLM_CONFIGS.items(), 1):
        print(f"{i}. {name.upper()}")
        print(f"   Description: {config['description']}")
        print(f"   Installation: {config['install']}")
        print(f"   Models: {', '.join(config['models'][:3])}")
        print(f"   Cost: {config['cost']}")
        print()
    
    print("\n" + "="*70)
    print("  RECOMMENDED OPTIONS FOR TRAINING")
    print("="*70 + "\n")
    
    print("🥇 BEST: Ollama (100% Local, No API needed)")
    print("   • Install: https://ollama.ai")
    print("   • Run: ollama pull llama2")
    print("   • Use: python free_llm_training.py --provider ollama")
    print()
    
    print("🥈 GOOD: Groq (Free API, Very Fast)")
    print("   • Sign up: https://console.groq.com")
    print("   • Add GROQ_API_KEY to .env")
    print("   • Use: python free_llm_training.py --provider groq")
    print()
    
    print("🥉 ALTERNATIVE: HuggingFace (Local or API)")
    print("   • Install: pip install transformers torch")
    print("   • Use: python free_llm_training.py --provider huggingface")
    print()
    
    print("\n" + "="*70)
    print("  QUICK START EXAMPLE")
    print("="*70 + "\n")
    
    example_code = """
# Using Ollama (100% free, runs locally)
from free_llm_adapter import FreeLLMAdapter

# Initialize
llm = FreeLLMAdapter(provider="ollama", model_name="llama2")

# Use it
response = llm.invoke("Explain machine learning in simple terms")
print(response)

# Integrate with RAG system
from rag_system import SelfCorrectingRAG

# Replace OpenAI with free LLM
rag = SelfCorrectingRAG()
rag.guardian.llm = llm
rag.generator.llm = llm
rag.evaluator.llm = llm

# Now use RAG for FREE!
result = rag.query("What is AI?")
"""
    
    print(example_code)
    
    print("\n" + "="*70)
    print("  COST COMPARISON")
    print("="*70 + "\n")
    
    print("OpenAI GPT-4:      $0.03-0.06 per 1K tokens  💰💰💰")
    print("OpenAI GPT-3.5:    $0.001-0.002 per 1K tokens 💰")
    print("Groq (Free Tier):  $0.00 (with limits)       🆓")
    print("Ollama (Local):    $0.00 (unlimited)          🆓🆓🆓")
    print("HuggingFace:       $0.00 (local) or API      🆓")
    print()


def create_env_template():
    """Create .env template for free LLMs"""
    template = """
# Free LLM API Keys (Optional)

# Groq (Fast inference) - Get from https://console.groq.com
GROQ_API_KEY=your_groq_key_here

# Together AI - Get from https://api.together.xyz
TOGETHER_API_KEY=your_together_key_here

# HuggingFace (Optional, for API mode) - Get from https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=your_hf_key_here

# Note: Ollama doesn't need an API key - runs 100% locally!
"""
    
    with open(".env.free_llm", "w") as f:
        f.write(template.strip())
    
    print("✓ Created .env.free_llm template")
    print("  Copy relevant keys to your .env file")


if __name__ == "__main__":
    setup_free_llm_guide()
    create_env_template()
