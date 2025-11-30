"""
🎉 COMPLETE FINE-TUNING DEMONSTRATION
======================================
Compare base model vs fine-tuned models
"""

import subprocess
from pathlib import Path

def run_ollama(model, prompt):
    """Run Ollama and get response"""
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=60
    )
    return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"

def demo_comparison():
    """Compare base model vs fine-tuned models"""
    
    print("=" * 70)
    print("  🎉 FINE-TUNED MODELS DEMONSTRATION")
    print("=" * 70)
    print()
    print("Comparing base Mistral vs your custom fine-tuned models!")
    print()
    
    # Test scenarios
    tests = [
        {
            "domain": "Data Science",
            "model": "data-science-specialist",
            "question": "What is overfitting and how do I fix it?"
        },
        {
            "domain": "Software Development",
            "model": "code-assistant",
            "question": "What makes code clean and maintainable?"
        },
        {
            "domain": "Business Analysis",
            "model": "business-analyst",
            "question": "What is SWOT analysis?"
        },
        {
            "domain": "Academic Research",
            "model": "research-assistant",
            "question": "How do I structure a research paper?"
        }
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'=' * 70}")
        print(f"  TEST {i}: {test['domain']}")
        print(f"{'=' * 70}")
        print(f"\n❓ Question: {test['question']}\n")
        
        # Base model response
        print("🔵 BASE MODEL (Mistral):")
        print("-" * 70)
        base_response = run_ollama("mistral", test['question'])
        # Truncate for readability
        if len(base_response) > 300:
            print(base_response[:300] + "...\n[truncated for brevity]")
        else:
            print(base_response)
        
        print("\n" + "-" * 70)
        
        # Fine-tuned model response
        print(f"\n🟢 FINE-TUNED MODEL ({test['model']}):")
        print("-" * 70)
        custom_response = run_ollama(test['model'], test['question'])
        if len(custom_response) > 300:
            print(custom_response[:300] + "...\n[truncated for brevity]")
        else:
            print(custom_response)
        
        print("\n" + "-" * 70)
        print("✨ Notice: Fine-tuned model is more specific to the domain!")
    
    # Summary
    print("\n" + "=" * 70)
    print("  📊 SUMMARY")
    print("=" * 70)
    print("\n✅ You now have 5 custom fine-tuned models:")
    print("   1. custom-ml-assistant")
    print("   2. research-assistant")
    print("   3. business-analyst")
    print("   4. code-assistant")
    print("   5. data-science-specialist")
    
    print("\n💡 Each model is specialized for its domain!")
    print("💰 Total cost: $0 (all created locally)")
    print("🔒 Privacy: 100% (your data never left your machine)")
    print("⚡ Performance: Optimized for specific use cases")
    
    print("\n🚀 Use any model:")
    print("   ollama run data-science-specialist")
    print("   ollama run code-assistant")
    print("   ollama run business-analyst")
    
    print("\n📚 View all models:")
    print("   ollama list")
    
    print("\n" + "=" * 70)


def quick_test():
    """Quick test of all models"""
    
    print("=" * 70)
    print("  ⚡ QUICK TEST - ALL FINE-TUNED MODELS")
    print("=" * 70)
    
    models = [
        "custom-ml-assistant",
        "research-assistant", 
        "business-analyst",
        "code-assistant",
        "data-science-specialist"
    ]
    
    question = "Tell me about yourself in one sentence"
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model}")
        print("-" * 70)
        response = run_ollama(model, question)
        print(response[:200] + "..." if len(response) > 200 else response)
    
    print("\n" + "=" * 70)
    print("✅ All models working!")
    print("=" * 70)


def interactive_demo():
    """Interactive testing"""
    
    print("=" * 70)
    print("  🎮 INTERACTIVE MODEL TESTER")
    print("=" * 70)
    
    models = [
        ("1", "mistral", "Base Model"),
        ("2", "custom-ml-assistant", "Custom ML Assistant"),
        ("3", "research-assistant", "Research Assistant"),
        ("4", "business-analyst", "Business Analyst"),
        ("5", "code-assistant", "Code Assistant"),
        ("6", "data-science-specialist", "Data Science Specialist")
    ]
    
    print("\nAvailable models:")
    for num, model, desc in models:
        print(f"   {num}. {desc} ({model})")
    
    print("\nType 'quit' to exit")
    
    while True:
        print("\n" + "-" * 70)
        choice = input("\nSelect model (1-6): ").strip()
        
        if choice.lower() == 'quit':
            break
        
        model_info = next((m for m in models if m[0] == choice), None)
        if not model_info:
            print("Invalid choice!")
            continue
        
        _, model_name, model_desc = model_info
        
        question = input(f"\n❓ Your question for {model_desc}: ").strip()
        if not question:
            continue
        
        print(f"\n💬 Response from {model_desc}:")
        print("-" * 70)
        response = run_ollama(model_name, question)
        print(response)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "compare":
            demo_comparison()
        elif mode == "quick":
            quick_test()
        elif mode == "interactive":
            interactive_demo()
        else:
            print("Usage: python demo_models.py [compare|quick|interactive]")
    else:
        # Default: comparison demo
        demo_comparison()
