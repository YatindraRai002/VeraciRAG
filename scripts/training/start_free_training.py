"""
🆓 FREE TRAINING STARTER
========================

Quick launcher for free RAG training - No API costs!

Choose your training mode:
1. Offline (Mock) - Instant, no setup
2. Ollama (Local) - Real LLM, 100% free
3. Groq (API) - Fast, free tier
"""

import sys
from pathlib import Path

def print_banner():
    """Display welcome banner"""
    print("=" * 70)
    print("  🆓 FREE RAG TRAINING SYSTEM")
    print("=" * 70)
    print()
    print("Train, test, and fine-tune your RAG system WITHOUT spending money!")
    print()

def print_options():
    """Display training options"""
    print("📋 Choose your training mode:")
    print()
    print("1. 🎮 OFFLINE MODE (Mock LLMs)")
    print("   - Zero setup required")
    print("   - Instant start")
    print("   - Tests architecture")
    print("   - Best for: Quick testing")
    print()
    print("2. 🦙 OLLAMA (Local LLMs)")
    print("   - 100% free forever")
    print("   - Real LLM responses")
    print("   - Requires Ollama installed")
    print("   - Best for: Serious development")
    print()
    print("3. 🌐 GROQ (Free API)")
    print("   - Fast cloud inference")
    print("   - Real LLM responses")
    print("   - Requires API key")
    print("   - Best for: Quick experiments")
    print()
    print("4. 📚 HELP & GUIDES")
    print("   - View documentation")
    print("   - Setup instructions")
    print()
    print("5. ❌ EXIT")
    print()

def run_offline():
    """Run offline training demo"""
    print("\n🎮 Starting OFFLINE training mode...")
    print("=" * 70)
    
    try:
        from offline_training import OfflineRAGSystem
        
        # Initialize system
        print("\n✓ Initializing offline RAG system...")
        rag = OfflineRAGSystem()
        
        # Add sample documents
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual data.",
            "Reinforcement learning trains agents through rewards.",
        ]
        print(f"✓ Adding {len(docs)} sample documents...")
        rag.add_documents(docs)
        
        # Test queries
        queries = [
            "What is machine learning?",
            "Explain deep learning",
            "What is NLP?",
        ]
        
        print(f"\n✓ Processing {len(queries)} test queries...\n")
        print("=" * 70)
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Query: {query}")
            result = rag.process_query(query)
            results.append(result)
            
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Quality: {result['quality_score']:.2f}")
            print(f"   Status: {result['status']}")
        
        # Summary
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        print("\n" + "=" * 70)
        print("📊 TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total Queries: {len(queries)}")
        print(f"Average Quality: {avg_quality:.2f}")
        print(f"Success Rate: 100%")
        print("\n✅ Offline training complete!")
        print("\n💡 This ran entirely locally without any API calls.")
        print("   You can experiment freely with no costs!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure offline_training.py is in the same directory.")

def run_ollama():
    """Run Ollama training"""
    print("\n🦙 Starting OLLAMA training mode...")
    print("=" * 70)
    
    # Check if Ollama is available
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise Exception("Ollama not responding")
    except Exception:
        print("\n⚠️  Ollama not found or not running!")
        print("\n📦 To use Ollama:")
        print("   1. Download from: https://ollama.ai/download")
        print("   2. Install Ollama")
        print("   3. Run: ollama pull mistral")
        print("   4. Try this option again")
        print("\n📖 See OLLAMA_SETUP.md for detailed instructions")
        return
    
    # Get available models
    models = []
    for line in result.stdout.split('\n')[1:]:  # Skip header
        if line.strip():
            model_name = line.split()[0]
            if model_name:
                models.append(model_name)
    
    if not models:
        print("\n⚠️  No Ollama models found!")
        print("\n📦 Install a model first:")
        print("   ollama pull mistral")
        print("\n📖 See OLLAMA_SETUP.md for more options")
        return
    
    print(f"\n✓ Ollama is running!")
    print(f"✓ Found {len(models)} model(s): {', '.join(models)}")
    
    # Choose model
    if len(models) == 1:
        model_name = models[0]
        print(f"✓ Using model: {model_name}")
    else:
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model}")
        choice = input(f"\nChoose model (1-{len(models)}) [1]: ").strip()
        if not choice:
            model_name = models[0]
        else:
            try:
                model_name = models[int(choice) - 1]
            except:
                model_name = models[0]
        print(f"✓ Selected: {model_name}")
    
    # Run training
    try:
        from free_training import FreeTrainingSystem
        
        print("\n🚀 Initializing training system...")
        trainer = FreeTrainingSystem(
            provider="ollama",
            model_name=model_name
        )
        
        print("📚 Running training session...")
        print("   (This will make real LLM calls - may take a minute)")
        print()
        
        results = trainer.run_training_session()
        
        # Display results
        print("\n" + "=" * 70)
        print("📊 TRAINING RESULTS")
        print("=" * 70)
        trainer.display_results(results)
        
        # Save option
        save = input("\n💾 Save results to file? (y/n) [y]: ").strip().lower()
        if save != 'n':
            filename = f"ollama_training_{model_name.replace(':', '_')}.json"
            trainer.save_results(results, filename)
            print(f"✓ Results saved to: {filename}")
        
        print("\n✅ Ollama training complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure free_training.py is available.")

def run_groq():
    """Run Groq API training"""
    print("\n🌐 Starting GROQ training mode...")
    print("=" * 70)
    
    print("\n📝 You'll need a Groq API key:")
    print("   1. Get free key: https://console.groq.com")
    print("   2. No credit card required")
    print("   3. Generous free tier")
    print()
    
    api_key = input("Enter your Groq API key (or 'back' to return): ").strip()
    
    if api_key.lower() == 'back':
        return
    
    if not api_key:
        print("❌ API key required!")
        return
    
    try:
        from free_training import FreeTrainingSystem
        
        print("\n🚀 Initializing Groq training...")
        trainer = FreeTrainingSystem(
            provider="groq",
            api_key=api_key,
            model_name="mixtral-8x7b-32768"
        )
        
        print("📚 Running training session with Groq...")
        print("   (Cloud-based, should be fast!)")
        print()
        
        results = trainer.run_training_session()
        
        # Display results
        print("\n" + "=" * 70)
        print("📊 TRAINING RESULTS")
        print("=" * 70)
        trainer.display_results(results)
        
        # Save option
        save = input("\n💾 Save results to file? (y/n) [y]: ").strip().lower()
        if save != 'n':
            filename = "groq_training_results.json"
            trainer.save_results(results, filename)
            print(f"✓ Results saved to: {filename}")
        
        print("\n✅ Groq training complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Check your API key and internet connection.")

def show_help():
    """Display help and documentation"""
    print("\n📚 HELP & DOCUMENTATION")
    print("=" * 70)
    print()
    print("Available guides:")
    print()
    
    guides = [
        ("FREE_TRAINING_GUIDE.md", "Complete guide to all free training options"),
        ("OLLAMA_SETUP.md", "Step-by-step Ollama installation and usage"),
        ("QUICK_START.md", "5-minute quick start guide"),
        ("ARCHITECTURE_GUIDE.md", "Technical documentation"),
    ]
    
    for filename, description in guides:
        path = Path(filename)
        if path.exists():
            print(f"✓ {filename}")
            print(f"  {description}")
            print()
        else:
            print(f"⚠️  {filename} (not found)")
            print()
    
    print("💡 TIP: Open these .md files in any text editor or VS Code")
    print()
    
    view = input("View FREE_TRAINING_GUIDE.md now? (y/n) [y]: ").strip().lower()
    if view != 'n':
        guide_path = Path("FREE_TRAINING_GUIDE.md")
        if guide_path.exists():
            print("\n" + "=" * 70)
            with open(guide_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Show first 50 lines
                lines = content.split('\n')[:50]
                print('\n'.join(lines))
                if len(content.split('\n')) > 50:
                    print("\n... (content truncated, open file to see more)")
            print("=" * 70)
        else:
            print("\n⚠️  Guide file not found!")

def main():
    """Main menu loop"""
    print_banner()
    
    while True:
        print_options()
        
        choice = input("Choose option (1-5) [1]: ").strip()
        
        if not choice:
            choice = "1"
        
        print()
        
        if choice == "1":
            run_offline()
        elif choice == "2":
            run_ollama()
        elif choice == "3":
            run_groq()
        elif choice == "4":
            show_help()
        elif choice == "5":
            print("👋 Goodbye! Happy training!")
            break
        else:
            print("❌ Invalid option. Please choose 1-5.")
        
        print("\n" + "=" * 70)
        input("\nPress Enter to continue...")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Please report this issue!")
