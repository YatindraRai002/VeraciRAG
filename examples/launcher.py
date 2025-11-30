"""
🚀 ALL-IN-ONE LAUNCHER
=======================
Your complete AI training & fine-tuning hub
"""

import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("  🚀 AI TRAINING & FINE-TUNING HUB")
    print("=" * 70)
    print()
    print("Everything you need to train, test, and fine-tune AI models")
    print("💰 Cost: $0 | 🔒 Privacy: 100% | ⚡ Performance: Unlimited")
    print()

def print_menu():
    """Print main menu"""
    print("=" * 70)
    print("  MAIN MENU")
    print("=" * 70)
    print()
    print("📚 TRAINING & TESTING:")
    print("  1. Offline Training (Mock LLMs - Instant)")
    print("  2. Simple Ollama RAG (Real LLMs - Local)")
    print("  3. Fine-Tuning Demo (See 4 approaches)")
    print()
    print("🎯 FINE-TUNING:")
    print("  4. Create Custom Model (Interactive)")
    print("  5. Quick Preset Models (6 ready-made)")
    print("  6. Auto-Create All Presets")
    print()
    print("🧪 TESTING:")
    print("  7. Test All Models (Quick)")
    print("  8. Compare Models (Detailed)")
    print("  9. Interactive Model Tester")
    print()
    print("🦙 OLLAMA:")
    print("  10. List All Models")
    print("  11. Chat with Model")
    print()
    print("📖 HELP:")
    print("  12. View Documentation")
    print("  13. System Status")
    print()
    print("  0. Exit")
    print()

def run_script(script_name, *args):
    """Run a Python script"""
    try:
        cmd = [sys.executable, script_name] + list(args)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}")
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_name}")

def run_ollama_command(*args):
    """Run an Ollama command"""
    try:
        subprocess.run(["ollama"] + list(args), check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
    except FileNotFoundError:
        print("\n❌ Ollama not found. Please install Ollama first.")

def list_models():
    """List all Ollama models"""
    print("\n📋 Your Ollama Models:")
    print("=" * 70)
    run_ollama_command("list")

def chat_with_model():
    """Interactive chat with a model"""
    print("\n🦙 Available Models:")
    run_ollama_command("list")
    print()
    model = input("Enter model name (or 'back'): ").strip()
    if model and model.lower() != 'back':
        print(f"\n💬 Starting chat with {model}...")
        print("Type your messages, Ctrl+C to exit")
        run_ollama_command("run", model)

def system_status():
    """Show system status"""
    print("\n" + "=" * 70)
    print("  📊 SYSTEM STATUS")
    print("=" * 70)
    
    # Check Ollama
    print("\n🦙 Ollama Status:")
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"   ✅ Installed: {result.stdout.strip()}")
        else:
            print("   ❌ Not working properly")
    except FileNotFoundError:
        print("   ❌ Not installed")
    
    # Check models
    print("\n📦 Models:")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            model_count = len(lines) - 1  # Exclude header
            print(f"   ✅ {model_count} models available")
        else:
            print("   ⚠️  Could not list models")
    except:
        print("   ❌ Error checking models")
    
    # Check files
    print("\n📁 Training Files:")
    files = [
        "offline_training.py",
        "simple_ollama_rag.py",
        "fine_tune_ollama.py",
        "customize_model.py",
        "create_presets.py",
        "demo_models.py"
    ]
    for file in files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
    
    # Check fine-tuning data
    print("\n💾 Fine-Tuning Data:")
    data_dir = Path("fine_tuning_data")
    if data_dir.exists():
        print(f"   ✅ Directory exists")
        if (data_dir / "training_data.json").exists():
            print(f"   ✅ Training data saved")
        if (data_dir / "Modelfile").exists():
            print(f"   ✅ Modelfile present")
    else:
        print(f"   ⚠️  No data yet (create with option 4-6)")
    
    print("\n" + "=" * 70)
    print("✅ System ready for training and fine-tuning!")
    print("=" * 70)

def view_docs():
    """Show available documentation"""
    print("\n" + "=" * 70)
    print("  📖 DOCUMENTATION")
    print("=" * 70)
    print()
    
    docs = [
        ("COMPLETE_SETUP_SUMMARY.md", "Complete setup overview"),
        ("FREE_TRAINING_GUIDE.md", "All free training options"),
        ("FINE_TUNING_GUIDE.md", "Complete fine-tuning guide"),
        ("OLLAMA_SETUP.md", "Ollama installation & setup"),
        ("QUICK_START.md", "5-minute quick start"),
        ("ARCHITECTURE_GUIDE.md", "Technical documentation")
    ]
    
    print("Available guides:")
    for i, (filename, desc) in enumerate(docs, 1):
        status = "✅" if Path(filename).exists() else "❌"
        print(f"  {i}. {status} {filename}")
        print(f"     {desc}")
        print()
    
    print("💡 Open any .md file in VS Code or a text editor")
    print("=" * 70)

def main():
    """Main launcher loop"""
    print_banner()
    
    while True:
        print_menu()
        choice = input("Select option (0-13): ").strip()
        print()
        
        if choice == "0":
            print("👋 Goodbye! Happy training!")
            break
        
        elif choice == "1":
            print("🎮 Running Offline Training...")
            run_script("offline_training.py")
        
        elif choice == "2":
            print("🦙 Running Simple Ollama RAG...")
            run_script("simple_ollama_rag.py")
        
        elif choice == "3":
            print("🎯 Running Fine-Tuning Demo...")
            run_script("fine_tune_ollama.py")
        
        elif choice == "4":
            print("🎨 Interactive Model Customization...")
            run_script("customize_model.py")
        
        elif choice == "5":
            print("⚡ Quick Preset Models...")
            run_script("customize_model.py")
        
        elif choice == "6":
            print("🚀 Auto-Creating All Presets...")
            run_script("create_presets.py")
        
        elif choice == "7":
            print("🧪 Quick Testing All Models...")
            run_script("demo_models.py", "quick")
        
        elif choice == "8":
            print("📊 Detailed Model Comparison...")
            run_script("demo_models.py", "compare")
        
        elif choice == "9":
            print("🎮 Interactive Model Tester...")
            run_script("demo_models.py", "interactive")
        
        elif choice == "10":
            list_models()
        
        elif choice == "11":
            chat_with_model()
        
        elif choice == "12":
            view_docs()
        
        elif choice == "13":
            system_status()
        
        else:
            print("❌ Invalid option. Please choose 0-13.")
        
        if choice != "0":
            input("\n⏎ Press Enter to continue...")
            print("\n" * 2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Please report this issue!")
