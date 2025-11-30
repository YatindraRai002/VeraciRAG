"""
🚀 AUTO-CREATE PRESET MODELS
==============================
Automatically create all preset fine-tuned models
"""

from fine_tune_ollama import OllamaFineTuner
import subprocess
import sys

def create_all_presets():
    """Create all preset models automatically"""
    
    presets = {
        "research-assistant": {
            "examples": [
                {"input": "Summarize this paper", "output": "I'll provide a structured summary with: Background, Methods, Results, and Conclusions."},
                {"input": "Find related work", "output": "I'll identify relevant papers in the field and explain their connections."},
                {"input": "Critique methodology", "output": "I'll analyze the research methods, highlighting strengths and potential limitations."}
            ],
            "docs": [
                "Academic research follows the scientific method: hypothesis, experiment, analysis, conclusion.",
                "Peer review ensures research quality through expert evaluation.",
                "Citations are crucial for attributing ideas and building on previous work.",
                "Methodology describes how research was conducted for reproducibility.",
                "Results should be presented objectively with appropriate statistical analysis."
            ]
        },
        "business-analyst": {
            "examples": [
                {"input": "Analyze market trends", "output": "I'll examine current trends, growth patterns, and competitive landscape."},
                {"input": "Create SWOT analysis", "output": "I'll identify Strengths, Weaknesses, Opportunities, and Threats systematically."},
                {"input": "Forecast revenue", "output": "I'll use historical data and market indicators to project future performance."}
            ],
            "docs": [
                "SWOT analysis evaluates Strengths, Weaknesses, Opportunities, and Threats.",
                "KPIs (Key Performance Indicators) measure business success metrics.",
                "Market segmentation divides customers into distinct groups for targeting.",
                "Competitive analysis assesses rivals' strengths and market position.",
                "ROI (Return on Investment) measures profitability of investments."
            ]
        },
        "code-assistant": {
            "examples": [
                {"input": "Explain this code", "output": "I'll break down the code's functionality, logic, and best practices used."},
                {"input": "Debug this error", "output": "I'll identify the root cause and suggest specific fixes with explanations."},
                {"input": "Optimize performance", "output": "I'll suggest improvements for efficiency, readability, and maintainability."}
            ],
            "docs": [
                "Clean code is readable, maintainable, and follows consistent style guidelines.",
                "DRY (Don't Repeat Yourself) principle reduces code duplication.",
                "SOLID principles guide object-oriented design for better software.",
                "Unit tests verify individual components work correctly in isolation.",
                "Code reviews improve quality through peer feedback and knowledge sharing."
            ]
        },
        "data-science-specialist": {
            "examples": [
                {"input": "Choose ML algorithm", "output": "I'll recommend algorithms based on your data type, problem, and constraints."},
                {"input": "Explain model metrics", "output": "I'll clarify accuracy, precision, recall, F1-score, and when to use each."},
                {"input": "Prevent overfitting", "output": "I'll suggest regularization, cross-validation, and data augmentation techniques."}
            ],
            "docs": [
                "Feature engineering creates informative variables from raw data for better model performance.",
                "Cross-validation splits data into multiple folds to assess model generalization reliably.",
                "Regularization techniques (L1, L2, dropout) prevent overfitting in models.",
                "Hyperparameter tuning optimizes model configuration for best performance.",
                "Ensemble methods combine multiple models to improve prediction accuracy."
            ]
        }
    }
    
    print("=" * 70)
    print("  🚀 AUTO-CREATING PRESET MODELS")
    print("=" * 70)
    print(f"\nCreating {len(presets)} fine-tuned models...")
    print("This will take a few minutes.\n")
    
    created_models = []
    
    for i, (name, config) in enumerate(presets.items(), 1):
        print(f"\n[{i}/{len(presets)}] Creating: {name}")
        print("-" * 70)
        
        try:
            # Create tuner
            tuner = OllamaFineTuner(base_model="mistral", custom_name=name)
            
            # Add data
            tuner.add_training_examples(config["examples"])
            tuner.add_domain_documents(config["docs"])
            
            # Save
            tuner.save_training_data()
            modelfile_path = tuner.create_modelfile()
            
            # Create Ollama model
            print(f"\n🔨 Building Ollama model: {name}")
            result = subprocess.run(
                ["ollama", "create", name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully created: {name}")
                created_models.append(name)
            else:
                print(f"⚠️  Error creating {name}: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ PRESET CREATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Successfully created {len(created_models)}/{len(presets)} models")
    
    if created_models:
        print("\n🎯 Available models:")
        for model in created_models:
            print(f"   • {model}")
        
        print("\n🚀 Test any model:")
        print(f"   ollama run {created_models[0]}")
        
        print("\n📚 List all your models:")
        print("   ollama list")
    
    print("\n💰 Cost: $0 | 🔒 Privacy: 100% | ⚡ Ready to use!")
    print("=" * 70)


def test_model(model_name):
    """Test a specific model"""
    print(f"\n🧪 Testing: {model_name}")
    print("-" * 70)
    
    test_questions = {
        "research-assistant": "How do I write a good research paper?",
        "business-analyst": "What is SWOT analysis?",
        "code-assistant": "What makes code clean?",
        "data-science-specialist": "How do I prevent overfitting?"
    }
    
    question = test_questions.get(model_name, "Tell me about yourself")
    
    print(f"❓ Question: {question}\n")
    print("💬 Response:")
    
    result = subprocess.run(
        ["ollama", "run", model_name, question],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"❌ Error: {result.stderr}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        model = sys.argv[2] if len(sys.argv) > 2 else "data-science-specialist"
        test_model(model)
    else:
        # Create all presets
        create_all_presets()
