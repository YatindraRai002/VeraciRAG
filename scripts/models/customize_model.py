"""
🎨 INTERACTIVE MODEL CUSTOMIZATION
===================================
Customize Ollama for your specific use case
"""

import json
from pathlib import Path
from fine_tune_ollama import OllamaFineTuner

def interactive_fine_tuning():
    """Interactive session to fine-tune Ollama"""
    
    print("=" * 70)
    print("  🎨 INTERACTIVE MODEL CUSTOMIZATION")
    print("=" * 70)
    print()
    print("Customize Ollama Mistral for YOUR specific domain!")
    print("This will create a specialized version using:")
    print("  • Your training examples (few-shot learning)")
    print("  • Your domain documents (RAG)")
    print("  • Combined approach for best results")
    print()
    print("💰 Cost: $0 | 🔒 100% Local | ⚡ Unlimited Usage")
    print()
    
    # Initialize
    model_name = input("📝 Enter custom model name [my-custom-model]: ").strip()
    if not model_name:
        model_name = "my-custom-model"
    
    tuner = OllamaFineTuner(base_model="mistral", custom_name=model_name)
    
    # Step 1: Add training examples
    print("\n" + "=" * 70)
    print("  STEP 1: ADD TRAINING EXAMPLES")
    print("=" * 70)
    print("\nTraining examples teach the model your preferred response style.")
    print("Format: Question → Desired Answer")
    print()
    
    examples = []
    while True:
        print(f"\n[Example {len(examples) + 1}]")
        question = input("❓ Question (or 'done' to finish): ").strip()
        if question.lower() == 'done':
            break
        if not question:
            continue
        
        answer = input("💡 Desired Answer: ").strip()
        if not answer:
            continue
        
        examples.append({"input": question, "output": answer})
        print(f"✓ Added example {len(examples)}")
    
    if examples:
        tuner.add_training_examples(examples)
    else:
        print("⚠️  No examples added. Using demo examples...")
        demo_examples = [
            {
                "input": "What is your purpose?",
                "output": f"I am {model_name}, a specialized AI assistant fine-tuned for your specific needs."
            },
            {
                "input": "How can you help me?",
                "output": "I can assist with domain-specific questions using my specialized knowledge base."
            }
        ]
        tuner.add_training_examples(demo_examples)
    
    # Step 2: Add domain documents
    print("\n" + "=" * 70)
    print("  STEP 2: ADD DOMAIN DOCUMENTS")
    print("=" * 70)
    print("\nDomain documents provide factual knowledge for the model.")
    print("Add as many as you want - they'll be searchable via RAG.")
    print()
    
    documents = []
    while True:
        doc = input(f"\n📄 Document {len(documents) + 1} (or 'done' to finish):\n").strip()
        if doc.lower() == 'done':
            break
        if not doc:
            continue
        
        documents.append(doc)
        print(f"✓ Added document {len(documents)}")
    
    if documents:
        tuner.add_domain_documents(documents)
    else:
        print("⚠️  No documents added. You can add them later.")
    
    # Step 3: Test the customization
    print("\n" + "=" * 70)
    print("  STEP 3: TEST YOUR CUSTOMIZATION")
    print("=" * 70)
    print()
    
    test = input("🧪 Want to test your customization? (y/n) [y]: ").strip().lower()
    if test != 'n':
        print("\nEnter test questions (type 'done' when finished)")
        test_queries = []
        
        while True:
            query = input(f"\n❓ Test Question {len(test_queries) + 1}: ").strip()
            if query.lower() == 'done':
                break
            if query:
                test_queries.append(query)
        
        if test_queries:
            print("\n🔄 Processing test queries...")
            for i, query in enumerate(test_queries, 1):
                print(f"\n[{i}/{len(test_queries)}] {query}")
                print("-" * 70)
                
                # Show different approaches
                if examples:
                    print("🟢 Few-Shot Response:")
                    response = tuner.few_shot_query(query)
                    print(response[:300] + "..." if len(response) > 300 else response)
                    print()
                
                if documents:
                    print("🟡 RAG Response:")
                    response = tuner.rag_query(query)
                    print(response[:300] + "..." if len(response) > 300 else response)
                    print()
                
                if examples or documents:
                    print("🟣 Combined Response (BEST):")
                    response = tuner.combined_query(query)
                    print(response[:300] + "..." if len(response) > 300 else response)
    
    # Step 4: Save everything
    print("\n" + "=" * 70)
    print("  STEP 4: SAVE CUSTOMIZATION")
    print("=" * 70)
    print()
    
    tuner.save_training_data()
    modelfile_path = tuner.create_modelfile()
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ CUSTOMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   • Model Name: {model_name}")
    print(f"   • Training Examples: {len(examples)}")
    print(f"   • Domain Documents: {len(documents)}")
    print(f"   • Saved To: {tuner.data_dir}")
    
    print(f"\n🎯 Next Steps:")
    print(f"\n1️⃣  CREATE CUSTOM OLLAMA MODEL (Optional):")
    print(f"   cd {tuner.data_dir}")
    print(f"   ollama create {model_name} -f Modelfile")
    print(f"   ollama run {model_name}")
    
    print(f"\n2️⃣  USE IN PYTHON (Recommended):")
    print(f"""
from fine_tune_ollama import OllamaFineTuner

tuner = OllamaFineTuner(custom_name="{model_name}")
tuner.load_training_data()

# Query with your customization
response = tuner.combined_query("Your question here")
print(response)
""")
    
    print(f"\n3️⃣  ADD MORE DATA ANYTIME:")
    print(f"   • Edit: {tuner.data_dir / 'training_data.json'}")
    print(f"   • Run this script again")
    
    print(f"\n💡 Your customization is saved and reusable!")
    print(f"💰 Cost: $0 | 🔒 Privacy: 100% | ⚡ Performance: Optimized!")
    print("=" * 70)


def quick_customize():
    """Quick customization with preset options"""
    
    print("=" * 70)
    print("  ⚡ QUICK MODEL CUSTOMIZATION")
    print("=" * 70)
    print()
    print("Choose a preset customization:")
    print()
    print("1. 📚 Academic Research Assistant")
    print("2. 💼 Business Analyst")
    print("3. 💻 Software Developer Helper")
    print("4. 🎓 Student Tutor")
    print("5. 🎨 Creative Writing Assistant")
    print("6. 🔬 Data Science Specialist")
    print("7. 🌐 Custom (I'll provide my own data)")
    print()
    
    choice = input("Select preset (1-7) [7]: ").strip()
    
    presets = {
        "1": {
            "name": "research-assistant",
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
        "2": {
            "name": "business-analyst",
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
        "3": {
            "name": "code-assistant",
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
        "6": {
            "name": "data-science-specialist",
            "examples": [
                {"input": "Choose ML algorithm", "output": "I'll recommend algorithms based on your data type, problem, and constraints."},
                {"input": "Explain model metrics", "output": "I'll clarify accuracy, precision, recall, F1-score, and when to use each."},
                {"input": "Prevent overfitting", "output": "I'll suggest regularization, cross-validation, and data augmentation techniques."}
            ],
            "docs": [
                "Feature engineering creates informative variables from raw data for better model performance.",
                "Cross-validation splits data into folds to assess model generalization reliably.",
                "Regularization techniques (L1, L2, dropout) prevent overfitting in models.",
                "Hyperparameter tuning optimizes model configuration for best performance.",
                "Ensemble methods combine multiple models to improve prediction accuracy."
            ]
        }
    }
    
    if choice in presets:
        preset = presets[choice]
        tuner = OllamaFineTuner(base_model="mistral", custom_name=preset["name"])
        tuner.add_training_examples(preset["examples"])
        tuner.add_domain_documents(preset["docs"])
        tuner.save_training_data()
        tuner.create_modelfile()
        
        print(f"\n✅ Created customization: {preset['name']}")
        print(f"💾 Saved to: {tuner.data_dir}")
        print(f"\n🚀 To use:")
        print(f"   ollama create {preset['name']} -f {tuner.data_dir}/Modelfile")
        print(f"   ollama run {preset['name']}")
    else:
        interactive_fine_tuning()


if __name__ == "__main__":
    print("Choose mode:")
    print("1. ⚡ Quick Preset Customization")
    print("2. 🎨 Interactive Custom Tuning")
    print()
    
    mode = input("Select (1 or 2) [1]: ").strip()
    
    if mode == "2":
        interactive_fine_tuning()
    else:
        quick_customize()
