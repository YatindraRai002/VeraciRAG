"""
Complete Setup Verification Script
This checks everything that was built and provides status
"""

import os
import sys
import json
from pathlib import Path

class SetupVerifier:
    def __init__(self):
        # Get project root (2 levels up from scripts/utils/)
        self.base_dir = Path(__file__).parent.parent.parent
        self.status = {
            'files': {},
            'ollama': {},
            'models': {},
            'overall': True
        }
    
    def check_file(self, filename, description):
        """Check if a file exists"""
        filepath = self.base_dir / filename
        exists = filepath.exists()
        self.status['files'][filename] = {
            'exists': exists,
            'description': description,
            'size': filepath.stat().st_size if exists else 0
        }
        return exists
    
    def check_ollama(self):
        """Check Ollama installation"""
        print("\n🔍 Checking Ollama Installation...")
        
        # Try to run ollama
        import subprocess
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                self.status['ollama']['installed'] = True
                self.status['ollama']['version'] = result.stdout.strip()
                print(f"   ✅ Ollama installed: {result.stdout.strip()}")
                return True
            else:
                self.status['ollama']['installed'] = False
                print(f"   ❌ Ollama not responding")
                return False
        except FileNotFoundError:
            self.status['ollama']['installed'] = False
            print(f"   ❌ Ollama not found in PATH")
            print(f"   💡 Run: winget install Ollama.Ollama")
            return False
        except Exception as e:
            self.status['ollama']['installed'] = False
            print(f"   ❌ Error checking Ollama: {e}")
            return False
    
    def check_models(self):
        """Check available Ollama models"""
        if not self.status['ollama'].get('installed'):
            print("\n⚠️  Cannot check models - Ollama not installed")
            return
        
        print("\n🤖 Checking Custom Models...")
        import subprocess
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                models = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                
                self.status['models']['list'] = models
                self.status['models']['count'] = len(models)
                
                print(f"   Found {len(models)} models:")
                for model in models:
                    print(f"   ✅ {model}")
            else:
                print(f"   ❌ Could not list models")
        except Exception as e:
            print(f"   ❌ Error listing models: {e}")
    
    def check_core_files(self):
        """Check all core project files"""
        print("\n📁 Checking Core Files...")
        
        core_files = {
            # Root files
            'config.py': 'Configuration',
            'README.md': 'Main README',
            'PROJECT_STRUCTURE.md': 'Structure guide',
            
            # Examples
            'examples/launcher.py': 'All-in-one menu interface',
            'examples/simple_ollama_rag.py': 'Basic Ollama RAG',
            'examples/rag_system.py': 'Self-correcting RAG',
            
            # Training scripts
            'scripts/training/local_training.py': 'Main training (recommended)',
            'scripts/training/offline_training.py': 'Offline training',
            'scripts/training/free_training.py': 'Free training framework',
            
            # Model scripts
            'scripts/models/fine_tune_ollama.py': 'Core fine-tuning system',
            'scripts/models/create_presets.py': 'Preset creator',
            'scripts/models/demo_models.py': 'Model demo tool',
            
            # Utils
            'scripts/utils/verify_setup.py': 'This verification script',
            'scripts/utils/security_audit.py': 'Security audit',
            
            # Documentation
            'docs/README.md': 'Main documentation',
            'docs/SECURITY.md': 'Security guide',
            'docs/TRAINING_REPORT.md': 'Training report',
        }
        
        for filename, description in core_files.items():
            exists = self.check_file(filename, description)
            status = "✅" if exists else "❌"
            print(f"   {status} {filename}: {description}")
    
    def check_directories(self):
        """Check required directories"""
        print("\n📂 Checking Directories...")
        
        dirs = {
            'agents': 'RAG agents',
            'core': 'Core components',
            'retrieval': 'Document retrieval',
            'training': 'Training utilities',
            'validation': 'Testing suite',
            'scripts': 'Executable scripts',
            'scripts/training': 'Training scripts',
            'scripts/models': 'Model scripts',
            'scripts/utils': 'Utility scripts',
            'examples': 'Example apps',
            'docs': 'Documentation',
            'data': 'Data files',
            'tests': 'Test suite',
        }
        
        for dirname, description in dirs.items():
            dirpath = self.base_dir / dirname
            exists = dirpath.exists()
            status = "✅" if exists else "❌"
            
            if exists:
                file_count = len(list(dirpath.glob('*.py')))
                print(f"   {status} {dirname}/ - {description} ({file_count} .py files)")
            else:
                print(f"   {status} {dirname}/ - {description} (missing)")
    
    def generate_report(self):
        """Generate final status report"""
        print("\n" + "="*60)
        print("📊 SETUP VERIFICATION REPORT")
        print("="*60)
        
        # Count files
        total_files = len(self.status['files'])
        existing_files = sum(1 for f in self.status['files'].values() if f['exists'])
        
        print(f"\n✅ Files: {existing_files}/{total_files} exist")
        
        # Ollama status
        if self.status['ollama'].get('installed'):
            print(f"✅ Ollama: Installed ({self.status['ollama'].get('version', 'unknown')})")
        else:
            print(f"❌ Ollama: Not installed or not in PATH")
        
        # Models status
        model_count = self.status['models'].get('count', 0)
        if model_count > 0:
            print(f"✅ Models: {model_count} custom models created")
        else:
            print(f"⚠️  Models: No models found (need to create them)")
        
        # Overall status
        if self.status['ollama'].get('installed') and model_count >= 5:
            print("\n🎉 EVERYTHING IS COMPLETE AND READY!")
            print("   Run: python launcher.py")
        elif existing_files == total_files:
            print("\n⚠️  ALL FILES EXIST - Need to setup Ollama")
            print("   Next step: Install Ollama and create models")
        else:
            print("\n⚠️  SETUP INCOMPLETE - Missing some files")
        
        print("="*60)
    
    def save_report(self):
        """Save status to JSON"""
        report_path = self.base_dir / 'setup_status.json'
        with open(report_path, 'w') as f:
            json.dump(self.status, f, indent=2)
        print(f"\n💾 Detailed report saved to: setup_status.json")
    
    def run_full_check(self):
        """Run all checks"""
        print("🔍 VERIFYING COMPLETE SETUP...")
        print("="*60)
        
        self.check_core_files()
        self.check_directories()
        self.check_ollama()
        self.check_models()
        self.generate_report()
        self.save_report()
        
        return self.status

def main():
    verifier = SetupVerifier()
    status = verifier.run_full_check()
    
    # Provide next steps
    print("\n📋 NEXT STEPS:")
    
    if not status['ollama'].get('installed'):
        print("\n1. Install Ollama:")
        print("   winget install Ollama.Ollama")
        print("\n2. Restart terminal, then run:")
        print("   ollama pull mistral")
        print("\n3. Create custom models:")
        print("   python create_presets.py")
    elif status['models'].get('count', 0) == 0:
        print("\n1. Download base model:")
        print("   ollama pull mistral")
        print("\n2. Create custom models:")
        print("   python create_presets.py")
    else:
        print("\n✅ Everything ready! Start using:")
        print("   python launcher.py")
        print("\nOr test a model:")
        print("   ollama run data-science-specialist")

if __name__ == "__main__":
    main()
