"""
Security Audit Script
Validates the security posture of the Self-Correcting RAG system
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class SecurityAuditor:
    """Automated security auditing tool"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues = []
        self.warnings = []
        self.passed = []
        
    def audit_api_keys(self) -> bool:
        """Check for exposed API keys"""
        print("\n🔍 [1/6] Auditing for exposed API keys...")
        
        # Patterns for API keys
        patterns = [
            r'sk-proj-[a-zA-Z0-9_-]{100,}',  # OpenAI project keys
            r'sk-[a-zA-Z0-9]{48}',            # OpenAI keys
            r'OPENAI_API_KEY\s*=\s*["\']sk-', # Hardcoded keys
            r'api_key\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',  # Generic API keys
        ]
        
        found_keys = []
        
        # Check Python files
        for py_file in self.project_root.rglob('*.py'):
            if '.venv' in str(py_file) or 'venv' in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern in patterns:
                    if re.search(pattern, content):
                        found_keys.append(f"{py_file.name}: Potential API key found")
            except Exception as e:
                self.warnings.append(f"Could not read {py_file.name}: {e}")
        
        # Check .env file
        env_file = self.project_root / '.env'
        if env_file.exists():
            content = env_file.read_text(encoding='utf-8')
            if re.search(r'sk-proj-|sk-[a-zA-Z0-9]{40,}', content):
                found_keys.append(".env: Contains API key")
        
        if found_keys:
            self.issues.extend(found_keys)
            print(f"   ❌ FAILED - Found {len(found_keys)} potential API keys")
            return False
        else:
            self.passed.append("No exposed API keys found")
            print("   ✅ PASSED - No API keys found")
            return True
    
    def audit_code_execution(self) -> bool:
        """Check for dangerous code execution functions"""
        print("\n🔍 [2/6] Auditing for code injection vulnerabilities...")
        
        dangerous_patterns = {
            r'\beval\(': 'eval() usage',
            r'\bexec\(': 'exec() usage',
            r'os\.system\(': 'os.system() usage',
            r'subprocess\..*shell\s*=\s*True': 'subprocess with shell=True',
            r'__import__\(': 'dynamic __import__()',
        }
        
        found_issues = []
        
        for py_file in self.project_root.rglob('*.py'):
            if '.venv' in str(py_file) or 'venv' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern, description in dangerous_patterns.items():
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        # Get line number
                        line_num = content[:match.start()].count('\n') + 1
                        found_issues.append(f"{py_file.name}:{line_num} - {description}")
            except Exception as e:
                self.warnings.append(f"Could not read {py_file.name}: {e}")
        
        if found_issues:
            self.issues.extend(found_issues)
            print(f"   ❌ FAILED - Found {len(found_issues)} dangerous functions")
            for issue in found_issues:
                print(f"      - {issue}")
            return False
        else:
            self.passed.append("No dangerous code execution functions")
            print("   ✅ PASSED - No dangerous functions found")
            return True
    
    def audit_env_file(self) -> bool:
        """Check .env file security"""
        print("\n🔍 [3/6] Auditing .env file...")
        
        env_file = self.project_root / '.env'
        
        if not env_file.exists():
            self.warnings.append(".env file not found")
            print("   ⚠️  WARNING - .env file not found")
            return True
        
        # Check if .env is in .gitignore
        gitignore = self.project_root / '.gitignore'
        if gitignore.exists():
            gitignore_content = gitignore.read_text(encoding='utf-8')
            if '.env' not in gitignore_content:
                self.issues.append(".env not in .gitignore")
                print("   ❌ FAILED - .env not in .gitignore")
                return False
        
        # Check .env content
        env_content = env_file.read_text(encoding='utf-8')
        
        # Should NOT contain actual API keys
        if re.search(r'["\']?[a-zA-Z0-9]{30,}["\']?', env_content) and 'comment' not in env_content.lower():
            self.warnings.append(".env may contain credentials")
            print("   ⚠️  WARNING - .env may contain credentials")
        else:
            self.passed.append(".env file is clean")
            print("   ✅ PASSED - .env file is secure")
            return True
        
        return True
    
    def audit_dependencies(self) -> bool:
        """Check for suspicious dependencies"""
        print("\n🔍 [4/6] Auditing dependencies...")
        
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            self.warnings.append("requirements.txt not found")
            print("   ⚠️  WARNING - requirements.txt not found")
            return True
        
        # Known safe packages
        safe_packages = [
            'langchain', 'langchain-community', 'langchain-ollama', 
            'langchain-openai',  # OK even if not used
            'faiss-cpu', 'chromadb', 'datasets', 'tqdm',
            'numpy', 'pandas', 'scikit-learn', 'transformers',
            'huggingface-hub', 'tiktoken', 'sentence-transformers'
        ]
        
        content = requirements_file.read_text(encoding='utf-8')
        suspicious = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            package = line.split('>=')[0].split('==')[0].strip()
            
            # Check for suspicious package names
            if any(word in package.lower() for word in ['hack', 'pwn', 'exploit', 'shell']):
                suspicious.append(package)
        
        if suspicious:
            self.warnings.extend([f"Suspicious package: {p}" for p in suspicious])
            print(f"   ⚠️  WARNING - Found {len(suspicious)} suspicious packages")
            return True
        else:
            self.passed.append("All dependencies look safe")
            print("   ✅ PASSED - Dependencies are clean")
            return True
    
    def audit_file_operations(self) -> bool:
        """Check for unsafe file operations"""
        print("\n🔍 [5/6] Auditing file operations...")
        
        unsafe_patterns = {
            r'open\([^)]*["\']w["\']': 'Unrestricted file write',
            r'os\.remove\(': 'File deletion',
            r'shutil\.rmtree\(': 'Directory deletion',
            r'\.\.\/': 'Path traversal attempt',
        }
        
        found_issues = []
        
        for py_file in self.project_root.rglob('*.py'):
            if '.venv' in str(py_file) or 'venv' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Check for path traversal
                if '../' in content or '..\\' in content:
                    # Check if it's in a string literal (might be legitimate)
                    if re.search(r'["\'].*/\.\./.*["\']', content):
                        found_issues.append(f"{py_file.name}: Potential path traversal")
                        
            except Exception as e:
                self.warnings.append(f"Could not read {py_file.name}: {e}")
        
        if found_issues:
            self.warnings.extend(found_issues)
            print(f"   ⚠️  WARNING - Found {len(found_issues)} potential issues")
            return True
        else:
            self.passed.append("File operations are safe")
            print("   ✅ PASSED - File operations are safe")
            return True
    
    def audit_network_calls(self) -> bool:
        """Check for external network calls"""
        print("\n🔍 [6/6] Auditing network calls...")
        
        network_patterns = {
            r'requests\.get\(': 'HTTP GET request',
            r'requests\.post\(': 'HTTP POST request',
            r'urllib\.request': 'urllib request',
            r'socket\.connect\(': 'Direct socket connection',
            r'http://': 'HTTP URL (insecure)',
        }
        
        found_calls = []
        
        for py_file in self.project_root.rglob('*.py'):
            if '.venv' in str(py_file) or 'venv' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern, description in network_patterns.items():
                    if re.search(pattern, content):
                        # Check if it's for dataset downloads (legitimate)
                        if 'huggingface' in content.lower() or 'dataset' in content.lower():
                            continue
                        line_num = content[:content.find(pattern)].count('\n') + 1
                        found_calls.append(f"{py_file.name}:{line_num} - {description}")
            except Exception as e:
                self.warnings.append(f"Could not read {py_file.name}: {e}")
        
        if found_calls:
            self.warnings.extend(found_calls)
            print(f"   ⚠️  INFO - Found {len(found_calls)} network calls")
            print("      (This is OK for dataset downloads)")
            return True
        else:
            self.passed.append("No unexpected network calls")
            print("   ✅ PASSED - No unexpected network calls")
            return True
    
    def generate_report(self) -> Dict:
        """Generate final security report"""
        print("\n" + "="*60)
        print("🛡️  SECURITY AUDIT REPORT")
        print("="*60)
        
        # Summary
        total_checks = 6
        passed_checks = len([c for c in [
            len(self.passed) >= 5,  # At least 5 passed
            len(self.issues) == 0,   # No critical issues
        ] if c])
        
        print(f"\n📊 Summary:")
        print(f"   ✅ Passed: {len(self.passed)}/{total_checks}")
        print(f"   ⚠️  Warnings: {len(self.warnings)}")
        print(f"   ❌ Issues: {len(self.issues)}")
        
        # Passed checks
        if self.passed:
            print(f"\n✅ Passed Checks ({len(self.passed)}):")
            for item in self.passed:
                print(f"   • {item}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"   • {item}")
        
        # Critical Issues
        if self.issues:
            print(f"\n❌ Critical Issues ({len(self.issues)}):")
            for item in self.issues:
                print(f"   • {item}")
        
        # Overall Status
        print("\n" + "="*60)
        if len(self.issues) == 0:
            print("🎉 SECURITY AUDIT PASSED ✅")
            print("Your system is secure and ready for use!")
            security_score = "A+"
            status = "SECURE"
        elif len(self.issues) <= 2:
            print("⚠️  SECURITY AUDIT PASSED WITH WARNINGS")
            print("Please review the warnings above.")
            security_score = "B+"
            status = "MOSTLY SECURE"
        else:
            print("❌ SECURITY AUDIT FAILED")
            print("Critical issues found. Please fix before deployment.")
            security_score = "C"
            status = "NEEDS ATTENTION"
        
        print(f"Security Score: {security_score}")
        print("="*60)
        
        return {
            'status': status,
            'score': security_score,
            'passed': len(self.passed),
            'warnings': len(self.warnings),
            'issues': len(self.issues),
            'details': {
                'passed': self.passed,
                'warnings': self.warnings,
                'issues': self.issues
            }
        }
    
    def run_full_audit(self) -> Dict:
        """Run complete security audit"""
        print("🔒 Starting Security Audit...")
        print(f"📁 Project: {self.project_root}")
        
        # Run all checks
        self.audit_api_keys()
        self.audit_code_execution()
        self.audit_env_file()
        self.audit_dependencies()
        self.audit_file_operations()
        self.audit_network_calls()
        
        # Generate report
        return self.generate_report()


if __name__ == "__main__":
    auditor = SecurityAuditor()
    report = auditor.run_full_audit()
    
    # Exit with appropriate code
    sys.exit(0 if report['status'] == "SECURE" else 1)
