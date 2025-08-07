#!/usr/bin/env python3
"""
Test script to verify Kaggle kernel setup
"""

import os
import sys
import json

def test_files_exist():
    """Test that all required files exist."""
    required_files = [
        'main.py',
        'kernel-metadata.json',
        'requirements.txt',
        'config.json',
        'README.md',
        'emotion_classification_notebook.ipynb'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_kernel_metadata():
    """Test kernel metadata is valid JSON."""
    try:
        with open('kernel-metadata.json', 'r') as f:
            metadata = json.load(f)
        
        required_keys = ['id', 'title', 'code_file', 'language', 'kernel_type']
        for key in required_keys:
            if key not in metadata:
                print(f"❌ Missing key in kernel-metadata.json: {key}")
                return False
        
        print("✅ kernel-metadata.json is valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in kernel-metadata.json: {e}")
        return False
    except FileNotFoundError:
        print("❌ kernel-metadata.json not found")
        return False

def test_config_file():
    """Test config file is valid JSON."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        print("✅ config.json is valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config.json: {e}")
        return False
    except FileNotFoundError:
        print("❌ config.json not found")
        return False

def test_main_script():
    """Test main script can be imported."""
    try:
        # Try to compile the main script
        with open('main.py', 'r') as f:
            code = f.read()
        
        compile(code, 'main.py', 'exec')
        print("✅ main.py compiles successfully")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in main.py: {e}")
        return False
    except FileNotFoundError:
        print("❌ main.py not found")
        return False

def main():
    """Run all tests."""
    print("Testing Kaggle kernel setup...")
    print("=" * 40)
    
    tests = [
        test_files_exist,
        test_kernel_metadata,
        test_config_file,
        test_main_script
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! Kernel is ready for Kaggle.")
        print("\nTo push to Kaggle, run:")
        print("kaggle kernels push -p .")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
