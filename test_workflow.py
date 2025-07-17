#!/usr/bin/env python3
"""
Simple test script to verify the basic workflow of the text-to-audiobook system.
"""

import os
import json
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, '/app')

from src.text_processing.text_extractor import TextExtractor
from src.text_structurer import TextStructurer

def test_basic_workflow():
    """Test the basic workflow: text extraction -> text structuring -> JSON output"""
    print("🧪 Testing basic text-to-audiobook workflow...")
    
    # Test 1: Text Extraction
    print("\n📖 Step 1: Testing text extraction...")
    extractor = TextExtractor()
    
    # Test with sample.txt
    sample_path = "/app/input/sample.txt"
    if os.path.exists(sample_path):
        try:
            text = extractor.extract(sample_path)
            print(f"✅ Successfully extracted {len(text)} characters from sample.txt")
            print(f"   Preview: {text[:100]}...")
        except Exception as e:
            print(f"❌ Failed to extract text from sample.txt: {e}")
            return False
    else:
        print(f"❌ Sample file not found at {sample_path}")
        return False
    
    # Test 2: Text Structuring
    print("\n🔧 Step 2: Testing text structuring...")
    try:
        structurer = TextStructurer(engine='local')
        print("✅ TextStructurer initialized successfully")
        
        # Note: This will fail without LLM, but we test the initialization
        print("   (Note: Full structuring requires LLM connection)")
        
    except Exception as e:
        print(f"❌ Failed to initialize TextStructurer: {e}")
        return False
    
    # Test 3: Output Directory
    print("\n📁 Step 3: Testing output directory...")
    output_dir = Path("/app/output")
    output_dir.mkdir(exist_ok=True)
    
    if output_dir.exists():
        print("✅ Output directory exists")
        
        # Check existing output files
        json_files = list(output_dir.glob("*.json"))
        if json_files:
            print(f"   Found {len(json_files)} existing JSON files:")
            for json_file in json_files:
                print(f"     - {json_file.name}")
                
                # Validate JSON structure
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list) and len(data) > 0:
                        first_item = data[0]
                        if 'speaker' in first_item and 'text' in first_item:
                            print(f"       ✅ Valid structure: {len(data)} segments")
                        else:
                            print(f"       ⚠️  Invalid structure: missing speaker/text keys")
                    else:
                        print(f"       ⚠️  Invalid structure: not a list or empty")
                        
                except Exception as e:
                    print(f"       ❌ Error reading JSON: {e}")
        else:
            print("   No existing JSON files found")
    else:
        print("❌ Output directory does not exist")
        return False
    
    # Test 4: Input Files
    print("\n📥 Step 4: Testing input files...")
    input_dir = Path("/app/input")
    if input_dir.exists():
        input_files = list(input_dir.glob("*"))
        print(f"✅ Found {len(input_files)} input files:")
        for input_file in input_files:
            print(f"   - {input_file.name} ({input_file.stat().st_size} bytes)")
    else:
        print("❌ Input directory does not exist")
        return False
    
    print("\n🎉 Basic workflow test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_basic_workflow()
    sys.exit(0 if success else 1)