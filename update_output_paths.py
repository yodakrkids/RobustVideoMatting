#!/usr/bin/env python3
"""
Script to update all run_all*.bat files to organize outputs by script type
"""

import os
import re

def update_batch_file(filename):
    # Extract the suffix from filename (e.g., "high_gpu" from "run_all_high_gpu.bat")
    suffix = filename.replace("run_all", "").replace(".bat", "").lstrip("_")
    if not suffix:  # For "run_all.bat"
        suffix = "standard"
    
    folder_name = suffix
    print(f"Updating {filename} -> output/{folder_name}/")
    
    # Read the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add directory creation if not exists
    if f'if not exist "output/{folder_name}"' not in content:
        # Find insertion point after the header
        insertion_point = content.find('echo ================================================\n\necho.')
        if insertion_point == -1:
            insertion_point = content.find('echo ================================================') + len('echo ================================================')
        
        if insertion_point > 0:
            before = content[:insertion_point]
            after = content[insertion_point:]
            content = before + f'\necho Results will be saved to: output/{folder_name}/\n\nif not exist "output/{folder_name}" mkdir "output/{folder_name}"' + after
    
    # Update all output paths
    # Replace "output/output_" with "output/{folder_name}/output_"
    content = re.sub(r'"output/output_', f'"output/{folder_name}/output_', content)
    
    # Replace "> output/log_" with "> output/{folder_name}/log_"
    content = re.sub(r'> output/log_', f'> output/{folder_name}/log_', content)
    
    # Replace "output\output_" with "output\{folder_name}\output_"
    content = re.sub(r'"output\\output_', f'"output\\{folder_name}\\output_', content)
    
    # Replace "> "output\log_" with "> "output\{folder_name}\log_"
    content = re.sub(r'> "output\\log_', f'> "output\\{folder_name}\\log_', content)
    
    # For high_quality.bat special handling
    if 'high_quality' in filename:
        content = re.sub(r'"output\\hq_output_', f'"output\\{folder_name}\\hq_output_', content)
        content = re.sub(r'"output\\ultra_output_', f'"output\\{folder_name}\\ultra_output_', content)
        content = re.sub(r'> "output\\hq_log_', f'> "output\\{folder_name}\\hq_log_', content)
        content = re.sub(r'> "output\\ultra_log_', f'> "output\\{folder_name}\\ultra_log_', content)
    
    # Write back
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {filename}")

def main():
    # Find all run_all*.bat files
    batch_files = []
    for file in os.listdir('.'):
        if file.startswith('run_all') and file.endswith('.bat'):
            batch_files.append(file)
    
    print(f"Found {len(batch_files)} batch files:")
    for file in batch_files:
        print(f"  - {file}")
    print()
    
    # Update each file
    for file in batch_files:
        try:
            update_batch_file(file)
        except Exception as e:
            print(f"❌ Error updating {file}: {e}")
    
    print(f"\n🎯 All {len(batch_files)} batch files updated successfully!")
    print("\nNew output structure:")
    for file in batch_files:
        suffix = file.replace("run_all", "").replace(".bat", "").lstrip("_")
        if not suffix:
            suffix = "standard"
        print(f"  {file} → output/{suffix}/")

if __name__ == "__main__":
    main()