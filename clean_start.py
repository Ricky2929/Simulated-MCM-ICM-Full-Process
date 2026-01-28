# clean_start.py
# 彻底清理项目，准备重新开始

import os
import shutil

print("="*80)
print("项目彻底清理 - 准备重新开始")
print("Complete Project Cleanup - Fresh Start")
print("="*80)

# 定义要删除的文件夹
folders_to_clean = [
    'Cleaned_Data',
    'Results',
    'Figures',
]

# 定义要保留的文件夹
folders_to_keep = [
    'Original Data',
    'Code',
]

print("\n将要清理的文件夹:")
print("-" * 80)

for folder in folders_to_clean:
    if os.path.exists(folder):
        # 删除文件夹中的所有内容
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"  ✓ 已删除: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  ✓ 已删除文件夹: {item_path}")
            except Exception as e:
                print(f"  ✗ 删除失败: {item_path} ({e})")
        print(f"✓ 已清空: {folder}/")
    else:
        # 如果文件夹不存在，创建它
        os.makedirs(folder)
        print(f"✓ 已创建: {folder}/")

print("\n" + "="*80)
print("检查核心文件")
print("="*80)

# 检查Original Data
print("\nOriginal Data/ 应包含:")
original_data_files = [
    'summerOly_athletes.csv',
    'summerOly_medal_counts.csv',
    'summerOly_hosts.csv',
    'summerOly_programs.csv',
    'data_dictionary.csv',
]

all_present = True
for file in original_data_files:
    file_path = os.path.join('Original Data', file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"  ✓ {file} ({size:.1f} KB)")
    else:
        print(f"  ✗ {file} - 缺失！")
        all_present = False

if not all_present:
    print("\n⚠️ 警告: Original Data中有文件缺失！")
else:
    print("\n✓ Original Data完整")

# 检查Code文件夹
print("\nCode/ 应包含:")
code_files = [
    '01_initial_eda.py',
    '02_investigate_issues.py',
    '03_data_cleaning_part1.py',
    '03_data_cleaning_part2_FIXED.py',
    'create_complete_mapping.py',
]

if os.path.exists('Code'):
    for file in code_files:
        file_path = os.path.join('Code', file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✓ {file} ({size:.1f} KB)")
        else:
            print(f"  ○ {file} - 不存在（可选）")
else:
    print("  ⚠️ Code文件夹不存在")

print("\n" + "="*80)
print("✓✓✓ 清理完成！")
print("="*80)

print("\n项目现在处于干净状态，可以重新开始！")
print("\n下一步:")
print("  1. 确保Code文件夹中有5个Python脚本")
print("  2. 按顺序运行脚本")
print("  3. 验证每一步的输出")

print("\n运行顺序:")
print("  python Code/01_initial_eda.py")
print("  python Code/02_investigate_issues.py")
print("  python Code/03_data_cleaning_part1.py")
print("  python Code/create_complete_mapping.py")
print("  python Code/03_data_cleaning_part2_FIXED.py")
