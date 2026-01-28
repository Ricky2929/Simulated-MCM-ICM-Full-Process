# create_complete_mapping.py
# 创建超级完整的国家名称到NOC映射

import pandas as pd

print("="*80)
print("创建完整的国家名称到NOC映射")
print("Create Complete Country Name to NOC Mapping")
print("="*80)

# 1. 加载原始数据
print("\n1. 加载原始数据...")
df_medal_counts = pd.read_csv('Original Data/summerOly_medal_counts.csv')
df_athletes = pd.read_csv('Original Data/summerOly_athletes.csv')

# 去除空格
df_medal_counts['NOC'] = df_medal_counts['NOC'].str.strip()

print(f"✓ Medal Counts: {len(df_medal_counts)} 行")
print(f"✓ Athletes: {len(df_athletes)} 行")

# 2. 获取所有唯一的国家名称和NOC
country_names = df_medal_counts['NOC'].unique()
noc_codes = df_athletes['NOC'].unique()

print(f"\n唯一国家名称: {len(country_names)}")
print(f"唯一NOC代码: {len(noc_codes)}")

# 3. 创建超级完整的映射字典
print("\n3. 创建映射字典...")

# 策略：
# A. 从Athletes获取Team名称，建立NOC到国家名称的映射
# B. 反向映射得到国家名称到NOC
# C. 手动添加无法自动匹配的历史国家

# A. 自动学习映射
print("\n正在从Athletes数据学习NOC到国家名称的映射...")

noc_to_country_learned = {}
for noc in noc_codes:
    # 获取该NOC的所有Team名称
    teams = df_athletes[df_athletes['NOC'] == noc]['Team'].unique()
    if len(teams) > 0:
        # 使用最常见的Team名称
        most_common_team = teams[0]
        noc_to_country_learned[noc] = most_common_team

print(f"✓ 自动学习了 {len(noc_to_country_learned)} 个映射")

# B. 创建反向映射
country_to_noc_learned = {}
for noc, country in noc_to_country_learned.items():
    # 去掉Team名称中的后缀（如Germany-1 → Germany）
    country_clean = country.split('-')[0].strip()
    country_to_noc_learned[country_clean] = noc

print(f"✓ 创建了 {len(country_to_noc_learned)} 个反向映射")

# C. 手动补充和修正（基于深度诊断结果）
print("\n添加手动映射和历史国家...")

manual_mappings = {
    # 主要国家的标准名称
    'United States': 'USA',
    'Great Britain': 'GBR',
    'Germany': 'GER',
    'France': 'FRA',
    'Italy': 'ITA',
    'Japan': 'JPN',
    'China': 'CHN',
    'Australia': 'AUS',
    'Canada': 'CAN',
    'Russia': 'RUS',
    'Spain': 'ESP',
    'Brazil': 'BRA',
    'Netherlands': 'NED',
    'South Korea': 'KOR',
    'Sweden': 'SWE',
    'Hungary': 'HUN',
    'Poland': 'POL',
    'Romania': 'ROU',
    'Cuba': 'CUB',
    'Norway': 'NOR',
    'Finland': 'FIN',
    'Bulgaria': 'BUL',
    'Denmark': 'DEN',
    'Switzerland': 'SUI',
    'Belgium': 'BEL',
    'New Zealand': 'NZL',
    'Greece': 'GRE',
    'Austria': 'AUT',
    'Turkey': 'TUR',
    'Mexico': 'MEX',
    'South Africa': 'RSA',
    'Argentina': 'ARG',
    'Ireland': 'IRL',
    'Egypt': 'EGY',
    'India': 'IND',
    'Jamaica': 'JAM',
    'Kenya': 'KEN',
    'Ethiopia': 'ETH',
    'Thailand': 'THA',
    'Czech Republic': 'CZE',
    'Ukraine': 'UKR',
    'Indonesia': 'INA',
    'Malaysia': 'MAS',
    'Philippines': 'PHI',
    'Vietnam': 'VIE',
    'North Korea': 'PRK',
    'Kazakhstan': 'KAZ',
    'Uzbekistan': 'UZB',
    'Georgia': 'GEO',
    'Azerbaijan': 'AZE',
    'Belarus': 'BLR',
    'Croatia': 'CRO',
    'Serbia': 'SRB',
    'Slovenia': 'SLO',
    'Slovakia': 'SVK',
    'Estonia': 'EST',
    'Latvia': 'LAT',
    'Lithuania': 'LTU',
    'Armenia': 'ARM',
    'Moldova': 'MDA',
    'Portugal': 'POR',
    'Israel': 'ISR',
    'Iran': 'IRI',
    'Chinese Taipei': 'TPE',
    'Hong Kong': 'HKG',
    'Singapore': 'SGP',
    'Algeria': 'ALG',
    'Tunisia': 'TUN',
    'Morocco': 'MAR',
    'Nigeria': 'NGR',
    'Cameroon': 'CMR',
    'Zimbabwe': 'ZIM',
    'Colombia': 'COL',
    'Venezuela': 'VEN',
    'Chile': 'CHI',
    'Ecuador': 'ECU',
    'Peru': 'PER',
    'Uruguay': 'URU',
    'Dominican Republic': 'DOM',
    'Puerto Rico': 'PUR',
    'Trinidad and Tobago': 'TTO',
    'Bahamas': 'BAH',
    'Pakistan': 'PAK',
    'Sri Lanka': 'SRI',
    'Mongolia': 'MGL',
    
    # 历史国家（映射到Part1中已处理的目标NOC）
    'Russian Empire': 'RUS',        # 俄罗斯帝国 → RUS
    'Soviet Union': 'RUS',          # 苏联 → RUS
    'Unified Team': 'RUS',          # 独联体 → RUS
    'Czechoslovakia': 'CZE',        # 捷克斯洛伐克 → CZE（将在后续分配）
    'Yugoslavia': 'SRB',            # 南斯拉夫 → SRB
    'Serbia and Montenegro': 'SRB', # 塞黑 → SRB
    'East Germany': 'GER',          # 东德 → GER
    'West Germany': 'GER',          # 西德 → GER
    'Bohemia': 'CZE',               # 波西米亚 → CZE
    'Australasia': 'ANZ',           # 澳大拉西亚（保持独立）
    
    # 特殊团队
    'Mixed team': 'ZZX',            # 混合队（特殊标记）
    'Independent Olympic Athletes': 'IOA',
    'Individual Olympic Athletes': 'IOA',
    'Refugee Olympic Team': 'EOR',
    
    # 其他可能的历史名称
    'Ceylon': 'SRI',                # 锡兰 → 斯里兰卡
    'Rhodesia': 'ZIM',              # 罗得西亚 → 津巴布韦
    'Burma': 'MYA',                 # 缅甸旧名
    'Siam': 'THA',                  # 泰国旧名
    'Zaire': 'COD',                 # 扎伊尔 → 刚果民主共和国
}

# 合并映射
final_mapping = {**country_to_noc_learned, **manual_mappings}

print(f"✓ 最终映射字典包含 {len(final_mapping)} 个映射")

# 4. 测试映射效果
print("\n" + "="*80)
print("4. 测试映射效果")
print("="*80)

# 尝试映射所有Medal Counts中的国家名称
unmapped = []
mapped_count = 0

for country in country_names:
    if country in final_mapping:
        mapped_count += 1
    else:
        unmapped.append(country)

print(f"\n成功映射: {mapped_count}/{len(country_names)} ({mapped_count/len(country_names)*100:.1f}%)")
print(f"未能映射: {len(unmapped)}/{len(country_names)} ({len(unmapped)/len(country_names)*100:.1f}%)")

if unmapped:
    print(f"\n未能映射的国家（前20个）:")
    for i, country in enumerate(unmapped[:20], 1):
        count = (df_medal_counts['NOC'] == country).sum()
        print(f"  {i:2d}. '{country}' ({count} 条记录)")

# 5. 应用映射
print("\n" + "="*80)
print("5. 应用映射")
print("="*80)

df_medal_counts['NOC_original'] = df_medal_counts['NOC'].copy()
df_medal_counts['NOC'] = df_medal_counts['NOC'].map(final_mapping)

# 对于未映射的，保持原值
df_medal_counts.loc[df_medal_counts['NOC'].isna(), 'NOC'] = \
    df_medal_counts.loc[df_medal_counts['NOC'].isna(), 'NOC_original']

mapped_records = df_medal_counts['NOC'].notna().sum()
print(f"✓ 成功映射 {mapped_records}/{len(df_medal_counts)} 条记录")

# 6. 验证结果
print("\n" + "="*80)
print("6. 验证结果")
print("="*80)

noc_athletes = set(df_athletes['NOC'].unique())
noc_medals = set(df_medal_counts['NOC'].dropna().unique())

common = noc_athletes & noc_medals
only_athletes = noc_athletes - noc_medals
only_medals = noc_medals - noc_athletes

print(f"\n共同的NOC: {len(common)}")
print(f"只在Athletes中: {len(only_athletes)}")
print(f"只在Medal Counts中: {len(only_medals)}")

if len(common) > 130:
    print(f"\n✓✓✓ 映射成功！共同NOC数 > 130")
else:
    print(f"\n⚠️ 共同NOC数偏少 ({len(common)})")

# 7. 显示映射示例
print("\n" + "="*80)
print("7. 映射示例")
print("="*80)

print("\n成功映射的前20个:")
mapped_examples = df_medal_counts[df_medal_counts['NOC_original'] != df_medal_counts['NOC']].drop_duplicates('NOC')[:20]
for _, row in mapped_examples.iterrows():
    print(f"  '{row['NOC_original']}' → {row['NOC']}")

# 8. 保存结果
print("\n" + "="*80)
print("8. 保存结果")
print("="*80)

# 保存映射后的Medal Counts
df_medal_counts.to_csv('Cleaned_Data/medal_counts_complete_mapping.csv', index=False)
print("✓ 已保存: Cleaned_Data/medal_counts_complete_mapping.csv")

# 保存映射字典
import json
with open('Cleaned_Data/country_to_noc_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(final_mapping, f, indent=2, ensure_ascii=False)
print("✓ 已保存映射字典: Cleaned_Data/country_to_noc_mapping.json")

# 更新checkpoint
df_medal_counts.to_csv('Cleaned_Data/checkpoint_medal_counts_after_step5.csv', index=False)
print("✓ 已更新checkpoint")

print("\n" + "="*80)
if len(common) > 130:
    print("✓✓✓ 映射完成！可以继续运行Part 2")
    print("="*80)
    print("\n下一步: python 03_data_cleaning_part2_FIXED.py")
else:
    print("⚠️ 映射基本完成，但还有改进空间")
    print("="*80)
    print("\n建议: 先验证映射结果，然后继续")
    print("运行: python verify_noc_fix.py")
