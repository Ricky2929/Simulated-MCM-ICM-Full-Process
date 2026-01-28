import pandas as pd

# 加载主数据
df = pd.read_csv('Cleaned_Data/master_data.csv')

print("="*60)
print("Master Data 最终验证")
print("="*60)

# 基本信息
print(f"\n总记录数: {len(df):,}")
print(f"唯一NOC: {df['NOC'].nunique()}")
print(f"年份范围: {df['Year'].min()} - {df['Year'].max()}")
print(f"年份数: {df['Year'].nunique()}")
print(f"列数: {len(df.columns)}")

# 查看列名
print(f"\n列名 ({len(df.columns)}个):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# 关键列检查
required = ['Year', 'NOC', 'Gold', 'is_host', 'Total_Events', 'Market_Share_Gold']
missing = [col for col in required if col not in df.columns]
if missing:
    print(f"\n❌ 缺失关键列: {missing}")
else:
    print(f"\n✓ 所有关键列都存在")

# 东道主统计
print(f"\n东道主统计:")
print(f"  东道主记录数: {df['is_host'].sum()}")
host_avg = df[df['is_host']==1]['Gold'].mean()
non_host_avg = df[df['is_host']==0]['Gold'].mean()
print(f"  东道主平均金牌: {host_avg:.2f}")
print(f"  非东道主平均金牌: {non_host_avg:.2f}")
boost = ((host_avg / non_host_avg) - 1) * 100
print(f"  东道主增幅: +{boost:.0f}%")

# 市场份额检查
has_market = (df['Market_Share_Gold'] > 0).sum()
print(f"\n市场份额数据:")
print(f"  有Market Share的记录: {has_market:,}")

# 2024年数据
df_2024 = df[df['Year'] == 2024]
print(f"\n2024年统计:")
print(f"  参赛国家: {len(df_2024)}")
print(f"  总金牌数: {df_2024['Gold'].sum()}")
print(f"  总奖牌数: {df_2024['Total'].sum()}")

# 2024年前10名
print(f"\n2024年金牌榜前10:")
top10 = df_2024.nlargest(10, 'Gold')[['NOC', 'Gold', 'Silver', 'Bronze', 'Total', 'is_host']]
print(top10.to_string(index=False))

# 查看前5行
print(f"\n数据前5行:")
print(df.head()[['Year', 'NOC', 'Gold', 'Total', 'is_host', 'Total_Events', 'Market_Share_Gold']])

print("\n" + "="*60)
print("✓✓✓ 数据验证完成！")
print("="*60)