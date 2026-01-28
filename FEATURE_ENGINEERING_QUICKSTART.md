# 🚀 特征工程 - 快速开始
# Feature Engineering - Quick Start

## 📋 准备工作（已完成✅）

- [x] 数据清洗完成
- [x] master_data.csv 已生成
- [x] 数据质量验证通过

---

## 🎯 现在执行（只需2步）

### **第1步：运行特征工程脚本（1-2分钟）**

```bash
python Code/04_feature_engineering.py
```

或者把脚本移到Code文件夹：
```bash
move 04_feature_engineering.py Code/
python Code/04_feature_engineering.py
```

---

### **第2步：验证结果（30秒）**

运行完后，检查：

**✅ 文件生成：**
- [ ] `Cleaned_Data/master_data_with_features.csv` 存在
- [ ] `Results/11_new_features_list.csv` 存在
- [ ] `Results/12_feature_summary.csv` 存在

**✅ 终端输出：**
- [ ] 看到 "✓✓✓ 特征工程完成!"
- [ ] 新增特征数 ~85个
- [ ] 没有报错

---

## 📊 创建的特征概览

### **总共约85个新特征：**

1. **滞后特征** (12个)
   - Gold_lag1, Gold_lag2, Gold_lag3
   - Silver/Bronze/Total 同样处理

2. **滚动统计** (32个)
   - 滚动平均、最大、最小、标准差
   - 窗口：3届、5届

3. **趋势特征** (24个)
   - 增长率、变化量、加速度
   - 趋势方向、连续性

4. **交互特征** (9个)
   - 东道主加成
   - 市场份额交互
   - 奖牌比率

5. **竞争特征** (5个)
   - 相对排名
   - 年度统计

6. **特殊事件** (3个)
   - 首次参赛/获奖
   - 中断回归

---

## 🔍 快速验证代码

```python
import pandas as pd

# 加载数据
df = pd.read_csv('Cleaned_Data/master_data_with_features.csv')

print(f"✓ 总特征数: {len(df.columns)}")
print(f"✓ 记录数: {len(df):,}")

# 查看关键新特征
key_features = ['Gold_lag1', 'Gold_rolling_mean_3', 
                'Gold_growth_rate', 'Gold_host_boost']

print("\n关键新特征 (2024年美国示例):")
usa_2024 = df[(df['NOC']=='USA') & (df['Year']==2024)]
print(usa_2024[['Year', 'NOC', 'Gold', 'is_host'] + key_features])

# 缺失值检查
print(f"\n缺失值情况:")
print(f"滞后特征缺失: {df['Gold_lag1'].isna().sum()} 条")
print(f"  (原因: 首次参赛国家没有历史数据)")
```

---

## ✅ 成功标志

运行完后应该看到：

```
✓✓✓ 特征工程完成!

关键统计:
  原始特征数: 19
  新增特征数: ~85
  最终特征数: ~104
  数据记录数: 949

生成的文件:
  - Cleaned_Data/master_data_with_features.csv
  - Results/11_new_features_list.csv
  - Results/12_feature_summary.csv

下一步:
  1. 处理缺失值 (如果需要)
  2. 特征选择 (可选)
  3. 开始建模!
```

---

## 🎯 完成后：查看特征摘要

```python
import pandas as pd

# 查看特征摘要
summary = pd.read_csv('Results/12_feature_summary.csv')
print(summary)

# 输出示例:
#            Category  Count
# 0      Lag Features     12
# 1  Rolling Statistics  32
# 2     Trend Features     24
# 3  Interaction Features  9
# 4  Competition Features  5
# 5     Special Events     3
```

---

## 📈 关键特征预览（TOP 10）

最重要的特征（预测贡献度）：

1. **Gold_lag1** ⭐⭐⭐ - 上届金牌（最强）
2. **Gold_rolling_mean_3** ⭐⭐⭐ - 近3届平均
3. **is_host** ⭐⭐⭐ - 东道主（+597%!）
4. **Gold_host_boost** ⭐⭐ - 东道主加成
5. **gold_percentile** ⭐⭐ - 相对排名
6. **Gold_growth_rate** ⭐⭐ - 增长趋势
7. **Market_Share_Gold** ⭐⭐ - 市场份额
8. **Total_Events** ⭐ - 项目规模
9. **olympic_experience** ⭐ - 参赛经验
10. **Gold_rolling_std_3** ⭐ - 稳定性

---

## 🚀 下一步

特征工程完成后：

**选项1: 直接建模（推荐）**
```
→ 开始建立基线模型
→ 比较多个模型
→ 选择最佳模型
```

**选项2: 特征分析**
```
→ 特征相关性分析
→ 特征重要性排序
→ 移除冗余特征
```

**选项3: 缺失值处理**
```
→ 分析缺失模式
→ 选择填充策略
→ 创建缺失标记
```

---

## 💡 提示

**关于缺失值：**
- 滞后特征的缺失是**正常的**（首次参赛国家）
- 大多数树模型（RF、XGBoost）可以直接处理NaN
- 如果需要填充，建议用0（无历史=无优势）

**关于特征数量：**
- 85个特征不算多（相对于950条数据）
- 树模型可以自动选择重要特征
- 如果担心过拟合，可以后期做特征选择

---

## 📞 遇到问题？

**问题1: 运行时间太长**
→ 正常，数据量大时需要2-3分钟

**问题2: 内存不足**
→ 关闭其他程序，或减少滚动窗口数量

**问题3: 某个特征全是NaN**
→ 检查原始数据是否有该列

**问题4: 报错"KeyError"**
→ 确保master_data.csv包含所有必需列

---

## ✅ 检查清单

运行前：
- [ ] master_data.csv 存在于 Cleaned_Data/
- [ ] Python环境有pandas和numpy

运行后：
- [ ] master_data_with_features.csv 已生成
- [ ] 文件大小 > 原始数据（更多列）
- [ ] 没有报错信息

---

**准备好了吗？运行脚本，然后告诉我结果！** 🎯

```bash
python Code/04_feature_engineering.py
```

**把终端输出的最后10-15行截图给我！** 📊
