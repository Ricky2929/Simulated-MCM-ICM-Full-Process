# 特征工程说明文档
# Feature Engineering Guide

## 📊 创建的特征类别

### 1️⃣ 滞后特征 (Lag Features) - 12个

**目的:** 使用历史奖牌数作为预测特征

**特征列表:**
- `Gold_lag1`, `Gold_lag2`, `Gold_lag3` - 上1/2/3届的金牌数
- `Silver_lag1`, `Silver_lag2`, `Silver_lag3` - 上1/2/3届的银牌数
- `Bronze_lag1`, `Bronze_lag2`, `Bronze_lag3` - 上1/2/3届的铜牌数
- `Total_lag1`, `Total_lag2`, `Total_lag3` - 上1/2/3届的总奖牌数

**为什么重要:**
- 历史表现是未来表现的最强预测因子
- 奖牌数通常具有连续性和惯性

---

### 2️⃣ 滚动统计特征 (Rolling Statistics) - 32个

**目的:** 计算过去N届的统计指标

**特征类型:**
- **滚动平均** (8个): `Gold_rolling_mean_3`, `Gold_rolling_mean_5`, ...
  - 衡量近期平均水平
  
- **滚动最大值** (8个): `Gold_rolling_max_3`, `Gold_rolling_max_5`, ...
  - 捕捉历史最佳表现
  
- **滚动最小值** (8个): `Gold_rolling_min_3`, `Gold_rolling_min_5`, ...
  - 识别表现底线
  
- **滚动标准差** (8个): `Gold_rolling_std_3`, `Gold_rolling_std_5`, ...
  - 衡量表现稳定性

**为什么重要:**
- 平滑短期波动
- 捕捉长期趋势
- 评估稳定性

---

### 3️⃣ 趋势特征 (Trend Features) - 24个

**目的:** 衡量奖牌数的变化趋势

**特征类型:**
- **增长率** (4个): `Gold_growth_rate`, `Silver_growth_rate`, ...
  - 相对上届的百分比变化
  
- **绝对变化** (4个): `Gold_change`, `Silver_change`, ...
  - 相对上届的差值
  
- **加速度** (4个): `Gold_acceleration`, `Silver_acceleration`, ...
  - 增长率的变化（二阶导数）
  
- **趋势方向** (8个): `Gold_is_growing`, `Gold_streak`, ...
  - 是否增长、连续增长次数

**为什么重要:**
- 捕捉上升/下降趋势
- 识别突破性进步或衰退
- 动量效应

---

### 4️⃣ 交互特征 (Interaction Features) - 9个

**目的:** 组合多个特征捕捉复杂关系

**特征列表:**
- `Gold_host_boost` - 东道主 × 历史金牌平均
- `Silver_host_boost` - 东道主 × 历史银牌平均
- `Bronze_host_boost` - 东道主 × 历史铜牌平均
- `Total_host_boost` - 东道主 × 历史总奖牌平均
- `market_size_interaction` - 市场份额 × 规模
- `experience_performance` - 经验 × 表现
- `gold_ratio` - 金牌率 (Gold/Total)
- `silver_ratio` - 银牌率
- `bronze_ratio` - 铜牌率
- `gold_silver_ratio` - 金银比

**为什么重要:**
- **东道主效应加成**: 东道主优势对强队影响更大
- **市场份额**: 相对实力比绝对数量更稳定
- **奖牌结构**: 反映国家的竞技实力分布

---

### 5️⃣ 竞争强度特征 (Competition Intensity) - 5个

**目的:** 衡量相对竞争力

**特征列表:**
- `avg_gold_year` - 该年度的平均金牌数
- `std_gold_year` - 该年度的金牌标准差
- `max_gold_year` - 该年度的最大金牌数
- `num_countries` - 该年度的参赛国家数
- `gold_vs_avg` - 金牌数相对年度平均的差值
- `gold_percentile` - 该年度的金牌排名百分位

**为什么重要:**
- 标准化不同时期的表现
- 考虑竞争环境变化
- 相对表现比绝对数量更有意义

---

### 6️⃣ 特殊事件特征 (Special Events) - 3个

**目的:** 标记特殊情况

**特征列表:**
- `is_first_participation` - 是否首次参赛
- `is_first_medal` - 是否首次获得金牌
- `is_comeback` - 是否中断后回归 (超过2届未参赛)

**为什么重要:**
- 首次参赛通常表现较弱
- 首次获奖是重要里程碑
- 中断回归可能影响表现

---

## 📈 特征统计摘要

**总特征数:** ~85个新特征

| 类别 | 数量 | 占比 |
|------|------|------|
| 滞后特征 | 12 | 14% |
| 滚动统计 | 32 | 38% |
| 趋势特征 | 24 | 28% |
| 交互特征 | 9 | 11% |
| 竞争特征 | 5 | 6% |
| 特殊事件 | 3 | 3% |

---

## 🎯 关键特征（预期最重要的TOP 10）

根据领域知识，这些特征最可能对预测有重要贡献：

1. **Gold_lag1** - 上届金牌数（最强预测因子）
2. **Gold_rolling_mean_3** - 近3届金牌平均（稳定指标）
3. **is_host** - 东道主标记（+597%效应！）
4. **Gold_host_boost** - 东道主加成（交互效应）
5. **gold_percentile** - 相对排名（标准化表现）
6. **Gold_growth_rate** - 增长率（趋势）
7. **Market_Share_Gold** - 金牌市场份额
8. **Total_Events** - 项目规模（影响机会）
9. **olympic_experience** - 参赛经验
10. **Gold_rolling_std_3** - 表现稳定性

---

## ⚠️ 缺失值情况

**预期缺失值来源:**

1. **滞后特征**: 国家首次参赛时没有历史数据
   - 例如：Albania 2024年首次参赛，所有lag特征为NaN
   
2. **滚动标准差**: 需要至少2个数据点
   - 首次参赛时为NaN

3. **趋势特征**: 需要至少2个时间点
   - 首次参赛时growth_rate为NaN

**处理策略（建模时）:**
- 方法1: 填充为0（假设无历史=无优势）
- 方法2: 填充为全局平均（保守估计）
- 方法3: 使用模型处理（如树模型可以处理NaN）
- 方法4: 创建"is_missing"标记特征

---

## 🚀 使用方法

### **运行特征工程:**

```bash
python Code/04_feature_engineering.py
```

### **预期输出:**

**文件:**
- `Cleaned_Data/master_data_with_features.csv` - 主数据（含新特征）
- `Results/11_new_features_list.csv` - 新特征列表
- `Results/12_feature_summary.csv` - 特征分类摘要

**终端输出:**
- 每个步骤的进度
- 新增特征统计
- 缺失值情况
- 完成摘要

### **预期运行时间:**
- **1-2分钟**（取决于数据量）

---

## 📊 验证特征质量

运行完后，可以用这段代码快速验证：

```python
import pandas as pd

# 加载特征工程后的数据
df = pd.read_csv('Cleaned_Data/master_data_with_features.csv')

print(f"总特征数: {len(df.columns)}")
print(f"记录数: {len(df):,}")

# 查看新特征的前几行
new_features = ['Gold_lag1', 'Gold_rolling_mean_3', 'Gold_growth_rate', 
                'Gold_host_boost', 'gold_percentile']
print("\n新特征示例:")
print(df[['Year', 'NOC', 'Gold', 'is_host'] + new_features].head(10))

# 检查缺失值
print("\n缺失值统计:")
missing = df[new_features].isna().sum()
print(missing)
```

---

## 🎯 下一步

完成特征工程后：

1. **特征选择**（可选）
   - 移除高度相关的特征
   - 使用特征重要性分析

2. **开始建模**
   - 准备训练/验证/测试集
   - 尝试多个模型
   - 评估预测效果

3. **2028预测**
   - 使用最佳模型
   - 生成预测区间
   - 特别关注美国（东道主效应！）

---

## 💡 特征工程设计原则

我们遵循的设计原则：

1. ✅ **领域驱动**: 基于奥运会和运动竞技的领域知识
2. ✅ **时间一致性**: 只使用"过去"的信息（避免数据泄漏）
3. ✅ **模型通用**: 特征适用于所有模型类型
4. ✅ **可解释性**: 每个特征都有明确的业务含义
5. ✅ **稳健性**: 考虑缺失值和异常值情况

---

**准备好了吗？运行特征工程脚本，让我们看看效果！** 🚀
