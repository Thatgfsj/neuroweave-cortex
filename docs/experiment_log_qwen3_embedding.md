# 实验记录：Embedding 模型选择消融

## 背景
将默认嵌入模型从 `all-MiniLM-L6-v2` (384-dim) 替换为 `Qwen3-Embedding-0.6B` (1024-dim)，预期利用更大的模型和更高维度提升检索质量。

## 实验结果（Conv-26, 199 QA, pure cosine top-40）

| 指标 | all-MiniLM-L6-v2 | Qwen3-Embedding-0.6B | Δ |
|------|:----------------:|:--------------------:|:---:|
| Cat 4 (Composite) | 37.1% | 32.9% | -4.2 pp |
| Cat 5 (Adversarial) | 46.8% | 42.6% | -4.2 pp |

## 结论
Qwen3-Embedding-0.6B (1024-dim) 在 LoCoMo 检索任务上反而比 MiniLM (384-dim) 更差。原因分析：
1. LoCoMo 的 token-matching 指标（has_answer）偏好精确的词法对齐而非语义泛化
2. 1024维密集向量稀释了 BM25 融合所依赖的稀疏关键词信号
3. MiniLM 在句子相似度任务上的针对性微调更适合短文本精确匹配

## 建议
- 论文中使用 MiniLM 作为默认嵌入模型
- 将 Qwen3 实验结果写入论文消融章节，体现工程洞察
- 精排阶段引入 ColBERT/BGE-M3 token-level 匹配，而非替换召回模型
