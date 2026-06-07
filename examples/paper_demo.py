"""
NeuroWeave Cortex — 论文可复现 Demo
=====================================
一键运行：展示完整认知生命周期

1. 初始化系统
2. 喂入长期对话（20+ 轮，模拟多会话）
3. 展示：remember → recall → sleep → 再次 recall
4. 打印睡眠前后图结构变化
5. 打印工作记忆空间内容

Run: python examples/paper_demo.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_graph import MemoryManager, AgentContext, seed_everything


def main():
    seed_everything(42)
    mgr = MemoryManager(storage_path="paper_demo.db")

    print("=" * 65)
    print("  NeuroWeave Cortex — 认知记忆引擎演示")
    print("=" * 65)

    # ── Phase 1: 喂入多轮对话 ──
    print("\n[Phase 1] 喂入长期对话 (3 会话 × 7 轮)")
    print("-" * 45)

    conversations = [
        # 会话 1：项目启动
        ("s01", [
            "我们决定用 Python 写这个爬虫工具，requests + BeautifulSoup 就够了",
            "目标网站是动态渲染的，需要加上 Selenium 处理 JavaScript",
            "用户认证用 JWT token，从 localStorage 提取",
            "数据存 MongoDB，按 domain 分 collection",
            "反爬策略：随机 User-Agent + 代理池 + 请求间隔 2-5s",
            "优先级队列用 Redis sorted set，score = 1/age",
            "今天先搭框架，明天开始写具体的解析器",
        ]),
        # 会话 2：遇到问题
        ("s02", [
            "Selenium 太慢了，改成 requests + 直接 API 调用",
            "发现目标 API 返回的是 GraphQL，需要解析嵌套结构",
            "MongoDB 连接数太多，加连接池限制到 10",
            "内存泄漏：解析器没关文件句柄，加 context manager",
            "代理池有一个节点返回了错误数据，需要加验证中间件",
            "JWT token 会过期，加自动刷新逻辑",
            "性能基准：单线程 50req/s，目标是 200req/s",
        ]),
        # 会话 3：维护期
        ("s03", [
            "用户反馈：爬虫跑了一周，数据量 500GB，查询变慢了",
            "加 MongoDB 索引：domain + timestamp 复合索引",
            "Redis 内存告警：清理过期 key，加 LRU eviction",
            "加监控：Prometheus + Grafana，关键指标是队列深度和错误率",
            "需要写文档：API 文档 + 部署文档 + 运维手册",
            "代码 review：抽象出一个 BaseParser，各站点继承",
            "下个迭代：支持分布式部署，用 Redis 做任务队列",
        ]),
    ]

    for session_id, turns in conversations:
        for i, text in enumerate(turns):
            importance = 0.7 if i in [0, 3, 6] else 0.5  # 重要节点
            mgr.remember(text, source_session=session_id,
                         tags=["crawler", session_id], importance=importance)
        print(f"  ✅ Session {session_id}: {len(turns)} 条记忆")

    # ── Phase 2: 首次检索 ──
    print("\n[Phase 2] 检索 (sleep 前)")
    print("-" * 45)
    queries = [
        "爬虫性能太慢怎么优化",
        "Redis 配置和内存管理",
        "MongoDB 查询优化",
    ]
    for q in queries:
        ctx = mgr.recall(query=q, context=AgentContext(task_type="qa"))
        top = ctx.items[0] if ctx.items else None
        if top:
            text = top.compressed_text or (top.anchor.text[:120] if top.anchor else "")
            score = top.relevance_score
        else:
            text, score = "(无结果)", 0
        print(f"  Q: {q}")
        print(f"  A: [{score:.2f}] {text[:100]}...")
        print()

    # ── Phase 3: 睡眠 ──
    stats_before = mgr.get_memory_stats()
    print("\n[Phase 3] 睡眠整合")
    print("-" * 45)
    print(f"  睡眠前: {stats_before['anchor_count']} anchors, {stats_before['edge_count']} edges")
    t0 = time.time()
    report = mgr.sleep()
    sleep_ms = (time.time() - t0) * 1000
    print(f"  睡眠耗时: {sleep_ms:.0f}ms")
    print(f"  合并: {report.get('merged', 0)}, 修剪: {report.get('pruned', 0)}")

    stats_after = mgr.get_memory_stats()
    print(f"  睡眠后: {stats_after['anchor_count']} anchors, {stats_after['edge_count']} edges")

    # ── Phase 4: 再次检索 ──
    print("\n[Phase 4] 检索 (sleep 后)")
    print("-" * 45)
    for q in queries:
        ctx = mgr.recall(query=q, context=AgentContext(task_type="qa"))
        top = ctx.items[0] if ctx.items else None
        if top:
            text = top.compressed_text or (top.anchor.text[:120] if top.anchor else "")
            score = top.relevance_score
        else:
            text, score = "(无结果)", 0
        print(f"  Q: {q}")
        print(f"  A: [{score:.2f}] {text[:100]}...")
        print()

    # ── Phase 5: 图统计 ──
    print("[Phase 5] 记忆统计")
    print("-" * 45)
    stats = mgr.get_memory_stats()
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 65)
    print("  Demo 完成 ✅")
    print("=" * 65)


if __name__ == "__main__":
    main()
