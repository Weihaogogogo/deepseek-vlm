# fake-vlm

伪多模态封装：给无识图能力的纯文本 LLM（deepseek-v4-flash）补一双眼睛。

对外表现为一个 OpenAI 兼容的"模型"（base_url + api_key），内部按输入路由：

- **无图** → 直接透传 v4flash，零额外开销
- **有图** → 并发调用两个 qwen3-vl-flash：
  - VLM-1：纯图片，输出画面结构化描述（区分表格截图类 / 画面类模板）
  - VLM-2：图片 + 用户原文，输出问题驱动的聚焦描述
  - 合并两路描述 → 连同用户问题发给 v4flash 推理

## 目录结构

```
fake-vlm/
├── schemas/    # 双通道 JSON schema + 合并格式
├── prompts/    # VLM-1 / VLM-2 / v4flash 消费 prompt 模板
├── src/        # 路由、VLM 客户端、合并器、OpenAI 兼容服务层
├── tests/      # 盲测图片 + 用例记录
└── README.md
```

## 关键决策（讨论中，定稿后更新）

- 模型：qwen3-vl-flash（¥0.15/M 输入，单图 ~1024 token，千张成本约 ¥0.6）
- 图片统一压缩到 1024 边长再发，控制 token 和上传体积
- 成本大头是输出 token，描述走精简 JSON
- 双通道合并规则：字段级分工，冲突以聚焦通道为准

## 状态

- [x] 目录骨架
- [ ] 双通道 schema 定稿
- [ ] 路由 + 合并器原型
- [ ] OpenAI 兼容服务层
- [ ] 盲测（截图/场景/图表/计数）
