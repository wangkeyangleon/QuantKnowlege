## 整体技术栈

| 层次        | 主要技术                                              | 作用                               |
| --------- | ------------------------------------------------- | -------------------------------- |
| 前端        | React 18、TypeScript、Vite                          | 管理后台、知识库、聊天、Agent 编排             |
| Python 后端 | Python 3.13、Quart、Peewee                          | 业务 API、账号权限、模型调用、RAG 服务          |
| Go 后端     | Go、Gin、GORM                                       | 高性能服务、数据摄取、解析管道、CLI、部分 Agent 运行时 |
| 文档解析      | DeepDoc、ONNX Runtime、OpenCV、PDF/DOCX/XLSX/PPT 解析库 | OCR、版面分析、表格识别、文本提取               |
| RAG 检索    | Elasticsearch、Infinity、OpenSearch                 | 全文检索、向量检索、混合召回                   |
| 元数据存储     | MySQL，可配置 PostgreSQL/OceanBase 等                  | 用户、知识库、文档、任务、配置                  |
| 对象存储      | MinIO、S3、OSS                                      | 保存原始文件、图片和解析产物                   |
| 缓存与任务     | Redis/Valkey、异步 Worker                            | 缓存、任务队列、状态和并发控制                  |
| Agent     | 工作流 Canvas、MCP、LangGraph、工具组件                     | 构建可视化 Agent 和工具调用流程              |
| 部署        | Docker、Docker Compose、Nginx、Helm/Kubernetes       | 本地部署和集群化部署                       |
| 工程质量      | uv、pytest、Ruff、Jest、ESLint、TypeScript、Lefthook    | 依赖、测试、代码检查和 Git Hook             |

## 目录结构
ragflow/
├── web/                # React 管理后台
├── api/                # Python API、数据库服务、业务接口
├── rag/                # 摄取、检索、LLM、GraphRAG
├── deepdoc/            # OCR、版面识别、表格识别、文档解析
├── agent/              # Python Agent 组件、工具、模板、Canvas
├── memory/             # Agent 记忆相关能力
├── mcp/                # MCP Server 和协议接入
├── cmd/                # Go 服务和 CLI 入口
├── internal/
│   ├── agent/          # Go Agent Runtime
│   ├── ingestion/      # Go 数据摄取管道
│   ├── parser/         # Go 文档解析和 Chunk
│   ├── deepdoc/        # Go 与原生 DeepDoc 集成
│   ├── engine/         # ES、Infinity 等搜索引擎适配
│   ├── handler/        # HTTP Handler
│   ├── service/        # Go 业务服务
│   ├── dao/            # Go 数据访问层
│   └── cpp/            # Go 原生能力使用的 C++ 代码
├── sdk/python/         # Python SDK
├── docker/             # Docker Compose、Nginx、配置
├── helm/               # Kubernetes Helm 部署
└── test/               # 自动化测试

## 一次请求完整链路
用户上传文档
       ↓
React 前端 / HTTP API
       ↓
MySQL 记录文档与任务
MinIO 保存原始文件
       ↓
异步摄取任务
       ↓
DeepDoc / MinerU / Docling
OCR + 版面分析 + 表格识别
       ↓
Chunker 分块、Tokenizer、信息提取
       ↓
Embedding 模型生成向量
       ↓
Elasticsearch / Infinity 建立
全文索引 + 向量索引
       ↓
用户提出问题
       ↓
关键词召回 + 向量召回 + 其他召回
       ↓
融合排序 + Reranker
       ↓
LLM 生成答案
       ↓
返回答案、原文引用与页面坐标

## 核心技术DeepDoc
传统 RAG 经常直接调用 PDF 文本提取器，然后按固定字数切割。RAGFlow 会进一步识别文档版面结构，包括：

- OCR 文字识别
- DLA 文档版面分析
- TSR 表格结构识别
- 标题、正文、页眉、页脚
- 图片及图片标题
- 表格及表格标题
- 参考文献
- 数学公式
- 页面坐标与页码

DeepDoc 可以把复杂表格重新组织成大模型更容易理解的文本，并保留文字在 PDF 中的位置，从而支持答案引用原文时跳转到对应页面区域。

当前 PDF 解析可以选择：

- **DeepDoc**：默认方案，执行 OCR、表格和版面分析
- **Naive**：纯文本 PDF 快速解析
- **MinerU**
- **Docling**
- **OpenDataLoader**
- 第三方视觉大模型/VLM

## Rag检索
偏向企业搜索的混合检索方案：

1. 文档解析后生成 Chunk；
2. 对 Chunk 执行 Embedding；
3. 同时建立全文索引和向量索引；
4. 查询时执行多路召回；
5. 合并关键词、向量和其他召回结果；
6. 使用 Reranker 重排序；
7. 将排名靠前的上下文送入 LLM；
8. 返回答案，并附带原文引用。

项目默认可以使用 **Elasticsearch** 同时保存全文和向量索引，也可以切换到 InfiniFlow 自己的 **Infinity**；代码结构还提供 OpenSearch 等搜索引擎适配。

它还包含：

- GraphRAG
- RAPTOR 类层级语义构建
- 查询改写
- 多路召回
- 融合排序
- Reranker 模型
- Chunk 可视化和人工修正
- 可追溯引用

## Agent 工作流
主要负责：

- 可视化工作流 Canvas
- LLM 节点
- 知识库检索节点
- 条件判断
- 循环与变量
- HTTP/API 工具
- 搜索工具
- Python/JavaScript 代码执行
- 多 Agent 或工作流组合
- MCP 工具连接
- Agent 模板