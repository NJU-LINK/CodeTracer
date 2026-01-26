# mini-tracer 使用说明

## 1. 从 git 克隆并安装

```bash
git clone <your-repo-url>
cd mini-tracer
pip install -e .
```

安装后，`tracer` 命令将全局可用。

## 2. 配置

编辑 `src/tracer/config/tracer.yaml` 中的模型配置：

```yaml
model:
  api_base: "http://14.103.68.46/v1"  # OpenAI-compatible API endpoint
  api_key: "sk-VDcVBa2lirJCzMIdwCdoe1kXdphzUgNphNG530DcJZlN1TWz"
  model_name: "gpt-5"  # 可选，会自动检测
  model_kwargs: {}
```

**或者**使用环境变量（优先级更高）：

```bash
export TRACER_API_BASE="http://your-api-endpoint/v1"
export TRACER_API_KEY="your-api-key"
```

## 3. 运行 tracer

### 基本用法

```bash
# 对单个任务目录运行 tracer
tracer /path/to/task_dir --input-format step_id_maps --model gpt-5

# 使用 stage_starts 参数（如果 stage_ranges.json 不存在）
tracer /path/to/task_dir --input-format step_id_maps --stage-starts '1,8,23,38'

# 仅验证输入，不运行 agent
tracer /path/to/task_dir --input-format step_id_maps --dry-run
```

### 示例（使用内置 step_id_maps 数据）

```bash
tracer step_id_maps/miniswe/Anthropic__Claude-Sonnet-4-20250514-Thinking/miniswe-claude/add-benchmark-lm-eval-harness/miniswe-Anthropic__Claude-Sonnet-4-20250514-Thinking-add-benchmark-lm-eval-harness-54ac67f0 \
  --input-format step_id_maps \
  --model gpt-5
```

## 4. 输出文件

tracer 会在每个任务目录下创建一个 `<model_slug>/` 子文件夹，包含：

- `<model_slug>/mini_tracer_labels.json` - 标注的错误/无用步骤
- `<model_slug>/mini_tracer.traj.json` - tracer 自己的执行轨迹

例如：
```
task_dir/
  ├── task.md
  ├── steps.json
  ├── stage_ranges.json
  ├── tree.md
  └── gpt-5/
      ├── mini_tracer_labels.json
      └── mini_tracer.traj.json
```

## 5. 常见问题

**Q: 如何指定不同的配置文件？**
```bash
tracer /path/to/task_dir --config /path/to/custom.yaml
```

**Q: 如何控制成本限制？**
```bash
tracer /path/to/task_dir --cost-limit 1.0
```

**Q: 输出文件在哪里？**
输出文件默认位于 `<task_dir>/<model_slug>/`，其中 `model_slug` 是根据 `--model` 参数或配置文件中的 `model_name` 自动生成的。

## 6. 迁移/部署

mini-tracer 设计为完全可移植：
- 所有配置都在 `src/tracer/config/tracer.yaml` 或环境变量中
- `step_id_maps/` 数据已包含在仓库中
- 克隆后直接 `pip install -e .` 即可使用
- 无硬编码路径依赖

