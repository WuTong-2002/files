#!/bin/bash
# bash train_qwen3.5.sh
# ==================================================
# Qwen3-4B LoRA 多任务微调训练脚本（4:4:1 采样平衡版）
# 创建时间：2026-03-18
# 功能：自动记录训练日志，生成实验报告，适配长任务
# ==================================================

# 设置环境变量
source ~/.bashrc
export PATH="/root/miniforge3/bin:$PATH"
which python3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1
export DISABLE_VERSION_CHECK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo $CONDA_ROOT
echo $CONDA_DEFAULT_ENV

# ==================================================
# 关键参数配置（已针对多任务长度差异优化）
# ==================================================

# 模型相关
MODEL_PATH="/data1/LlamaFactory-01/Qwen3.5-2B"
TEMPLATE="qwen3_5"

# --- 修改点 1: 序列长度 (Task3 最大 1976，留余量防截断) ---
CUTOFF_LEN=2048

# --- 修改点 2: 显存与 Batch 平衡 (长度增加，单卡 Batch 减小) ---
BATCH_SIZE=8
ACCUMULATION_STEPS=8

# --- 修改点 3: 学习率与稳定性 (长序列需更小 LR) ---
LEARNING_RATE=1e-4
WARMUP_RATIO=0.15

# --- 修改点 4: LoRA 容量 (长短任务差异大，需更大 Rank) ---
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET="all"

# 其他设置
NUM_EPOCHS=3.0
MAX_SAMPLES=999999
SAVE_STEPS=500

# 输出配置
OUTPUT_DIR="/data1/LlamaFactory-01/saves/all/models/Qwen3.5-2B-$(date +%Y-%m-%d-%H-%M)"
LOG_DIR="${OUTPUT_DIR}/logs"
REPORT_FILE="${OUTPUT_DIR}/experiment_report.md"

# 数据集配置 (三个独立任务文件)
DATASET_DIR="/data1/LlamaFactory-01/data"
DATASETS="task1_train,task2_train,task3_train"

# 多任务采样配置 (4:4:1 样本比，Token 比约 1.5:1)
MIX_STRATEGY="interleave_under"
INTERLEAVE_PROBS="0.444,0.444,0.112"

# 任务数据文件路径 (用于报告统计)
TASK1_FILE="${DATASET_DIR}/task1_train.json"
TASK2_FILE="${DATASET_DIR}/task2_train.json"
TASK3_FILE="${DATASET_DIR}/task3_train.json"

# ==================================================
# 创建目录
# ==================================================

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# ==================================================
# 实验报告初始化
# ==================================================

cat > ${REPORT_FILE} << EOF
# 📊 Qwen3-4B LoRA 多任务微调实验报告 (4:4:1 采样)

**实验时间**: $(date '+%Y-%m-%d %H:%M:%S')

**实验人员**: $(whoami)@$(hostname)

**核心优化**:
1. Token 平衡：4:4:1 采样比例，缓解长任务梯度主导
2. 序列长度：${CUTOFF_LEN} (适配 Task3 长文本)
3. 模型容量：LoRA Rank ${LORA_RANK} (解耦长短任务特征)
4. 训练稳定：LR ${LEARNING_RATE}, Warmup ${WARMUP_RATIO}

---

## 1. 实验环境

### 1.1 硬件配置
EOF

# 记录 GPU 信息
echo "\`\`\`" >> ${REPORT_FILE}
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv >> ${REPORT_FILE}
echo "\`\`\`" >> ${REPORT_FILE}

cat >> ${REPORT_FILE} << EOF

### 1.2 软件配置
- **Python 版本**: $(python3 --version 2>&1)
- **PyTorch 版本**: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未知")
- **Transformers 版本**: $(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "未知")
- **LlamaFactory 版本**: $(python3 -c "import llamafactory; print(llamafactory.__version__)" 2>/dev/null || echo "未知")
- **CUDA 版本**: $(nvcc --version 2>/dev/null | grep release | cut -d',' -f1 || echo "未知")

### 1.3 GPU 数量
- **可见 GPU**: ${CUDA_VISIBLE_DEVICES}
- **GPU 数量**: $(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l) 卡

---

## 2. 训练配置

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| 基座模型 | ${MODEL_PATH} | |
| 对话模板 | ${TEMPLATE} | |
| 微调方式 | LoRA | |
| 训练轮数 | ${NUM_EPOCHS} | |
| 单卡 Batch Size | ${BATCH_SIZE} | 适配长序列显存 |
| 梯度累积步数 | ${ACCUMULATION_STEPS} | 保持全局 Batch 稳定 |
| 全局 Batch Size | $((${BATCH_SIZE} * ${ACCUMULATION_STEPS} * $(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l))) | 4 卡并行 |
| 学习率 | ${LEARNING_RATE} | 降低以防旧任务遗忘 |
| 最大序列长度 | ${CUTOFF_LEN} | 适配 Task3 (Max 1976) |
| LoRA Rank | ${LORA_RANK} | 增加多任务容量 |
| LoRA Alpha | ${LORA_ALPHA} | 随 Rank 缩放 |
| LoRA Dropout | ${LORA_DROPOUT} | 防止过拟合 |
| 目标模块 | ${LORA_TARGET} | |
| Warmup 比例 | ${WARMUP_RATIO} | 长任务需更长热身 |
| 保存步数 | ${SAVE_STEPS} | |
| 采样策略 | ${MIX_STRATEGY} | 4:4:1 样本比 |
| 采样概率 | ${INTERLEAVE_PROBS} | Token 比约 1.5:1 |
| 训练数据集 | ${DATASETS} | 三个独立任务文件 |

---

## 3. 数据集信息

EOF

# 记录数据集信息 (分别统计三个任务)
python3 << DATASET_INFO >> ${REPORT_FILE}
import json
import os

task_files = {
    "Task1": "${TASK1_FILE}",
    "Task2": "${TASK2_FILE}",
    "Task3": "${TASK3_FILE}"
}

print("### 各任务数据文件统计")
print()
print("| 任务 | 文件路径 | 样本数 | 状态 |")
print("| :--- | :--- | :--- | :--- |")

total_samples = 0
task_stats = {}

for task_name, file_path in task_files.items():
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        count = len(data)
        total_samples += count
        task_stats[task_name] = count
        status = "✅"
    else:
        count = 0
        status = "❌ 未找到"
    print(f"| {task_name} | {file_path} | {count:,} | {status} |")

print()
print(f"- **总样本数**: {total_samples:,} 条")
print()

# 计算采样比例
if len(task_stats) == 3:
    t1, t2, t3 = task_stats["Task1"], task_stats["Task2"], task_stats["Task3"]
    
    # 估算 Token
    t1_tokens = t1 * 157
    t2_tokens = t2 * 157
    t3_tokens = t3 * 840
    total_tokens = t1_tokens + t2_tokens + t3_tokens
    
    print("### Token 分布估算")
    print()
    print("| 任务 | 样本数 | 样本占比 | 平均 Token | 估算 Token 占比 |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    print(f"| Task1 | {t1:,} | {t1/total_samples*100:.1f}% | 157 | {t1_tokens/total_tokens*100:.1f}% |")
    print(f"| Task2 | {t2:,} | {t2/total_samples*100:.1f}% | 157 | {t2_tokens/total_tokens*100:.1f}% |")
    print(f"| Task3 | {t3:,} | {t3/total_samples*100:.1f}% | 840 | {t3_tokens/total_tokens*100:.1f}% |")
    print()
    
    short_tokens = t1_tokens + t2_tokens
    long_tokens = t3_tokens
    print(f"- **短任务 Token 总计**: {short_tokens:,} ({short_tokens/total_tokens*100:.1f}%)")
    print(f"- **长任务 Token 总计**: {long_tokens:,} ({long_tokens/total_tokens*100:.1f}%)")
    print()
    
    # 检查是否需要下采样
    if long_tokens > short_tokens * 1.5:
        print("⚠️ **警告**: 长任务 Token 占比过高！")
        print("   建议：在 dataset_info.json 中配置采样权重，或下采样 Task3 数据")
    else:
        print("✅ **Token 比例相对平衡**，配合 4:4:1 采样策略可有效防止遗忘")
    print()
    
    # 样本示例
    print("### 样本示例（各任务前 1 条）")
    print()
    for task_name, file_path in task_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if len(data) > 0:
                item = data[0]
                print(f"**{task_name} 示例**:")
                print(f"- Instruction: {item.get('instruction', '')[:100]}...")
                print(f"- Input: {item.get('input', '')[:50] if item.get('input') else '无'}...")
                print(f"- Output: {item.get('output', '')[:100]}...")
                print()
else:
    print("⚠️ 部分任务文件未找到，请检查配置")
DATASET_INFO

cat >> ${REPORT_FILE} << EOF

---

## 4. 训练过程监控

### 4.1 训练开始时间

**开始时间**: $(date '+%Y-%m-%d %H:%M:%S')

### 4.2 实时日志

训练日志将保存到：\`${LOG_DIR}/training.log\`

### 4.3 显存监控

显存使用日志将保存到：\`${LOG_DIR}/gpu_memory.log\`

---

## 5. 训练结果

EOF

# ==================================================
# 启动显存监控（后台进程）
# ==================================================

echo "启动显存监控..."
(
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S')" >> ${LOG_DIR}/gpu_memory.log
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv >> ${LOG_DIR}/gpu_memory.log
        echo "" >> ${LOG_DIR}/gpu_memory.log
        sleep 30
    done
) &
MONITOR_PID=$!

# 记录开始显存
echo "训练前显存状态:" >> ${LOG_DIR}/gpu_memory.log
nvidia-smi >> ${LOG_DIR}/gpu_memory.log

# ==================================================
# 记录开始时间
# ==================================================

START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

echo "开始时间：${START_TIME_STR}" >> ${LOG_DIR}/training.log
echo "==========================================" >> ${LOG_DIR}/training.log

# ==================================================
# 训练执行
# ==================================================

# 记录开始时间和参数
echo "开始训练：$(date)"
echo "=========================================="
echo "关键参数配置："
echo "模型路径：${MODEL_PATH}"
echo "模板：${TEMPLATE}"
echo "训练轮数：${NUM_EPOCHS}"
echo "批次大小：${BATCH_SIZE}"
echo "梯度累积：${ACCUMULATION_STEPS}"
echo "学习率：${LEARNING_RATE}"
echo "LoRA 秩：${LORA_RANK}"
echo "序列长度：${CUTOFF_LEN}"
echo "采样比例：${INTERLEAVE_PROBS}"
echo "数据集：${DATASETS}"
echo "保存步数：${SAVE_STEPS}"
echo "输出目录：${OUTPUT_DIR}"
echo "=========================================="

# 执行训练命令（同时输出到日志文件）
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path ${MODEL_PATH} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template ${TEMPLATE} \
    --flash_attn auto \
    --dataset_dir ${DATASET_DIR} \
    --dataset ${DATASETS} \
    --cutoff_len ${CUTOFF_LEN} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_samples ${MAX_SAMPLES} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps ${SAVE_STEPS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --packing False \
    --enable_thinking False \
    --report_to none \
    --output_dir ${OUTPUT_DIR} \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target ${LORA_TARGET} \
    --seed 42 \
    --mix_strategy ${MIX_STRATEGY} \
    --interleave_probs ${INTERLEAVE_PROBS} \
    2>&1 | tee ${LOG_DIR}/training.log

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# 停止显存监控
kill $MONITOR_PID 2>/dev/null || true

# ==================================================
# 记录结束信息
# ==================================================

END_TIME=$(date +%s)
END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

# 计算训练步数（从日志中提取）
TOTAL_STEPS=$(grep -c "Step:" ${LOG_DIR}/training.log 2>/dev/null || echo "未知")
FINAL_LOSS=$(grep "loss" ${LOG_DIR}/training.log | tail -1 | grep -oP 'loss=[\d.]+' | head -1 || echo "未知")

# 停止显存监控
echo "" >> ${LOG_DIR}/gpu_memory.log
echo "训练结束显存状态:" >> ${LOG_DIR}/gpu_memory.log
nvidia-smi >> ${LOG_DIR}/gpu_memory.log

# ==================================================
# 生成训练结果报告
# ==================================================

cat >> ${REPORT_FILE} << EOF

### 5.1 训练状态

| 项目 | 值 |
| :--- | :--- |
| 开始时间 | ${START_TIME_STR} |
| 结束时间 | ${END_TIME_STR} |
| 总耗时 | ${DURATION_MIN} 分 ${DURATION_SEC} 秒 |
| 训练步数 | ${TOTAL_STEPS} |
| 最终 Loss | ${FINAL_LOSS} |
| 退出代码 | ${TRAIN_EXIT_CODE} |

### 5.2 训练 Loss 曲线

训练 Loss 曲线图已保存到：\`${OUTPUT_DIR}/plots/loss_plot.png\`

### 5.3 显存使用统计

EOF

# 分析显存使用
python3 << GPU_ANALYSIS >> ${REPORT_FILE}
import os

log_file = "${LOG_DIR}/gpu_memory.log"

if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    memory_used = []
    for line in lines:
        if 'MiB' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    mem = int(parts[1].strip().replace(' MiB', ''))
                    memory_used.append(mem)
                except:
                    pass
    
    if memory_used:
        print(f"- **峰值显存**: {max(memory_used)} MiB")
        print(f"- **平均显存**: {sum(memory_used)//len(memory_used)} MiB")
        print(f"- **最低显存**: {min(memory_used)} MiB")
    else:
        print("- 显存数据解析失败")
else:
    print("- 显存日志文件不存在")
GPU_ANALYSIS

cat >> ${REPORT_FILE} << EOF

### 5.4 保存的 Checkpoint

EOF

# 列出保存的 checkpoint
ls -lh ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | awk '{print "- "$9" ("$5")"}' >> ${REPORT_FILE} || echo "- 未找到 checkpoint" >> ${REPORT_FILE}

cat >> ${REPORT_FILE} << EOF

---

## 6. 模型评估

### 6.1 推理测试

建议在合并 LoRA 后进行推理测试，记录以下指标：

- [ ] 任务 1 准确率
- [ ] 任务 2 准确率
- [ ] 任务 3 准确率
- [ ] 平均准确率

### 6.2 与基座模型对比

| 指标 | 基座模型 | 微调后模型 | 提升 |
| :--- | :--- | :--- | :--- |
| 任务 1 准确率 | - | - | - |
| 任务 2 准确率 | - | - | - |
| 任务 3 准确率 | - | - | - |
| 平均准确率 | - | - | - |

---

## 7. 经验总结

### 7.1 成功经验

- [x] 数据预处理：4:4:1 采样平衡 Token 分布
- [x] 参数调优：3 Epochs 避免过拟合
- [x] 批处理策略：小 Batch + 大梯度累积
- [x] 序列长度：3072 适配长任务
- [x] 多数据集配置：三个独立任务文件混合训练

### 7.2 待改进点

- [ ] 任务数据分布均衡性
- [ ] 学习率调度策略
- [ ] 验证集监控 (当前无验证集)

### 7.3 面试阐述要点

1. **项目背景**：多任务 SFT 微调，解决长短任务梯度不平衡问题
2. **技术方案**：LoRA 高效微调 + 4:4:1 采样 + 任务平衡
3. **关键指标**：训练 Loss 下降曲线、各任务准确率提升
4. **工程实践**：4 卡并行训练、显存监控、实验报告自动化

---

## 8. 附录

### 8.1 dataset_info.json 配置示例

确保 \`${DATASET_DIR}/dataset_info.json\` 包含以下配置：

\`\`\`json
{
  "task1_train": {
    "file_name": "task1_train.json"
  },
  "task2_train": {
    "file_name": "task2_train.json"
  },
  "task3_train": {
    "file_name": "task3_train.json"
  }
}
\`\`\`

### 8.2 完整训练命令

\`\`\`bash
llamafactory-cli train \\
    --stage sft \\
    --do_train True \\
    --model_name_or_path ${MODEL_PATH} \\
    --finetuning_type lora \\
    --template ${TEMPLATE} \\
    --dataset_dir ${DATASET_DIR} \\
    --dataset ${DATASETS} \\
    --mix_strategy ${MIX_STRATEGY} \\
    --interleave_probs ${INTERLEAVE_PROBS} \\
    --cutoff_len ${CUTOFF_LEN} \\
    --num_train_epochs ${NUM_EPOCHS} \\
    --per_device_train_batch_size ${BATCH_SIZE} \\
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \\
    --learning_rate ${LEARNING_RATE} \\
    --lora_rank ${LORA_RANK} \\
    --output_dir ${OUTPUT_DIR}
\`\`\`

### 8.3 相关文件路径

| 文件 | 路径 |
| :--- | :--- |
| 实验报告 | ${REPORT_FILE} |
| 训练日志 | ${LOG_DIR}/training.log |
| 显存日志 | ${LOG_DIR}/gpu_memory.log |
| Loss 曲线 | ${OUTPUT_DIR}/plots/loss_plot.png |
| 模型权重 | ${OUTPUT_DIR}/ |
| Task1 数据 | ${TASK1_FILE} |
| Task2 数据 | ${TASK2_FILE} |
| Task3 数据 | ${TASK3_FILE} |

---

**报告生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
EOF

# ==================================================
# 检查训练结果
# ==================================================

echo ""
echo "=========================================="
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "✅ 训练完成：$(date)"
    echo "=========================================="
    echo "模型保存在：${OUTPUT_DIR}"
    echo ""
    echo "📊 实验报告：${REPORT_FILE}"
    echo "📝 训练日志：${LOG_DIR}/training.log"
    echo "💾 显存日志：${LOG_DIR}/gpu_memory.log"
    echo ""
    echo "下一步操作："
    echo "1. 查看实验报告：cat ${REPORT_FILE}"
    echo "2. 检查 Token 平衡：查看报告中'数据集信息'章节"
    echo "3. 合并 LoRA 模型："
    echo "   llamafactory-cli export \\"
    echo "       --model_name_or_path ${MODEL_PATH} \\"
    echo "       --adapter_name_or_path ${OUTPUT_DIR} \\"
    echo "       --output_dir /path/to/merged \\"
    echo "       --finetuning_type lora"
    echo ""
    echo "4. 运行推理测试："
    echo "   bash inference.sh"
else
    echo "❌ 训练失败：$(date)"
    echo "=========================================="
    echo "请检查日志：${LOG_DIR}/training.log"
    exit 1
fi

echo "=========================================="
