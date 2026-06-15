"""
知识蒸馏（Knowledge Distillation）完整示例
===========================================
这个脚本展示如何用教师模型的知识训练学生模型

核心概念：
- Teacher: 大模型，知识丰富但推理慢
- Student: 小模型，推理快但知识少
- Soft Labels: 教师输出的概率分布（保留更多信息）
- Temperature: 控制分布"软度"的参数

运行前安装: pip install torch transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 第一部分：温度缩放 Softmax（蒸馏核心）
# ============================================================

def temperature_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    带温度的 Softmax

    Args:
        logits: 模型原始输出 [batch_size, vocab_size]
        temperature: 温度参数
            - T = 1.0: 标准 softmax
            - T > 1.0: 分布更平滑（更"软"），概率差距缩小
            - T < 1.0: 分布更尖锐（更"硬"），概率差距放大

    Returns:
        概率分布 [batch_size, vocab_size]
    """
    #这里是将温度当作调节器，控制教师输出的概率分布的平滑程度
    return F.softmax(logits / temperature, dim=-1)


# ============================================================
# 第二部分：蒸馏损失函数
# ============================================================

#nn.Module是干啥的
class DistillationLoss(nn.Module):
    """
    知识蒸馏损失

    组成两部分：
    1. 软标签损失：学生模仿教师的概率分布（KL 散度）
    2. 硬标签损失：学生预测真实标签（交叉熵）

    Loss = α * 软标签损失 + (1-α) * 硬标签损失
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.7,  # 软标签权重
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """
        计算蒸馏损失

        Args:
            student_logits: 学生模型输出 [batch, seq_len, vocab_size]
            teacher_logits: 教师模型输出 [batch, seq_len, vocab_size]
            labels: 真实标签 [batch, seq_len]

        Returns:
            total_loss, soft_loss, hard_loss
        """
        
        # 忽略 padding token
        active_mask = labels != -100

        # 1. 软标签损失（KL 散度）
        # 学生要模仿教师的"思考方式",
        soft_targets = temperature_softmax(teacher_logits, self.temperature)
        # 学生的数据在这里相当于Q，  D_KL(P||Q) = 教师概率 × log(教师概率 / 学生概率)  
        # 所以学生数据需要softmax化
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

        # KL 散度损失，乘以 T^2 保持梯度量级一致
        # D_KL(P||Q) = 教师概率 × log(教师概率 / 学生概率)    = 平均惊讶 - 真实惊讶
        """
              D_KL = sum(P * log P) - sum(P * log Q)
                = -熵 - (-交叉熵)                                                                                                           
                = 交叉熵 - 熵
                = 平均惊讶 - 真实惊讶   
        """
        soft_loss = F.kl_div(
            soft_student,
            soft_targets,
            reduction="none"
        ) * (self.temperature ** 2)

        # 只计算非 padding 位置
        soft_loss = soft_loss.sum(dim=-1)  # [batch, seq_len]
        soft_loss = (soft_loss * active_mask).sum() / active_mask.sum()

        # 2. 硬标签损失（交叉熵）
        # 学生也要学会正确答案，只针对学生
        hard_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
        )

        # 3. 总损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss, soft_loss, hard_loss


# ============================================================
# 第三部分：蒸馏训练主流程
# ============================================================


def main():
    print("=" * 60)
    print("知识蒸馏演示")
    print("=" * 60)

    # ---------- 配置 ----------
    teacher_model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 教师（大模型）
    student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 学生（这里用同个模型演示）

    # 实际场景中：
    # teacher = "Qwen/Qwen2.5-7B-Instruct"    # 7B 教师
    # student = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B 学生

    # 判断设备的这个方法还挺好用的
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n设备: {device}")

    # ---------- 加载 Tokenizer ----------
    print("\n[1/5] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   词表大小: {len(tokenizer)}")

    # ---------- 加载模型 ----------
    print("\n[2/5] 加载模型...")

    # 教师（冻结，不更新）
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device) #注意from_pretrained在加载的时候可以使用to制定设备
    teacher.eval()  # 冻结，所谓冻结就是把模型设置为评估模型，不计算梯度
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"   教师模型: {teacher_model_name} (冻结)")


    # 学生（会更新）
    student = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    student.train() # 将学生设置为train模式，启用梯度计算
    print(f"   学生模型: {student_model_name} (训练中)")

    # ---------- 准备数据 ----------
    print("\n[3/5] 准备训练数据...")

    training_texts = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的核心，它使计算机能够从数据中学习而无需明确编程。",
        "深度学习使用多层神经网络来学习数据的表示，在图像和语音识别方面表现出色。",
        "自然语言处理使计算机能够理解、解释和生成人类语言。",
        "强化学习是一种机器学习方法，智能体通过与环境交互来学习最优策略。",
    ]

    # Tokenize
    encodings = tokenizer(
        training_texts,
        padding=True, # 需要padding
        truncation=True,# 需要截断
        max_length=64, # 限制输入长度，避免内存过大
        return_tensors="pt", # 返回PyTorch张量
    )

    # 准备标签（对于语言模型，标签 = 输入）
    input_ids = encodings["input_ids"].to(device)
    # 这个attention_mask是用来标记哪些位置是padding的，计算损失时会忽略这些位置
    attention_mask = encodings["attention_mask"].to(device)
    labels = input_ids.clone()
    # 把 padding 位置的标签设为 -100（忽略）
    labels[attention_mask == 0] = -100

    print(f"   数据量: {len(training_texts)} 条")
    print(f"   序列长度: {input_ids.size(1)}")

    # ---------- 初始化训练组件 ----------
    print("\n[4/5] 初始化训练组件...")

    distill_loss = DistillationLoss(temperature=2.0, alpha=0.7)
    #通过动量训练吗？AdamW是一个常用的优化器，适合训练语言模型
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)

    print(f"   温度: {distill_loss.temperature}")
    print(f"   软标签权重 α: {distill_loss.alpha}")

    # ---------- 训练循环 ----------
    print("\n[5/5] 开始蒸馏训练...")
    print("-" * 60)

    num_epochs = 3

    for epoch in range(num_epochs):
        # 获取教师输出（不需要梯度）
        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_outputs.logits

        # 学生前向传播
        student_outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = student_outputs.logits

        # 计算蒸馏损失
        total_loss, soft_loss, hard_loss = distill_loss(
            student_logits, teacher_logits, labels
        )

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 打印统计
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")
        print(f"  ├─ Total Loss: {total_loss.item():.4f}")
        print(f"  ├─ Soft Loss:  {soft_loss.item():.4f} (模仿教师)")
        print(f"  └─ Hard Loss:  {hard_loss.item():.4f} (正确答案)")

    print("\n" + "=" * 60)

    # ---------- 对比测试 ----------
    print("\n测试：教师 vs 学生")
    print("-" * 60)

    test_text = "深度学习是一种"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    student.eval()
    with torch.no_grad():
        # 教师生成
        teacher_output = teacher.generate(**inputs, max_new_tokens=30, do_sample=False)
        teacher_text = tokenizer.decode(teacher_output[0], skip_special_tokens=True)

        # 学生生成
        student_output = student.generate(**inputs, max_new_tokens=30, do_sample=False)
        student_text = tokenizer.decode(student_output[0], skip_special_tokens=True)

    print(f"\n  输入: {test_text}")
    print(f"  教师: {teacher_text}")
    print(f"  学生: {student_text}")

    # ---------- 保存模型 ----------
    save_path = "./distilled_student"
    student.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n学生模型已保存到: {save_path}")



if __name__ == "__main__":

    # 3. 运行蒸馏训练
    main()
