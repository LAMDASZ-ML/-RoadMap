## 本科生科研入门学习指南（更新时间：2026.02.11）

### 前言

* **课题组负责人**：[郭兰哲](https://www.lamda.nju.edu.cn/guolz/) (南京大学智能科学与技术学院，准聘助理教授，博士生导师)
* **研究方向**：神经符号学习 (Neuro-Symbolic Learning)、大模型推理 (LLM Reasoning)、智能体 (Agent)
* **招生对象**：直博生、硕士生、科研实习生（支持 Remote）
* **联系方式**：欢迎感兴趣的同学联系 [guolz@nju.edu.cn](mailto:guolz@nju.edu.cn)。
    * *邮件标题建议注明*：`[科研实习/保研/直博申请] 姓名-学校-年级-专业`

### 学习目标

本学习大纲面向从 0 到 1 入门的本科生，目标是在 **6-8 周** 内，帮助同学建立对课题组研究方向的整体认知、核心技术理解与基础科研能力，为后续参与真实科研项目打下基础。

* **基础夯实**：了解人工智能与大模型基础知识，能够自主检索并读懂前沿论文与代码。
* **领域认知**：理解什么是大模型/多模态推理、LLM Agent、Neuro-Symbolic Learning。
* **科研素养**：具备“提出问题 -> 文献检索 -> 代码复现 -> 实验分析”的完整科研闭环能力。

---

### 阶段 1：人工智能与大模型基础（进组考核内容）

> **说明**：此阶段为“准入资格”学习。在正式加入课题组参与科研实习之前，你应当具备人工智能、大模型的基础知识。你可以参照下面的大纲，结合提供的参考资料，或者网上其他优质资料进行学习
>
> 如果你认为自己已具备相关能力，便可约时间进行入组考核

#### 1. 神经网络与深度学习基础

**学习目标**：
* 掌握神经网络的基本原理，能够理解前向传播与反向传播的数学推导与代码实现
* 掌握 CNN/ResNet、RNN 等机器学习基础模型架构
* 掌握 Pytorch 核心组件的使用，例如 Dataloader、损失函数、模型搭建、优化器等

**参考资料**：
* [动手学深度学习](https://zh.d2l.ai/)
* [Neural Networks: Zero to Hero (Andrej Karpathy)](https://karpathy.ai/zero-to-hero.html)
* [PyTorch Tutorial](https://pytorch.org/tutorials/)

#### 2. 大语言模型（LLMs）

**学习目标**：
* 掌握 Transformer 的核心机制 (Self-Attention, Positional Encoding, Decoder-only vs Encoder-Decoder 等)
* 了解 GPT 系列、LLaMA 系列等典型大语言模型
* 了解基础的 Prompt Engineering (Zero-shot, Few-shot) 及 API 调用方式
* 了解 Chain-of-Thought 等大模型推理机制

**参考资料**：
* [Happy-LLM (Datawhale)](https://datawhalechina.github.io/happy-llm/)
* [Prompt Engineering Guide](https://www.promptingguide.ai/)

#### 3. 视觉与多模态大模型

**学习目标**：
* 了解 Vision Transformer (ViT) 的基本原理
* 了解 LLaVA、Qwen-VL 系列等前沿多模态大模型

**参考资料**：
* [ViT 论文](https://arxiv.org/abs/2010.11929) | [代码库](https://github.com/lucidrains/vit-pytorch)
* [LLaVA 论文](https://arxiv.org/abs/2304.08485)
* [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
* 多模态大模型论文串讲：[上](https://www.bilibili.com/video/BV1Vd4y1v77v/?spm_id_from=333.337.search-card.all.click&vd_source=4deb09ae020d1de21482612e7102fb83)，[下](ttps://www.bilibili.com/video/BV1fA411Z772/?share_source=copy_web&vd_source=b19e968ea4fbdf0cb9dfd7fbc468280e)
---

**考核方式**：
完成上述基础知识学习之后，可以约时间进行交流（线下或线上会议），无需准备 PPT 等材料，交流方式为面试提问。主要围绕基础概念的理解，不会过多关注算法细节的记忆，通过后即可作为科研实习生加入课题组

---

### 阶段 2：文献阅读与代码实践（科研入门培训）

> **说明**：进入此阶段，你已经正式开始科研训练。本阶段重点在于论文调研、阅读、复现与思考

**预备工作**：
1.  请自行搜索，了解什么是 arXiv, HuggingFace, Google Scholar, DBLP
2.  了解 ICML, NeurIPS, ICLR 等人工智能顶级会议, 具备根据某个 topic 检索相关论文的能力

**核心流程**：
1.  **了解方向**：了解本组的主要科研方向（Neuro-Symbolic Learning，LLM Reasoning，Agent）。
2.  **选择题目**：结合自己的兴趣，选择一个方向
3.  **实践汇报**：完成相应的论文阅读以及代码实践，并形成 PPT 汇报

> **本训练计划并非考核某个固定答案，而是帮助你判断：**
> 你是否真正享受分析问题、阅读论文、调试代码和反思实验的过程。
> 如果你对“研究问题本身”感到兴奋，那么欢迎加入我们。

#### 研究方向初步认知 (建议阅读)

* **神经符号学习 (Neuro-Symbolic Learning)**
    * [Neuro-Symbolic Learning in the era of Large Models](https://www.lamda.nju.edu.cn/guolz/paper/AAAI_Logic_AI_Keynote.pdf)
    * [Neuro-Symbolic Artificial Intelligence: Towards Improving the Reasoning Abilities of Large Language Models](https://arxiv.org/abs/2508.13678)
* **大模型推理与规划 (LLM/MLLM Reasoning & Planning)**
    * [Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models](https://arxiv.org/abs/2503.09567)
    * [Perception, Reason, Think, and Plan: A Survey on Large Multi-Modal Reasoning Models](https://arxiv.org/pdf/2505.04921)
* **智能体 (Agent)**
    * [Agent AI: Surveying the Horizons of Multi-Modal Interaction](https://arxiv.org/pdf/2401.03568)
    * [A Survey on Agentic Multi-Modal Large Language Models](https://arxiv.org/pdf/2510.10991)
    * [Agentic Reasoning for Large Language Models](https://arxiv.org/abs/2601.12538)
    * [《从零开始构建智能体》](https://github.com/datawhalechina/hello-agents/tree/main)

#### 选题实战（任选其一）

请结合个人兴趣选择一个方向，检索并阅读相关论文并完成代码实践

**⚠️ 注意事项**：
* 不要求完整复现论文全部实验，但需要在至少 1 个数据集上跑通完整算法流程，并得到结果
* 尽量复用开源框架（如 `LLaMA-Factory`, `TRL`, `LangChain` 等），重点在于掌握算法流程以及分析实验结果，而非重复造轮子
* 若算力受限，请优先使用 PEFT (LoRA/QLoRA) 或小参数量模型（如 Qwen-2.5-1.5B/3B）
* 如果在实践过程中未能复现论文中的性能结果，并不视为失败；请尝试定位问题来源、分析原因并给出合理解释
* 如果要复现自行查找的论文，请优先选择近两年 (2025年后) 在顶会 (ICML/NeurIPS/ICLR) 上发表的，或具有较高影响力 (Citation>100) 的文章

---

#### 方向 1：数学、几何推理
* [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
* [SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models](https://arxiv.org/pdf/2504.11468)
* [SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://arxiv.org/abs/2501.17161)
* [Neuro-Symbolic Data Generation for Math Reasoning](https://arxiv.org/abs/2412.04857)
* [NeSyGeo: A Neuro-Symbolic Framework for Multimodal Geometric Reasoning Data Generation](https://arxiv.org/abs/2505.17121)(一作是智科院 2023 级本科生，大二期间完成)

> **🎯 实践任务**：
> 基于 SFT 或 GRPO 算法微调一个开源大语言模型或者多模态大模型（例如 Qwen-Math 系列），在一个数学推理数据集（例如 GSM8K、MATH、MathVista 等）进行评测，分析微调前后模型的性能变化
> *(注：SFT为必选任务，GRPO如果跑不起来，可以只掌握原理/代码实现)*

#### 方向 2：视觉图像推理
* [Visual Programming: Compositional visual reasoning without training](https://arxiv.org/pdf/2211.11559)
* [DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362)
* [Thyme: Think Beyond Images](https://arxiv.org/abs/2508.11630)
* [Thinking with Images for Multi-Modal Reasoning: Foundations, Methods, and Future Frontiers](https://arxiv.org/pdf/2506.23918)

> **🎯 实践任务**：
> 调研 "Think with Images" 方向的论文，尝试在一个视觉推理数据集上，复现一种方法，并进行结果分析
> *(注：如果算力不允许可以优先选择无需训练的方法)*

#### 方向 3：抽象视觉推理
* [ARC Challenge](https://arcprize.org/)
* [NSA: Neuro-symbolic ARC Challenge](https://arxiv.org/abs/2501.04424)

> **🎯 实践任务**：
> 了解什么是 ARC Challenge，调研相应的论文与解决方案，并尝试实现至少一种方法，分析其结果和瓶颈

#### 方向 4：Travel Agent (Tool-Use & Planning)
* [TravelPlanner: A Benchmark for Real-World Planning with Language Agents](https://arxiv.org/abs/2402.01622?)
* [ChinaTravel: An Open-Ended Benchmark for Language Agents in Chinese Travel Planning](https://arxiv.org/pdf/2412.13682)
* [Mind the Gap to Trustworthy LLM Agents: A Systematic Evaluation on Constraint Satisfaction for Real-World Travel Planning](https://openreview.net/pdf?id=SXKIaWTe4N) (**AAAI 2026 Trust Agent Workshop Best Student Paper, 前两位作者均为智科院本科生**)

> **🎯 实践任务**：
> 基于 ReAct 框架构建一个简单的 Agent，分析其在上述两个数据集中的性能表现

#### 方向 5：Game/Embodied Agent
* [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
* [WALL-E 2.0: World Alignment by NeuroSymbolic Learning improves World Model-based LLM Agents](https://arxiv.org/pdf/2504.15785)
* [InstructFlow: Adaptive Symbolic Constraint-Guided Code Generation for Long-Horizon Planning](https://openreview.net/pdf?id=nzwjvpCO4F)
* [Re2 Agent: Reflection and Re-execution Agent for Embodied Decision Making](https://openreview.net/pdf?id=nHhOvYrPMf) (NeurIPS 2025 EAI Challenge Most Innovative Approach)

> **🎯 实践任务**：
> 参考上面的论文，在我的世界 (MineCraft) 环境或具身数据集 ALFWorld 中进行实验，并汇报性能结果。
> *(注：MineCraft 相对来说环境更为复杂，且对模型能力要求较高，请根据自身工程能力选择)*

#### 方向 6：Symbolic Regression
* Tutorial: https://symbolicregression2025.github.io/

> **🎯 实践任务**：
> 基于上述 Tutorial，阅读相关论文，尝试复现论文 [LLM-SR: Scientific Equation Discovery via Programming with Large Language Models](https://openreview.net/forum?id=m2nmp8P5in)，根据论文给出的 Github 仓库跑通代码，并对比与原文中的结果。

---

**💡 自定义方向**
如果你对其他隶属于 Neuro-Symbolic Learning、Agent、LLM Reasoning 领域的研究方向感兴趣（比如多模态医学推理、遥感图像推理、Chart QA、智慧司法、或者我的世界之外的其他游戏场景等），也可以提前进行沟通，得到允许之后，可以自行发挥查阅相关文献，按对等要求完成（即复现至少 1 篇论文算法在 1 个数据集上的实验结果）

---

### 3. 汇报要求

完成上述任务之后，需要准备一份 PPT 进行汇报，内容应包含：

1.  **领域背景简介**：该方向主要解决什么科学问题？为什么重要？
2.  **代表方法介绍**：有哪些代表性的方法，核心思路是什么？（**尽量用自己的语言简洁叙述，不要照搬原文**）
3.  **实验结果分析**：实验设置、实验结果、分析讨论等
4.  **未来思考**：基于你的实践，你认为该方向下一步可以做什么？

**PPT 制作基本原则**：
* PPT 可以用中文或英文制作
* 不需要设置动画，导出为 **PDF 格式**
* 字体建议：中文使用微软雅黑，英文使用 Times New Roman
* 涉及到参考文献的需要添加引用，参考文献放在本页PPT的最下方
* 可以使用 Powerpoint，也可以使用 LaTeX，以文档排版美观、易于阅读为最终目标

---
*关于本文档的任何问题，欢迎留言或邮件咨询。*