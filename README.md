# 新同学进组学习指南-大模型/多模态推理与规划

### 请注意，并非所有的任务都要逐一完成，如果你觉得对于你来说过于简单，可以直接跳过~

## Stage 1：神经网络->大语言模型->多模态大模型

### 1. 神经网络与深度学习基础
  1. 掌握神经网络的基本原理，能够理解前向传播与反向传播的数学推导与代码实现
  2. 了解Pytorch核心组件的使用，例如常用损失函数、优化方法等
  3. 参考资料：
     1. 李沐，[动手学深度学习](https://zh.d2l.ai/)
     2. Andrej Karpathy，[Neural Networks:Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
     3. [Pytorch Tutorial](https://docs.pytorch.org/tutorials/intro.html)
  4. 实践任务：基于PyTorch 从零实现一个简单的多层感知机 (MLP)，完成 MNIST 手写数字分类任务
   
### 2. 常见深度神经网络架构
   1. 掌握CNN/ResNet、RNN/Transformer等机器学习基础模型架构
   2. 理解诸如残差连接、注意力机制、位置编码等核心思想
   3. 掌握GPT系列、LLaMA系列等大语言模型架构
   4. 参考资料：
      1. 《动手学深度学习》中的相关内容
      2. Happy-LLM 第二-五章：https://datawhalechina.github.io/happy-llm/ 
      3. Andrej Karpathy 课程关于Transformer与LLM部分
1. 实践任务：参考Happy-LLM相应章节，基于Pytorch手动搭建Transformer模型，无需完成训练，了解Transformer如何通过代码构建即可

### 3. Vision Transformer (ViT)
  1. 掌握Vision Transformer (ViT)的基本原理
  2. 参考资料：
      1. ViT论文：https://arxiv.org/abs/2010.11929
      2. 参考代码库：https://github.com/lucidrains/vit-pytorch
   3. 实践任务【非必须，了解ViT原理为主】：基于Pytorch构建ViT模型，尝试面向一个图像分类任务，例如Image-Net子集、CIFAR-10等，进行训练/微调与评测

### 4. 视觉语言模型CLIP
  1. 掌握基础视觉-语言模型CLIP的基本原理
  2. 参考资料：
     1. CLIP论文：https://arxiv.org/abs/2103.00020
     2. 官方代码：https://github.com/openai/CLIP
  4. 实践任务【非必须，了解CLIP原理为主】：基于预训练的CLIP完成一个图像分类数据集的分类任务，例如ImageNet，尝试替换不同的encoder，并对比分析结果

### 5. 多模态大模型前沿
  1. 掌握LLaVA、Qwen-VL系列前沿多模态大模型架构，了解并追踪其他常用多模态大模型架构
  2. 学习如何查找论文、技术报告，使用Huggingface中的模型代码等
  3. 参考资料：
     1. LLaVA论文：https://arxiv.org/abs/2304.08485
     2. Qwen3-VL Technical Report：https://arxiv.org/abs/2511.21631
     3. 多模态大模型论文串讲：[上](https://www.bilibili.com/video/BV1Vd4y1v77v/?spm_id_from=333.337.search-card.all.click)，[下](https://www.bilibili.com/video/BV1fA411Z772/?spm_id_from=333.337.search-card.all.click)
   4. 实践任务【必须】：选择一个典型的多模态推理或规划任务，例如几何数学推理、视觉问答、空间推理、视觉规划、具身规划、游戏智能体规划，尝试运行至少一个多模态大模型，获得评测结果；此部分可形成简要的PPT汇报
   5. 常用数据集示例
      - 视觉推理：
        - 几何数学推理：[MathVista](https://mathvista.github.io/), [MathVision](https://mathllm.github.io/mathvision/), [We-Math](https://we-math2.github.io/)
        - 图像逻辑推理：[VisuLogic](https://arxiv.org/pdf/2504.15279), [LogicVista](https://arxiv.org/pdf/2407.04973), [ARC-AGI](https://arcprize.org/arc-agi)
      - 空间推理：[SpatialScore](https://haoningwu3639.github.io/SpatialScore/), [商汤整合的空间推理benchmark及模型评测](https://arxiv.org/pdf/2508.13142)
      - 游戏智能体：[我的世界](https://minedojo.org/)
      - 具身智能体：[Embodied-Bench](https://arxiv.org/pdf/2502.09560), [ALFWorld](https://alfworld.github.io/)

### 6. 大模型微调算法
1. 掌握大语言模型、多模态大模型的常用微调算法，例如SFT、PPO/DPO/GRPO等
2. 参考资料：
    1. Happy-LLM 第六章：https://datawhalechina.github.io/happy-llm/ 
    2. Reinforcement Learning for LLM Reasoning Survey: https://arxiv.org/pdf/2509.08827 
3. 实践任务：阅读论文[SFT or RL](https://arxiv.org/pdf/2504.11468)，[SFT Memorize, RL Generalize](https://arxiv.org/abs/2501.17161), 尝试基于SFT与RL算法微调一个开源多模态大模型

有意提前进组的同学完成任务5后即可进一步交流研究方向。

## Stage 2：神经符号推理基础
### 1. 基础符号推理工具
   - 学习SMT Solver: [Z3](https://ericpony.github.io/z3py-tutorial/guide-examples.htm), 能够使用Z3编写代码求解数学推理问题
   - 学习Prolog：[Prolog](https://www.swi-prolog.org/GetStarted.html), 能够基于Prolog编写代码完成逻辑推理问题
### 2. 神经符号推理基础
  - 学习如何将神经网络与符号推理结合，尝试理解神经符号推理与端到端神经网络的区别
  - 实践任务1：基于神经符号的视觉推理
    - 阅读论文
      - [Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding](https://arxiv.org/abs/1810.02338)
      - [The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision](https://arxiv.org/abs/1904.12584)
    - 尝试在CLEVR数据集中实现基于神经符号的视觉问答方案
      - 参考资料：[Github: Neuro Symbolic VQA](https://github.com/nerdimite/neuro-symbolic-ai-soc)
  - 实践任务2：基于神经符号的具身规划
    - 阅读论文
      - [LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477)
      - [WALL-E 2.0: World Alignment by Neuro-Symbolic Learning improves World Model-based LLM Agents](https://arxiv.org/pdf/2504.15785)
    - 尝试在游戏或者具身环境中复现上述论文的技术方案
      - 参考代码：[Github: LLM+P](https://github.com/Cranial-XIX/llm-pddl)

## Stage 3：综述论文
1. 阅读相关领域综述，了解前沿进展，形成对该领域的整体认识
2. 参考综述：
    1. Agent:
      - Agent AI: https://arxiv.org/pdf/2401.03568 
      - Multi-Modal Agent: https://arxiv.org/pdf/2510.10991
      - VisualAgentBench: https://arxiv.org/abs/2408.06327
    2. 多模态推理：
      - Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models https://arxiv.org/pdf/2505.04921
      - Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers https://arxiv.org/pdf/2506.23918
      - Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey https://arxiv.org/pdf/2503.12605
    3. 神经符号融合：
      - Neuro-Symbolic Artificial Intelligence: Towards Improving the Reasoning Abilities of Large Language Models: https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/8905.pdf
  1. 任务：以综述论文为纲，阅读学习相关论文，完成一个PPT汇报，重在凝练形成整体认识，无需介绍方法细节。
  2. **目标：能够从技术和问题两个维度，形成对该领域的分类与认识，即这个方向主要有哪些科学问题？针对如何解决这些科学问题，有哪些相应的主流技术方案？**

## Stage 4：讨论确定具体方向，大量论文阅读、复现与改进
