# 新同学进组学习指南-大模型/多模态推理与规划

### 请注意，并非所有的任务都要逐一完成，仅任务6为必须，如果你觉得对于你来说过于简单，可以直接跳过~

## Stage 1：基础知识

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
   3. 参考资料：
      1. 《动手学深度学习》中的相关内容
      1. Happy-LLM 第二-五章：https://datawhalechina.github.io/happy-llm/ 
      2. Andrej Karpathy 课程关于Transformer与LLM部分
5. 实践任务：参考Happy-LLM相应章节，基于Pytorch手动搭建Transformer模型，无需完成训练，了解Transformer如何通过代码构建即可

### 3. 大语言模型实践
1. 掌握GPT系列、LLaMA系列大语言模型架构
2. 掌握大语言模型的常用微调算法，例如LoRA、SFT、PPO/DPO/GRPO等
3. 参考资料：
    1. HuggingFace Transformer库：https://huggingface.co/docs/transformers/index
    2. LLaMA-Factory：https://github.com/hiyouga/LLaMA-Factory
    3. Happy-LLM 第六章：https://datawhalechina.github.io/happy-llm/ 
4. 实践任务【非必须，了解相应框架即可】：基于HuggingFace Transformer库或LLaMA-Factory，完成一个开源大模型在某个垂域任务上的有监督微调与评测（例如司法、金融、数学等，相应数据集可自行搜索）

### 4. Vision Transformer (ViT)
  1. 掌握Vision Transformer (ViT)的基本原理
  2. 参考资料：
      1. ViT论文：https://arxiv.org/abs/2010.11929
      2. 参考代码库：https://github.com/lucidrains/vit-pytorch
   3. 实践任务：基于Pytorch构建ViT模型，尝试面向一个图像分类任务，例如Image-Net子集、CIFAR-10等，进行训练/微调与评测

### 5. 视觉语言模型CLIP
  1. 掌握基础视觉-语言模型CLIP的基本原理
  2. 参考资料：
     1. CLIP论文：https://arxiv.org/abs/2103.00020
     2. 官方代码：https://github.com/openai/CLIP
  4. 实践任务：基于预训练的CLIP完成一个图像分类数据集的分类任务，例如ImageNet，尝试替换不同的encoder，并对比分析结果

### 6. 多模态大模型前沿
  1. 掌握LLaVA、Qwen-VL系列前沿多模态大模型架构，了解并追踪其他常用多模态大模型架构
  2. 学习如何查找论文、技术报告，使用Huggingface中的模型代码等
  3. 参考资料：
     1. LLaVA论文：https://arxiv.org/abs/2304.08485
     2. Qwen2.5-VL Technical Report：https://arxiv.org/abs/2502.13923
     3. 多模态大模型论文串讲：[上](https://www.bilibili.com/video/BV1Vd4y1v77v/?spm_id_from=333.337.search-card.all.click)，[下](https://www.bilibili.com/video/BV1fA411Z772/?spm_id_from=333.337.search-card.all.click)
   4. 实践任务【必须】：选择一个典型的多模态推理或规划任务，例如几何数学推理、视觉问答、空间推理、视觉规划、具身规划、游戏智能体规划，尝试运行至少一个多模态大模型，获得评测结果；此部分可形成简要的PPT汇报
   5. 常用数据集示例
      - 几何数学推理：[MathVista](https://mathvista.github.io/)，[MathVision](https://mathllm.github.io/mathvision/)，[We-Math](https://we-math2.github.io/)
      - 视觉问答推理：
        - 高分辨率视觉问答：[V* Bench](https://vstar-seal.github.io/)
        - 遥感图像问答：[RSVQA](https://arxiv.org/abs/2003.07333)
        - 逻辑推理图像问答：[VisuLogic](https://arxiv.org/pdf/2504.15279), [LogicVista](https://arxiv.org/pdf/2407.04973), [ARC-AGI](https://arcprize.org/arc-agi)
        - 空间推理：[商汤整合的空间推理benchmark及模型评测](https://arxiv.org/pdf/2508.13142)
      - 视觉规划：[VSP](https://arxiv.org/pdf/2407.01863)
      - 具身规划：[Mini-Behavior](https://arxiv.org/abs/2310.01824), [ViPlan](https://arxiv.org/pdf/2505.13180), [Embodied-Bench](https://arxiv.org/pdf/2502.09560), [Embodied-Agent-Interface](https://embodied-agent-interface.github.io/)
      - 游戏智能体规划：[我的世界](https://minedojo.org/)，[星露谷物语](https://arxiv.org/pdf/2507.07445)
        
有意提前进组的同学完成Stage 1后即可进一步交流研究方向。

## Stage 2：神经符号基础
### 1. 逻辑与符号推理基础
  - 理解命题逻辑与一阶逻辑，掌握基本逻辑推理方法，了解Prolog语言
  - 了解PDDL语言，能够调用规划求解器完成经典规划任务求解，例如积木世界BlocksWorld
  - 参考资料：AIMA对应章节
  
### 2. 神经符号推理基础
  - 学习如何将神经网络与符号推理结合，尝试理解神经符号推理与端到端神经网络的区别
  - 实践任务：阅读论文[Neural-Symbolic Concept learner](https://arxiv.org/abs/1904.12584)，并在Clevr数据集中实现
  - 实践任务：阅读论文[Learning Adaptive Planning Representations with Natural Language Guidance](https://arxiv.org/pdf/2312.08566)，并在Mini Mincraft环境中实现

  

## Stage 3：综述论文
1. 阅读相关领域综述，了解前沿进展，形成对该领域的整体认识
2. 参考综述：
    1. 大模型推理：Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models https://arxiv.org/pdf/2503.09567
    2. 多模态推理：
      - Explain Before You Answer: A Survey on Compositional Visual Reasoning https://arxiv.org/abs/2508.17298
      - Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers https://arxiv.org/pdf/2506.23918
      - Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey https://arxiv.org/pdf/2503.12605
      - Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models https://arxiv.org/pdf/2505.04921
    3. 神经符号融合：
      - Neuro-Symbolic Concept: https://arxiv.org/abs/2505.06191
      - Neuro-Symbolic Artificial Intelligence: Towards Improving the Reasoning Abilities of Large Language Models: https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/8905.pdf
  1. 任务：以综述论文为纲，阅读学习相关论文，完成一个PPT汇报，重在凝练形成整体认识，无需介绍方法细节。
  2. **目标：能够从技术和问题两个维度，形成对该领域的分类与认识**

## Stage 4：讨论确定具体方向，论文阅读、复现与改进
