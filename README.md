# Adversarial
# 视觉对抗 (重点推荐**)

# 环境动态创建对抗 (重点推荐**)
[PAIRED](Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design
)
[知乎解读](https://zhuanlan.zhihu.com/p/301930130)
# [图对抗](https://zhuanlan.zhihu.com/p/88934914)
# 强化学习中的对抗攻击
## image based attack in DRL

- Huang, S., Papernot, N., Goodfellow, I., Duan, Y., and Abbeel, P. ***Adversarial attacks on neural network policies***. arXiv preprint arXiv:1702.02284, 2017.
[pdf](http://arxiv.org/abs/1702.02284.pdf)
在本质上类似于FSGM 
用标签进行图像分类的概率被采取行动的概率所取代
[知乎]()

- Kos, J. and Song, D. ***Delving into adversarial attacks on deep policies***. arXiv preprint arXiv:1705.06452, 2017.
[pdf](http://arxiv.org/abs/1705.06452.pdf)
在本质上类似于FSGM 
A3C
对抗性攻击更具弹性
[知乎]()

- Lin, Y.-C., Hong, Z.-W., Liao, Y.-H., Shih, M.-L., Liu, M.-Y., and Sun, M. ***Tactics of adversarial attack on deep reinforcement learning agents***. arXiv preprint arXiv:1703.06748, 2017.
[pdf](http://arxiv.org/abs/1703.06748.pdf)
在本质上类似于Carlini Wagner 攻击
[知乎]()

- Behzadan, V. and Munir, A. ***Vulnerability of deep reinforcement learning to policy induction attacks***. In International Conference on Machine Learning and Data Mining in Pattern Recognition, pp. 262–275. Springer, 2017.
[pdf](https://arxiv.org/pdf/1701.04143.pdf)
[【强化学习应用1４】深度强化学习易受策略诱导攻击的威胁](https://zhuanlan.zhihu.com/p/277484494)


- Pattanaik, A., Tang, Z., Liu, S., Bommannan, G., and Chowdhary, G.***Robust deep reinforcement learning with adversarial attacks***. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, pp. 2040–2042. International Foundation for Autonomous Agents and Multiagent Systems, 2018.
[pdf](https://arxiv.org/pdf/1712.03632.pdf)
[【强化学习应用13】强大的深度强化学习 对抗攻击](https://zhuanlan.zhihu.com/p/277414212)


- [Zhang, H., Chen, H., Xiao, C., Li, B., Liu, M., Boning, D., & Hsieh, C.-J. (2020). ***Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations***. 1–41. Retrieved from http://arxiv.org/abs/2003.08938]
[pdf](https://arxiv.org/pdf/2003.08938.pdf)
[【强化学习应用０8】鲁棒的深度强化学习，可防止对状态观测值的摄动干扰(1)](https://zhuanlan.zhihu.com/p/272828537)
[github](https://github.com/chenhongge/StateAdvDRL)
[SA_DQN](https://github.com/chenhongge/SA_DQN)
[SA_PPO](https://github.com/huanzhang12/SA_PPO)


## agent based attack in DRL


对手在受害者身上施加力矢量或改变诸如摩擦之类的动力学参数
- Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. ***Robust adversarial reinforcement learning***. In Proceedings of the International Conference on Machine Learning (ICML), volume 70, pages 2817–2826, 2017.
[pdf](http://proceedings.mlr.press/v70/pinto17a/pinto17a.pdf)
[【强化学习应用15】强大的对抗强化学习--未完待续【第四部分之后剩下】](https://zhuanlan.zhihu.com/p/277699296)
[github](https://github.com/lerrel/gym-adv)

- Ajay Mandlekar, Yuke Zhu, Animesh Garg, Li Fei-Fei, and Silvio Savarese. ***Adversarially robust policy learning: Active construction of physically-plausible perturbations***. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3932–3939, 2017.
[pdf](http://vision.stanford.edu/pdf/mandlekar2017iros.pdf)
[【强化学习应用16】对抗性强健的策略学习：积极构造物理上令人信服的扰动](https://zhuanlan.zhihu.com/p/279144986)

- [Gleave, A., Dennis, M., Wild, C., Kant, N., Levine, S., & Russell, S. (2019). ***Adversarial Policies: Attacking Deep Reinforcement Learning***. 1–16. Retrieved from http://arxiv.org/abs/1905.10615]
[pdf](http://arxiv.org/abs/1905.10615.pdf)
[【强化学习应用1１】对抗策略：深度强化学习攻击(1)](https://zhuanlan.zhihu.com/p/276002787)


# 综述２篇
- ***Characterizing attacks on deep reinforcement learning***
Xiao, C., Pan, X., He, W., Peng, J., Sun, M., Yi, J., … Song, D. (2019). Characterizing Attacks on Deep Reinforcement Learning. Retrieved from http://arxiv.org/abs/1907.09470
[pdf](https://arxiv.org/pdf/1907.09470.pdf)
[知乎]()
- ***Challenges and countermeasures for adversarial attacks on deep reinforcement learning***
Ilahi, I., Usama, M., Qadir, J., Janjua, M. U., Al-Fuqaha, A., Hoang, D. T., & Niyato, D. (2020). Challenges and Countermeasures for Adversarial Attacks on Deep Reinforcement Learning. 1–21. Retrieved from http://arxiv.org/abs/2001.09684
[pdf](https://arxiv.org/pdf/2001.09684.pdf)
[【强化学习应用18】对抗性攻击对强化学习的挑战与对策](https://zhuanlan.zhihu.com/p/280689431)

# 调研漏掉的文章
- ***Whatever Does Not Kill Deep Reinforcement Learning, Makes It Stronger***
Behzadan, V., & Munir, A. (2017). Whatever Does Not Kill Deep Reinforcement Learning, Makes It Stronger. Retrieved from http://arxiv.org/abs/1712.09344
[pdf](https://arxiv.org/pdf/1712.09344.pdf)
[github](https://github.com/behzadanksu/rl-attack)

- ***Sequential attacks on agents for long-term adversarial goals***
[55] E. Tretschk, S. J. Oh, and M. Fritz, “Sequential attacks on agents for long-term adversarial goals,” arXiv preprint arXiv:1805.12487, 2018.
[pdf](https://arxiv.org/pdf/1805.12487.pdf)
[知乎]()


- ***A malicious attack on the machine learning policy of a robotic system***
 [43] G. Clark, M. Doran, and W. Glisson, “A malicious attack on the machine learning policy of a robotic system,” in 2018 17th IEEE
International Conference On Trust, Security And Privacy In Computing
And Communications/12th IEEE International Conference On Big Data
Science And Engineering (TrustCom/BigDataSE). IEEE, 2018, pp.
516–521.
paper on IEEE
[知乎]()

- ***Gradient band-based adversarial training for generalized attack immunity of A3C path finding***
[63] T. Chen, W. Niu, Y. Xiang, X. Bai, J. Liu, Z. Han, and G. Li, “Gradient band-based adversarial training for generalized attack immunity of A3C path finding,” arXiv preprint arXiv:1807.06752, 2018.

[pdf](https://arxiv.org/pdf/1807.06752.pdf)
[知乎]()


- ***Reinforcement learning for autonomous defence in software-defined networking***
 [46] Y. Han, B. I. Rubinstein, T. Abraham, T. Alpcan, O. De Vel, S. Erfani, D. Hubczenko, C. Leckie, and P. Montague, “Reinforcement learning for autonomous defence in software-defined networking,” in
International Conference on Decision and Game Theory for Security.
Springer, 2018, pp. 145–165.

[pdf](https://arxiv.org/pdf/1808.05770.pdf)
[知乎]()


- ***Adversarial exploitation of policy imitation*** 
[59] B. Vahid and W. Hsu, “Adversarial exploitation of policy imitation,”
arXiv preprint arXiv:1906.01121, 2019.

[pdf](https://arxiv.org/pdf/1906.01121.pdf)
[知乎]()


-  ***Trojdrl: Trojan attacks on deep reinforcement learning agents***
[57] P. Kiourti, K. Wardega, S. Jha, and W. Li, “Trojdrl: Trojan attacks on deep reinforcement learning agents,” arXiv preprint arXiv:1903.06638,
2019.
[pdf](https://arxiv.org/pdf/1903.06638.pdf)
[知乎]()


- ***Spatiotemporally constrained action space attacks on deep reinforcement learning agents***
[50] X. Yeow Lee, S. Ghadai, K. L. Tan, C. Hegde, and S. Sarkar, “Spatiotemporally constrained action space attacks on deep reinforcement learning agents,” arXiv preprint arXiv:1909.02583, 2019.

[pdf](https://arxiv.org/pdf/1909.02583.pdf)
[知乎]()


- ***Targeted attacks on deep reinforcement learning agents through adversarial observations***
[60] L. Hussenot, M. Geist, and O. Pietquin, “Targeted attacks on deep reinforcement learning agents through adversarial observations,” arXiv
preprint arXiv:1905.12282, 2019.

[pdf](https://arxiv.org/pdf/1905.12282v1.pdf)
[知乎]()


- ***Deceptive reinforcement learning under adversarial manipulations on cost signals*** 
[47] Y. Huang and Q. Zhu, “Deceptive reinforcement learning under adversarial manipulations on cost signals,” arXiv preprint arXiv:1906.10571,
2019.

[pdf](https://arxiv.org/pdf/1906.10571.pdf)
[知乎]()


- ***Sequential triggers for watermarking of deep reinforcement learning policies*** 
[58] V. Behzadan and W. Hsu, “Sequential triggers for watermarking of deep reinforcement learning policies,” arXiv preprint arXiv:1906.01126,
2019.

[pdf](https://arxiv.org/pdf/1906.01126.pdf)
[知乎]()


- ***Adversarial examples construction towards white-box Q table variation in DQN pathfinding training*** 
[64] X. Bai, W. Niu, J. Liu, X. Gao, Y. Xiang, and J. Liu, “Adversarial examples construction towards white-box Q table variation in DQN pathfinding training,” in 2018 IEEE Third International Conference on
Data Science in Cyberspace (DSC). IEEE, 2018, pp. 781–787.
paper on IEEE
[知乎]()



# tools
[adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

### 单像素攻击
- One pixel attack for fooling deep neural networks
[pdf](https://arxiv.org/pdf/1710.08864.pdf)


综述论文
-  Review of Artificial Intelligence Adversarial Attack and Defense Technologies
[pdf](https://pdfs.semanticscholar.org/4af6/c0c61bbaaca04d33deea73b69b8494e0c77e.pdf?_ga=2.131584332.2118617150.1602253897-1057521812.1575539853&_gac=1.213189152.1602383972.EAIaIQobChMIwtmL58Cr7AIV_MEWBR0vKA9UEAAYASAAEgIxT_D_BwE)
- [Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey]
[pdf](https://arxiv.org/pdf/1801.00553.pdf)

- Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples
[论文笔记](https://www.zybuluo.com/wuxin1994/note/850755)

### ICLR 2020 对抗样本的攻守道
### Oral：
 - Adversarial Training and Provable Defenses: Bridging the Gap
    [pdf](https://openreview.net/attachment?id=SJxSDxrKDr&name=original_pdf)
### Spotlight：

 - Defending Against Physically Realizable Attacks on Image Classification 
    [pdf](https://openreview.net/attachment?id=H1xscnEKDr&name=original_pdf)

 - Training individually fair ML models with sensitive subspace robustness 
    [pdf](https://openreview.net/attachment?id=B1gdkxHFDH&name=original_pdf)
    
    We consider training machine learning models that are fair in the sense that their performance is invariant under certain sensitive perturbations to the inputs. For example, the performance of a resume screening system should be invariant under changes to the gender and/or ethnicity of the applicant. We formalize this notion of algorithmic fairness as a variant of individual fairness and develop a distributionally robust optimization approach to enforce it during training. We also demonstrate the effectiveness of the approach on two ML tasks that are susceptible to gender and racial biases.
    我们认为训练机器学习模型是公平的，因为在对输入的某些敏感扰动下，它们的性能是不变的。 例如，简历筛选系统的性能应根据申请人的性别和/或种族的变化而保持不变。 我们将算法公平性的概念形式化为个体公平性的变体，并开发了一种分布式健壮的优化方法来在训练期间对其进行实施。 我们还证明了该方法对容易受到性别和种族偏见影响的两个机器学习任务的有效性。

 - White Noise Analysis of Neural Networks
    [pdf](https://openreview.net/attachment?id=H1ebhnEYDH&name=original_pdf)

 - Enhancing Adversarial Defense by k-Winners-Take-All
    [pdf](https://openreview.net/attachment?id=Skgvy64tvr&name=original_pdf)

 - Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets
    [pdf](https://openreview.net/attachment?id=BJlRs34Fvr&name=original_pdf)

    
###   Poster：

 - Enhancing Transformation-Based Defenses Against Adversarial Attacks with a Distribution Classifier 
    [pdf](https://openreview.net/forum?id=BkgWahEFvr)

 - Jacobian Adversarially Regularized Networks for Robustness
    [pdf](https://openreview.net/forum?id=Hke0V1rKPS)

 - Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks 
    [pdf](https://openreview.net/forum?id=ByxtC2VtPB)

 - GAT: Generative Adversarial Training for Adversarial Example Detection and Robust Classification
    [pdf](https://openreview.net/forum?id=SJeQEp4YDH)

 - Detecting and Diagnosing Adversarial Images with Class-Conditional Capsule Reconstructions 
    [pdf](https://openreview.net/forum?id=Skgy464Kvr)

 - Adversarial Policies: Attacking Deep Reinforcement Learning
    [pdf](https://openreview.net/forum?id=HJgEMpVFwB)

 - Improving Adversarial Robustness Requires Revisiting Misclassified Examples
    [pdf](https://openreview.net/forum?id=rklOg6EFwS)

 - Fooling Detection Alone is Not Enough: Adversarial Attack against Multiple Object Tracking 
    [pdf](https://openreview.net/forum?id=rJl31TNYPr)

 - Robust Local Features for Improving the Generalization of Adversarial Training
    [pdf](https://openreview.net/forum?id=H1lZJpVFvr)


 - Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness
    [pdf](https://openreview.net/forum?id=Byg9A24tvB)

 - Black-Box Adversarial Attack with Transferable Model-based Embedding
    [pdf](https://openreview.net/forum?id=SJxhNTNYwB)

 - Certified Robustness for Top-k Predictions against Adversarial Perturbations via Randomized Smoothing
    [pdf](https://openreview.net/forum?id=BkeWw6VFwr)

 - Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness
    [pdf](https://openreview.net/forum?id=SJgwzCEKwH)

 - Sign-OPT: A Query-Efficient Hard-label Adversarial Attack 
    [pdf](https://openreview.net/forum?id=SklTQCNtvS)

 - Fast is better than free: Revisiting adversarial training
    [pdf](https://openreview.net/forum?id=BJx040EFvH)

 - Intriguing Properties of Adversarial Training at Scale
    [pdf](https://openreview.net/forum?id=HyxJhCEFDS)

 - Biologically inspired sleep algorithm for increased generalization and adversarial robustness in deep neural networks 
    [pdf](https://openreview.net/forum?id=r1xGnA4Kvr)

 - Certified Defenses for Adversarial Patches
    [pdf](https://openreview.net/forum?id=HyeaSkrYPH)

 - Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks
    [pdf](https://openreview.net/forum?id=SJlHwkBYDH)

 - Provable robustness against all adversarial lp-perturbations for  p>=1
    [pdf](https://openreview.net/forum?id=rklk_ySYPB)

 - EMPIR: Ensembles of Mixed Precision Deep Networks for Increased Robustness Against Adversarial Attacks
    [pdf](https://openreview.net/pdf?id=HJem3yHKwH)

 - BayesOpt Adversarial Attack 
    [pdf](https://openreview.net/pdf?id=Hkem-lrtvH)

 - Unrestricted Adversarial Examples via Semantic Manipulation 
    [pdf](https://openreview.net/pdf?id=Sye_OgHFwH)

 - BREAKING CERTIFIED DEFENSES: SEMANTIC ADVERSARIAL EXAMPLES WITH SPOOFED ROBUSTNESS CERTIFICATES 
    [pdf](https://openreview.net/pdf?id=HJxdTxHYvB)

 - Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks 
    [pdf](https://openreview.net/pdf?id=SyevYxHtDB)

 - MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius
    [pdf](https://openreview.net/pdf?id=rJx1Na4Fwr)


 - A Target-Agnostic Attack on Deep Models: Exploiting Security Vulnerabilities of Transfer Learning 
    [pdf](https://openreview.net/pdf?id=BylVcTNtDS)

 - Sign Bits Are All You Need for Black-Box Attacks
    [pdf](https://openreview.net/pdf?id=SygW0TEFwH)
    
 ### Sound Adversarial Attack
- Adversarial Attacks in Sound Event Classification
 [pdf](https://arxiv.org/pdf/1907.02477.pdf)
[code](https://github.com/VinodS7/audio-adversaries)
[复现坑扫除](https://www.zhezhi.press/2019/11/27/audio-adversarial-examples-targeted-attacks-on-speech-to-text-%E5%A4%8D%E7%8E%B0/)
- C. Kereliuk, B. L. Sturm, and J. Larsen. Deep Learning and Music Adversaries. IEEE Transactions on Multimedia, 17(11):2059–2071, November 2015.
[pdf](https://arxiv.org/pdf/1507.04761.pdf)
[code](https://github.com/coreyker/dnn-mgr)
- Tianyu Du, Shouling Ji, Jinfeng Li, Qinchen Gu, Ting Wang, and Raheem Beyah. Sirenattack: Generating adversarial audio for end-to-end acoustic systems. CoRR, abs/1901.07846, 2019.
[pdf](https://dl.acm.org/doi/pdf/10.1145/3320269.3384733)
### 几种机制来产生对抗性音频
几种机制来产生对抗性音频(如下３种方法）。 它们都是基于梯度信息的，因此彼此之间的共享差异很小。
- Nicholas Carlini and David Wagner. 2018. **Audio Adversarial Examples: Targeted Attacks on Speech-to-Text.** (2018). arXiv:1801.01944
Carlini等人提出了一种可以产生对抗性音频的方法，该对抗性音频可以由DeepSpeech在白盒设置下转录为所需文本。
[pdf](https://arxiv.org/pdf/1801.01944.pdf)
[code](https://github.com/carlini/audio_adversarial_examples)
- Moustapha M Cisse, Yossi Adi, Natalia Neverova, and Joseph Keshet. 2017. **Hou- dini: Fooling Deep Structured Visual and Speech Recognition Models with Ad- versarial Examples.** In Advances in Neural Information Processing Systems 30, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.). Curran Associates, Inc., 6977–6987.
Cisse等人提出了Houdini攻击，该攻击可以转移到不同的未知ASR模型中。但是，它只能构建针对语音相似短语的对抗性音频。
[pdf](https://papers.nips.cc/paper/7273-houdini-fooling-deep-structured-visual-and-speech-recognition-models-with-adversarial-examples.pdf)
[code]()
- Dan Iter, Jade Huang, and Mike Jermann. 2017. **Generating adversarial examples for speech recognition.** (2017).
Iter等人通过向梅尔频率倒谱系数（MFCC）特征添加扰动来生成对抗性音频，然后从扰动的MFCC特征中重建语音。但是，反MFCC过程引入的噪声使它们的对抗性音频听起来对人类来说很奇怪。
[pdf](http://diyhpl.us/~bryan/papers2/ai/speech-recognition/Generating%20adversarial%20examples%20for%20speech%20recognition.pdf)
[code]()
- Yuan Gong and Christian Poellabauer. 2017. **Crafting Adversarial Examples For Speech Paralinguistics Applications.** (2017). arXiv:1711.03280
Gong等人证明语音的2％失真会使基于深度神经网络（DNN）的模型无法识别说话者的身份。
[Audio Adversarial Examples Paper List](https://github.com/imcaspar/audio-adv-papers)
[awesome-deep-learning-music](https://github.com/ybayle/awesome-deep-learning-music)

[CV||对抗攻击领域综述（adversarial attack）](https://zhuanlan.zhihu.com/p/104532285)
[ICLR 2020 对抗样本的攻守道](https://zhuanlan.zhihu.com/p/98419512)
[吐血整理 | AI新方向：对抗攻击](https://zhuanlan.zhihu.com/p/88886843)
[综述论文：对抗攻击的12种攻击方法和15种防御方法----机器之心](https://www.jiqizhixin.com/articles/2018-03-05-4)

## Gradient-Based基于梯度的方法
- Intriguing properties of neural networks
***BFGS***
[pdf](https://arxiv.org/pdf/1312.6199.pdf)
```
2014年，文章《Intriguing properties of neural networks》发布，这是对抗攻击领域的开山之作，文中介绍了一些NN的性质，首次提出了对抗样本的概念。在文章中，作者指出了深度神经网络学习的输入-输出映射在很大程度是相当不连续的，我们可以通过应用某些难以感知的扰动来使网络对图像分类错误，该扰动是通过最大化网络的预测误差来发现的。此外，这些扰动的特定性质并不是学习的随机产物：相同的扰动会导致在数据集的不同子集上进行训练的不同网络对相同输入进行错误分类 [公式]
同时文章中也提出了BFGS：通过寻找最小的损失函数添加项，使得神经网络做出误分类，将问题转化成了凸优化。问题的数学表述如下：
```
- Explaining and Harnessing Adversarial Examples
***FGSM：Fast Gradient Sign Method***
ICLR 2015
[pdf](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
[论文笔记](https://zhuanlan.zhihu.com/p/166364358)
```
Goodfellow等人在2014年提出了提出了快速梯度符号方法（FGSM）以生成具有单个梯度步长的对抗性示例。在进行反向传播之前，该方法用于扰动模型的输入，这是对抗训练的早期形式。
```
- DeepFool: a simple and accurate method to fool deep neural networks
***DeepFool***
[pdf](https://arxiv.org/pdf/1511.04599.pdf)
```
SM Moosavi-Dezfooli等人在2015年提出的DeepFool通过计算出最小的必要扰动，并应用到对抗样本构建的方法，使用的限制扰动规模的方法是L2范数，得到比较优的结果
```
- The Limitations of Deep Learning in Adversarial Settings
***JSMA***
[pdf](https://arxiv.org/pdf/1511.07528.pdf)
[论文笔记](https://baidinghub.github.io/2020/04/03/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/#%E4%BA%8C%E3%80%81%E8%AE%BA%E6%96%87%E8%83%8C%E6%99%AF%E5%8F%8A%E7%AE%80%E4%BB%8B)
```
N Papernot等人在2015年的时候提出的JSMA，通过计算神经网络前向传播过程中的导数生成对抗样本
```
- Towards evaluating the robustness of neural networks
***R + FGSM***
[pdf](https://arxiv.org/pdf/1608.04644.pdf)
```
Florian Tramèr等人在2017年通过添加称为R + FGSM的随机化步骤来增强此攻击，后来，基本迭代方法对FGSM进行了改进，采取了多个较小的FGSM步骤，最终使基于FGSM的对抗训练均无效。
```

- Ensemble Adversarial Training: Attacks and Defenses
ICLR 2018
[pdf](https://arxiv.org/pdf/1705.07204.pdf)
```
N Carlini,D Wagner等人在2017年提出了一个更加高效的优化问题，能够以添加更小扰动的代价得到更加高效的对抗样本。
```

- 
[pdf]()
```
Nicolas Papernot等人在2017年的时候利用训练数据，训练出可以从中生成对抗性扰动的完全可观察的替代物模型。
```

- SafetyNet: Detecting and Rejecting Adversarial Examples Robustly
[pdf](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lu_SafetyNet_Detecting_and_ICCV_2017_paper.pdf)
ICCV 2017
```
Liu等人2016年的时候在论文《SafetyNet: Detecting and Rejecting Adversarial Examples Robustly》证明了：如果在一组替代模型上创建对抗性示例，则在某些情况下，被攻击模型的成功率可以达到100％。
```
- ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models
[pdf](https://dl.acm.org/doi/pdf/10.1145/3128572.3140448)
```
 从概念上讲，这些攻击使用数值估算梯度的预测。该方法的开始真正work在于2017年PY Chen等人提出的ZOO方法《ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models》文中通过对一阶导和二阶导的近似、层次攻击等多种方式减少了训练时间，保障了训练效果
```
- BayesOpt Adversarial Attack 
[pdf](https://openreview.net/attachment?id=Hkem-lrtvH&name=original_pdf)
ICLR 2020
```
B. Ru, A. Cobb等人在2020年发表文章《BayesOpt Adversarial Attack》利用了贝叶斯优化来以高查询效率找到成功的对抗扰动。此外该论文还通过采用可替代的代理结构来减轻通常针对高维任务的优化挑战，充分利用了我们的统计替代模型和可用的查询数据，以通过贝叶斯模型选择来了解搜索空间的最佳降维程度。
```
- Yet another but more efficient black-box adversarial attack: tiling and evolution strategies
[pdf](https://arxiv.org/pdf/1910.02244.pdf)
```
L. Meunier等人在2020年发表文章《Yet another but more efficient black-box adversarial attack: tiling and evolution strategies》利用了evolutional algorithms。通过结合两种优化方法实现了无导数优化。
```
- Query-efficient Meta Attack to Deep Neural Networks
[pdf](https://arxiv.org/pdf/1906.02398.pdf)
ICLR 2020
```
J. Du等人在2020年发表了文章《Query-efficient Meta Attack to Deep Neural Networks》采用了meta learning来近似估计梯度。该方法可以在不影响攻击成功率和失真的情况下，大大减少所需的查询次数。 通过训练mata attacker，并将其纳入优化过程以减少查询数量。
```
- Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models
[pdf](https://arxiv.org/pdf/1712.04248.pdf)
```
Wieland Brendel等人在2018年提出的基于决策边界的的方法，这种方法是第三种方法的一种变化，有点像运筹学中的原始对偶方法，保证结果的情况之下，逐渐使得条件可行。通过多次迭代的方式使得条件逐渐收敛。
```
- Trust Region Based Adversarial Attack on Neural Networks
CVPR 2019
[pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yao_Trust_Region_Based_Adversarial_Attack_on_Neural_Networks_CVPR_2019_paper.pdf)
[论文笔记](https://blog.csdn.net/Invokar/article/details/96869889)
```
Z Yao 等人在2019年的CVPR上发表的《Trust region basedadversarial attackon neural networks》，这种方法在非凸优化问题上有着非常好的效果
```
- Universal Adversarial Attack on Attention and the Resulting Dataset DAmageNet
***AoA***   Attack on Attntion
[pdf](https://arxiv.org/pdf/2001.06325.pdf)
[论文笔记]()
```
S Chen 等人在2020年发表的《Universal Adversarial Attack on Attention and the Resulting Dataset DAmageNet》是该方面的开山之作，同时设计了一个AoA数据集DAmageNet。
```
- Tactics of Adversarial Attack on Deep Reinforcement Learning Agents
[pdf](https://arxiv.org/pdf/1703.06748.pdf)
[论文笔记]()
```
Lin等人在《Tactics of adversarial attack on deep reinforcement learning agents》提出了两种不同的针对深度强化学习训练的代理的对抗性攻击。在第一种攻击中，被称为策略定时攻击，对手通过在一段中的一小部分时间步骤中攻击它来最小化对代理的奖励值。提出了一种方法来确定什么时候应该制作和应用对抗样本，从而使攻击不被发现。在第二种攻击中，被称为迷人攻击，对手通过集成生成模型和规划算法将代理引诱到指定的目标状态。生成模型用于预测代理的未来状态，而规划算法生成用于引诱它的操作。这些攻击成功地测试了由最先进的深度强化学习算法训练的代理。
```
- Crafting adversarial input sequences for recurrent neural networks
[pdf](https://ieeexplore.ieee.org/abstract/document/7795300)
[论文笔记]()
```
循环神经网络：2016年Papernot等人在《Crafting adversarial input sequences for recurrent neural networks》提出
```
- Audio adversarial examples: Targeted attacks on speech-to-text
[pdf](https://people.eecs.berkeley.edu/~daw/papers/audio-dls18.pdf)
[论文笔记]()
```
语义切割和物体检测：2018年Carlini N等人在《Audio adversarial examples: Targeted attacks on speech-to-text》提出
```
- Efficient defenses against adversarial attacks
[pdf](https://arxiv.org/pdf/1707.06728.pdf)
[论文笔记]()
```
V Zantedeschi 等人在2017年发表文章《Efficient defenses against adversarial attacks》提出了一种双重defence方法，能够在对原来模型标准训练代价影响较小的情况下完成配置。防御从两个方面入手，其一是通过改变ReLU激活函数为受限的ReLU函数以此来增强网络的稳定性。另一方面是通过高斯数据增强，增强模型的泛化能力，让模型能将原始数据和经过扰动后的数据分类相同。
```
- Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples
[pdf](https://arxiv.org/pdf/1802.00420.pdf)
[论文笔记]()
```
A Athalye等人在2018年的时候发表文章《Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples》在文章中提到 发现了一种「混淆梯度」（obfuscated gradient）现象，它给对抗样本的防御带来虚假的安全感。在案例研究中，试验了 ICLR 2018 接收的 8 篇论文，发现混淆梯度是一种常见现象，其中有 7 篇论文依赖于混淆梯度，并被的这一新型攻击技术成功攻克。
```
- Universal adversarial training
[pdf](https://arxiv.org/pdf/1811.11304.pdf)
[论文笔记]()
```
A Shafahi等人在2019年发表文章《Universal adversarial training》。作者通过使用交替或同时随机梯度方法优化最小-最大问题来训练鲁棒模型来防御universal adversarial attack。 同时证明了：使用某些使用“归一化”梯度的通用噪声更新规则，这是可能的。
```
- Adversarial Robustness Against the Union of Multiple Perturbation Models
ICML '20
[pdf](https://arxiv.org/pdf/1909.04068.pdf)
[论文笔记]()
```
P Maini等人在2019年的时候发表文章《Adversarial Robustness Against the Union of Multiple Perturbation Models》。在文中，作者证明了在针对多个Perturbation模型的联合进行训练时，对抗训练可以非常有效。 同时作者还比较了对抗训练的两个简单概括和改进的对抗训练程序，即多重最速下降，该方法将不同的扰动模型直接合并到最速下降的方向。
```
- MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius
[pdf](https://arxiv.org/pdf/2001.02378.pdf)
[论文笔记]()
```
R Zhai等人在2020年发表文章《MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius》这是一种无攻击且可扩展的鲁棒的训练方法，它可以直接最大化平滑分类器的认证半径。同时文章证明了对抗性训练不是鲁棒的训练的必要条件，基于认证的防御是未来研究的有希望的方向！
```
- 
[pdf]()
[论文笔记]()




    



