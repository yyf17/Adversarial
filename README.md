# Adversarial
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
