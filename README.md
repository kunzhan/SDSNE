# Stationary Diffusion State Neural Estimation
Although many graph-based clustering methods attempt to model the stationary diffusion state in their objectives, using a predefined graph limits their performance. We argue that the estimation of the stationary diffusion state can be achieved by gradient descent over neural networks. We specifically design the Stationary Diffusion State Neural Estimation (SDSNE) to exploit multiview structural graph information for co-supervised learning. We explore how to design a graph neural network specially for unsupervised multiview learning and integrate multiple graphs into a unified consensus graph by a shared self-attentional module. The view-shared self-attentional module utilizes the graph structure to learn a view-consistent global graph. Meanwhile, instead of using auto-encoder in most unsupervised learning graph neural networks, SDSNE uses a co-supervised strategy with structure information to supervise the model learning. The co-supervised strategy as the loss function guides SDSNE in achieving the stationary state. Experiments on several multiview datasets demonstrate the effectiveness of SDSNE in terms of six clustering evaluation metrics.

# Experiment
```sh
# SDSNE_km needs to use scikit-learn==0.23.1
conda install scikit-learn==0.23.1
```
```sh
bash demo_SDSNE.sh
```

# Citation
We appreciate it if you cite the following paper:
```
@InProceedings{LiuAAAI2022,
  author =    {Chenghua Liu and Zhuolin Liao and Yixuan Ma and Kun Zhan},
  title =     {Stationary diffusion state neural estimation for multiview clustering},
  booktitle = {AAAI},
  year =      {2022},
  pages =     {1413--1421}
}

```

# Contact
https://kunzhan.github.io/

If you have any questions, feel free to contact me. (Email: `ice.echo#gmail.com`)