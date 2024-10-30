<div align="center">
    <h2>MIND: <ins>M</ins>ulti-modal <ins>I</ins>ntegrated Predictio<ins>N</ins> and <ins>D</ins>ecision-making with Adaptive Interaction Modality Explorations</h2>
    <br>
        <a href="https://uav.hkust.edu.hk/current-members/" target="_blank">Tong Li</a><sup>*†</sup>,
        <a href="https://masterizumi.github.io/" target="_blank">Lu Zhang</a><sup>*</sup>,
        <a href="https://github.com/sikang" target="_blank">Sikang Liu</a>,
        <a href="https://uav.hkust.edu.hk/group/" target="_blank">Shaojie Shen</a>
    <p>
        <h45>
            HKUST Aerial Robotics Group &nbsp;&nbsp;
            <br>
        </h5>
        <sup>*</sup>Equal Contributions
        <sup>†</sup>Corresponding Author
    </p>
    <a href='https://arxiv.org/pdf/2408.13742'><img src='https://img.shields.io/badge/arXiv-MIND-red' alt='arxiv'></a>
    <a href='https://www.youtube.com/watch?v=Bwlb5Dz2OZQ'><img src='https://img.shields.io/badge/Video-MIND-blue' alt='youtube'></a>
</div>

## 📃 Abstract
Navigating dense and dynamic environments poses a significant challenge for autonomous driving systems, owing to the intricate nature of multimodal interaction, wherein the actions of various traffic participants and the autonomous vehicle are complex and implicitly coupled. In this paper, we propose a novel framework, Multi-modal Integrated predictioN and Decision-making (MIND), which addresses the challenges by efficiently generating joint predictions and decisions covering multiple distinctive interaction modalities. Specifically, MIND leverages learning-based scenario predictions to obtain integrated predictions and decisions with social-consistent interaction modality and utilizes a modality-aware dynamic branching mechanism to generate scenario trees that efficiently capture the evolutions of distinctive interaction modalities with low variation of interaction uncertainty along the planning horizon. The scenario trees are seamlessly utilized by the contingency planning under interaction uncertainty to obtain clear and considerate maneuvers accounting for multi-modal evolutions. Comprehensive experimental results in the closed-loop simulation based on the real-world driving dataset showcase superior performance to other strong baselines under various driving contexts.

<div align="center">
  <img src="misc/overview.png" alt="system overview" />
</div>



## 🔎 Quantitative Comparison of AIME
<p align="center">
  <img src="misc/aime_quan.png"/>
</p>

## 🔎 Qualitative Results On Argoverse 2
<p align="center">
  <img src="misc/av2_sim_1.gif" width = "200"/>
  <img src="misc/av2_sim_2.gif" width = "200"/>
  <img src="misc/av2_sim_3.gif" width = "200"/>
  <img src="misc/av2_sim_4.gif" width = "200"/>
</p>

## 🛠️ Getting started
### Create a new conda virtual environment
```
conda create -n mind python=3.10
conda activate mind
```

### Install dependencies
```
pip install -r requirements.txt 
```

## 🕹️ Run a closed-loop simulation
```
python run_sim.py --config configs/demo_{1,2,3,4}.json
```
- The whole simulation takes about 10 minutes to finish.
You are supposed to get the rendered simulation results saved in the outputs folder.
##  ❤️ Acknowledgment
We would like to express sincere thanks to the authors of the following packages and tools:
- [SIMPL](https://github.com/HKUST-Aerial-Robotics/SIMPL)
- [ILQR](https://github.com/anassinator/ilqr)

