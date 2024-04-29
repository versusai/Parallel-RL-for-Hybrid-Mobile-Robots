<h1 align="center">Parallel Deep Reinforcement Learning <br> for Hybrid Mobile Robots</h1>

This repository contains the code and simulation environments for the paper "Parallel Deep Reinforcement Learning for Hybrid Mobile Robots," which introduces a parallel deep reinforcement learning methodology for mapless navigation of Hybrid Unmanned Aerial Underwater Vehicles (HUAUVs).

## Abstract

We present a novel parallel deep reinforcement learning framework that significantly improves the autonomous navigation, obstacle avoidance, and medium transition capabilities of HUAUVs in simulated air and water environments. By leveraging the parallelization of learning agents, we demonstrate enhanced learning effectiveness and a reduction in required training time.

## Contents

- `LOGS`: Directory for log files generated during training and evaluation.
- `Parallel-Hydrone-DRL`: Source code for the parallel deep reinforcement learning algorithm.
- `evaluation`: Evaluation scripts and utilities to assess HUAUV navigation performance.
- `gym_hydrone`: Custom Gym environment for HUAUV simulation.
- `models`: Trained model files and configuration for the reinforcement learning agents.

## Installation

Please ensure that you have installed all dependencies as listed in `Parallel-Hydrone-DRL/requirements.txt`.

## Usage

To train the models using our parallel deep reinforcement learning framework:
```bash
python3 Parallel-Hydrone-DRL/train.py
```
For evaluation of the trained models:
```bash
python3 evaluation/evaluate.py
```
## Simulation

All simulations were performed in the Gazebo simulator, enhanced with RotorS and UUVSim plugins for realistic aerial and underwater dynamics.

| ![Grando jint a-w-pd-rl](media/grando_jint_a-w-pd-rl.gif) | ![Grando jint w-a-pd-rl](media/grando_jint_w-a-pd-rl.gif) |
|:----------------------------------------------------------:|:----------------------------------------------------------:|
|    A-W PD-RL                                    |    W-A PD-RL                                    |
| ![Grando jint a-w-ps-rl](media/grando_jint_a-w-ps-rl.gif)  | ![Grando jint w-a-ps-rl](media/grando_jint_w-a-ps-rl.gif)  |
|    A-W PS-RL                                    |    W-A PS-RL                                    |

## Paper and Citation

If you use our methodology or this codebase in your work, please cite:

TODO: Add BibTeX entry for the paper.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Our thanks to the contributors and institutions that supported the research and development of this project:

- Ricardo B. Grando
- Raul Steinmetz
- Junior D. Jesus
- Victor A. Kich
- Alisson H. Kolling
- Rodrigo S. Guerra
- Paulo L. J. Drews-Jr
- Robotics and AI Lab, Technological University of Uruguay
- Centro de Ciencias Computacionais, Universidade Federal do Rio Grande - FURG
- Centro de Tecnologia, Universidade Federal de Santa Maria - UFSM
- Intelligent Robot Laboratory, University of Tsukuba

## Contact

For any inquiries, please contact us at `ricardo.bedin@utec.edu.uy`.

## Additional Resources

- [Video Demonstrations](https://youtu.be/mI5DAcXI988)
