# Updates for March 20, 2024

## Progress

- Installed `libfranka` and demonstrated it on the real robot.
- RealSense: Got the RealSense Viewer working with the RealSense D415 camera.
- Installed `panda_simulation`, a simulation environment for the robot using Gazebo and MoveIt!
- **Drake**: Been working on installing the LCM driver for using Drake with the physical robot. I tried `drake-franka-driver` from `DexaiRobotics` as well as `drake-franka-driver` from Russ Tedrake's group. Ran into a lot of dependancy issues that I wasn't able to solve yet, but might try some more.
- Tried using `panda-gym` to train an RL policy for block pushing in simulation. Seems to require very careful hyperparameter tuning, or not the best control strategy for this task ([link](https://github.com/dbombara/franka-pipeline/tree/main/reinforcement-learning)).
- Modified some of the examples from the MIT Robotic Manipulation textbook to show the Franka Panda instead of the Kuku iiwa ([link](https://github.com/dbombara/franka-pipeline/tree/main/drake_examples)).
- Started on the [knowledge bank / research pitch](https://github.com/dbombara/franka-pipeline/blob/main/knowledge-bank-research-pitch.md) for the Diffusion Policy project.

## Next Steps

1. Work on the [research pitch and knowledge bank](https://github.com/dbombara/franka-pipeline/blob/main/knowledge-bank-research-pitch.md) for the Diffusion Policy project (present to Ashok next week).
2. Try to get the LCM Drake driver working with the physical robot.
3. Work more on the RealSense integration with the `diffusion-policy` code base. Currently, issue with our computer's current kernal version.
4. Demonstrate a control policy trained using Deep RL or the Diffusion Policy on the physical robot.