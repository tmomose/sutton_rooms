# sutton_rooms

## Environment
Room world from Sutton et al (Between MDP and semi-MDP)

![room-map](https://github.com/tmomose/sutton_rooms/blob/master/images/default_room.png)

Paper: [Between MDPs and semi-MDPs](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)

## Experiment

Used the room environment to test out some ideas for hierarchical reinforcement learning and planning in HRL.

1. Plain flat Q-learning
   1. Just one version. [[code](https://github.com/tmomose/sutton_rooms/blob/master/q_learning_test.py)]
1. Hierarchical Q-learning
   1. Basic version (s-MDP; two-layer hierarchy with predefined deterministic lower-level policy) [[code](https://github.com/tmomose/sutton_rooms/blob/master/smdp_q_learning_test.py)]
   1. Intraoption-learning version (lower-level is trainable) [[code](https://github.com/tmomose/sutton_rooms/blob/master/smdp_q_learning_test_intraoption.py)]
1. Planning Hierarchical Q-learning
   1. Basic version (same as Hierarchical Q-learning but with a 2-step plan output from the upper level; No replanning) [[code](https://github.com/tmomose/sutton_rooms/blob/master/smdp_plan_q_learning_test.py)]
   1. Version with replanning [[code](https://github.com/tmomose/sutton_rooms/blob/master/smdp_replan_q_learning_test.py)]
