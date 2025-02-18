This project presents an approach which integrates Natural Language Processing (NLP), and Deep Reinforcement Learning (DRL) techniques to classify and assign customer complaints efficiently.
The system assigns an appropriate employee to resolve the complaint based on availability, expertise, sentiment score and time slots. The available time slots are predefined.
<br>
**Reinforcement Learning-Based Employee Selection:** The Deep Q-Network (DQN) model is implemented to intelligently assign complaints to the most suitable employee. 
<br>
The model consists of:
<br>
**Input Layer:** 24 neurons (Relu activation)
<br>
**Hidden Layer:** 24 neurons (Relu activation)
<br>
**Output Layer:** Action space size (Linear activation)
<br>
The model is compiled using Mean Squared Error (MSE) loss and Adam optimizer. 
The Q-values are computed using: Q(s,a) = r+ max Q(s',a')
where: Q(s,a) is the expected reward for taking action a in state s. r is the reward (based on employee availability and sentiment score). γ=0.95 is the discount factor. Q(s′,a′) is the max Q-value of the next state.
The model learns from experience using a memory buffer (deque). The Bellman equation is applied to update the Q-values. Epsilon decay is used to transition from exploration to exploitation. Once an employee is assigned, the system updates the database to reflect the complaint allocation.
The DQN model predicts the best employee based on past experiences.

![Screenshot 2025-02-18 160102](https://github.com/user-attachments/assets/a63f8aa9-2e6a-416f-ba75-eda6157cdfc2)
