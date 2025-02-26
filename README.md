This project presents an approach which integrates Natural Language Processing (NLP), and Deep Reinforcement Learning (DRL) techniques to classify and assign customer complaints efficiently.
The system assigns an appropriate employee to resolve the complaint based on availability, expertise, sentiment score and time slots. The available time slots are predefined.
<br>
In order to properly assign complaints to the most appropriate employee, the Deep Q-Network (DQN) model is used.<br>
The model includes:<br>
24 neurones in the input layer (Relu activation)<br>
24 neurones in the hidden layer (Relu activation)<br>
Action space size (linear activation) in the output layer<br><br>
The Adam optimiser and Mean Squared Error (MSE) loss are used to construct the model. <br>
The Q-values are computed using: Q(s,a) = r+ max Q(s',a')
where: Q(s,a) is the expected reward for taking action a in state s. r is the reward (based on employee availability and sentiment score). γ=0.95 is the discount factor. Q(s′,a′) is the max Q-value of the next state.

The model learns from experience using a memory buffer (deque). The Bellman equation is applied to update the Q-values. Epsilon decay is used to transition from exploration to exploitation. Once an employee is assigned, the system updates the database to reflect the complaint allocation.
The DQN model predicts the best employee based on past experiences.

![Screenshot 2025-02-18 160102](https://github.com/user-attachments/assets/a63f8aa9-2e6a-416f-ba75-eda6157cdfc2)

![Image](https://github.com/user-attachments/assets/dc6d0c4a-e0f3-4dc7-b72d-ae4ca1bf1284)
