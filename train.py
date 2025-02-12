from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd
import re
import random
from collections import deque
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time

# Initialize sentiment analysis components
model_name = "assemblyai/distilbert-base-uncased-sst2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load pre-trained models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
nlp = spacy.load("en_core_web_sm")

# Load employee database
employee_database_path = "/content/employee_database (1).csv"
employee_df = pd.read_csv(employee_database_path)

# Load complaints from CSV
complaints_file_path = "/content/complaints_dataset.csv"  # Update this path to your CSV file
complaints_df = pd.read_csv(complaints_file_path)

# Define time slots
TIME_SLOTS = ["10AM-11AM", "11AM-12PM", "12PM-1PM", "2PM-6PM"]

# Load LDA model
vectorizer = CountVectorizer(stop_words="english")
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)

# Issue Category Keywords
ISSUE_CATEGORIES = {
    "Billing": ["invoice", "billing", "bill", "charges", "cost", "payment", "duplicate", "processing", "extra", "charge"],
    "Technical": ["engine", "brakes", "noise", "battery", "broken", "engine stalling", "transmission", "overheating"],
    "Maintenance": ["service", "schedule", "repair", "replace", "routine", "parts", "availability"],
    "Vehicle Performance": ["engine", "stalling", "noise", "shifting", "brakes", "vibration", "transmission"],
    "Sales and Purchase": ["delivery", "financing", "loan", "payment", "damage", "warranty"],
    "Customer Service": ["communication", "response", "representative", "dealer", "advertising", "complaints"],
    "Safety Recalls": ["recall", "repair", "safety", "notification"],
    "Technology and Connectivity": ["infotainment", "bluetooth", "ADAS", "autonomous", "privacy", "cybersecurity"],
    "Legal and Regulatory": ["lemon laws", "safety regulations", "compliance"],
    "Environmental Concerns": ["fuel", "electric", "emissions", "range", "emission"]
}

def get_sentiment_score(text):
    """Get sentiment score for input text using the pre-trained model"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    negative_score, positive_score = probs.squeeze().tolist()
    sentiment_score = (positive_score - negative_score) * 10
    return sentiment_score

# Function to extract named entities
def extract_entities(doc):
    parsed_doc = nlp(doc)
    entities = {ent.label_: ent.text for ent in parsed_doc.ents}
    return entities

# Function to interpret date context
def interpret_date_context(doc, entities):
    date_patterns = {
        "Last Serviced Date": r"\b(last service|serviced on| last maintenance)\b",
        "Service Requesting Date": r"\b(request service|book for|schedule|appointment|visit on|schedule service on)\b",
        "Issue Start Date": r"\b(since|started on|from|began|yesterday)\b",
        "Invoice/Billing Date": r"\b(invoice|billing|charged on|last invoice|payment due)\b"
    }

    date_results = {}
    if "DATE" in entities:
        for date in entities["DATE"].split(","):
            matched = False
            for context, pattern in date_patterns.items():
                if re.search(pattern, doc, re.IGNORECASE):
                    date_results[date] = context
                    matched = True
                    break
            if not matched:
                classification = classifier(doc, list(date_patterns.keys()))
                predicted_label = classification["labels"][0]
                date_results[date] = predicted_label

    today = datetime.today()
    tomorrow = today + timedelta(days=1)

    if "today" in doc.lower():
        date_results[today.strftime("%Y-%m-%d")] = "Service Requesting Date"
    if "tomorrow" in doc.lower():
        date_results[tomorrow.strftime("%Y-%m-%d")] = "Service Requesting Date"

    # Check for dates in various formats
    date_formats = [
        "%Y-%m-%d",  
        "%d-%m-%Y",  
        "%d/%m/%Y",  
        "%Y/%m/%d", 
        "%d %B %Y",  
        "%d %b %Y",  
        "%B %d, %Y",  
        "%b %d, %Y", 
    ]

    for fmt in date_formats:
        found_dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', doc)
        for found_date in found_dates:
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(found_date, fmt)
                    date_results[parsed_date.strftime("%Y-%m-%d")] = "Service Requesting Date"
                    break  
                except ValueError:
                    continue

    return date_results

# Function to classify issue category
def classify_issue(doc):
    matched_label = None
    for label, keywords in ISSUE_CATEGORIES.items():
        if any(word in doc.lower() for word in keywords):
            matched_label = label
            break
    if not matched_label:
        classification = classifier(doc, list(ISSUE_CATEGORIES.keys()))
        matched_label = classification["labels"][0]
    return matched_label

def assign_time_slot(employee, requested_date=None):
    servicing_dates = employee.get("Servicing_Dates", "")
    existing_slots = [slot.strip() for slot in servicing_dates.split(";") if slot]

    today = datetime.today()

    # Handle requested date
    if requested_date:
        if requested_date.lower() == "today":
            requested_date_obj = today
        elif requested_date.lower() == "tomorrow":
            requested_date_obj = today + timedelta(days=1)
        else:
            date_formats = [
                "%Y-%m-%d",  # y-m-d
                "%d-%m-%Y",  # d-m-y
                "%d/%m/%Y",  # d/m/y
                "%Y/%m/%d",  # y/m/d
                "%d %B %Y",  # d Month y
                "%d %b %Y",  # d Mon y
                "%B %d, %Y",  # Month d, y
                "%b %d, %Y",  # Mon d, y
            ]
            requested_date_obj = None
            for fmt in date_formats:
                try:
                    requested_date_obj = datetime.strptime(requested_date, fmt)
                    break  # Exit loop if parsing is successful
                except ValueError:
                    continue

            if requested_date_obj is None:
                print(f"Warning: Requested date '{requested_date}' could not be parsed.")
                return None  

        # Check if the requested date has available slots
        for slot in TIME_SLOTS:
            full_slot = f"{requested_date_obj.strftime('%Y-%m-%d')} {slot}"
            if full_slot not in existing_slots:
                return full_slot 

    for slot in TIME_SLOTS:
        full_slot = f"{today.strftime('%Y-%m-%d')} {slot}"
        if full_slot not in existing_slots:
            return full_slot  
    return None  

# DQN Agent for Complaint Assignment
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
    # Reshape the state to have a batch dimension
        state = np.reshape(state, [1, self.state_size])  
        if np.random.rand() <= self.epsilon:
          return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

# Function to prepare the state vector

def prepare_state(complaint, employees):
    unique_categories = employee_df["Category"].unique().tolist()

    # Initialize category encoding
    category_encoding = np.zeros(len(unique_categories))

    if complaint["category"] in unique_categories:
        category_encoding[unique_categories.index(complaint["category"])] = 1
    else:
        print(f"Warning: Category '{complaint['category']}' not found in employee categories.")

    # Normalize sentiment score to be between 0 and 1
    sentiment_score = (complaint["sentiment_score"] + 10) / 20  

    # Create the state vector
    state_vector = np.concatenate([category_encoding, [sentiment_score]])

    # print(f"State vector shape: {state_vector.shape}, State vector: {state_vector}")

    return state_vector

# Function to assign complaint using DQN
def assign_complaint_dqn(agent, complaint, requested_date=None):
    eligible_employees = employee_df[
        (employee_df["Category"] == complaint["category"])
    ]

    if eligible_employees.empty:
        print("No eligible employees available for this complaint.")
        return False, None 

    agent.action_size = len(eligible_employees)
    state = prepare_state(complaint, eligible_employees)

    state = np.reshape(state, [1, agent.state_size])  # Reshape to (1, state_size)

    action = agent.act(state)

    if action >= len(eligible_employees):
        print("DQN action index out of bounds. Skipping this complaint.")
        return False, None  

    selected_employee = eligible_employees.iloc[action]
    employee_index = employee_df.index[employee_df["Employee_ID"] == selected_employee["Employee_ID"]][0]
    employee_df.at[employee_index, "Number_of_Customers"] += 1

    assigned_time_slot = assign_time_slot(selected_employee, requested_date)
    if assigned_time_slot:
        employee_df.at[employee_index, "Servicing_Dates"] += f"; {assigned_time_slot}"
        print(f"Complaint assigned to Employee {selected_employee['Employee_ID']} on {assigned_time_slot}.")
        return True, action  
    else:
        print(f"Employee {selected_employee['Employee_ID']} has no available slots.")
        return False, None 


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Initialize the DQN agent
    unique_categories = employee_df["Category"].unique().tolist()
    num_unique_categories = len(unique_categories)
    print(f"Number of unique categories: {num_unique_categories}")
    agent = DQNAgent(state_size=num_unique_categories + 1, action_size=10) 

    # Define training parameters
    num_episodes = 1000
    batch_size = 10

    # Evaluation metrics
    total_reward = 0
    success_count = 0
    losses = []
    action_counts = np.zeros(agent.action_size)

    # Iterate over each complaint in the CSV file
    for index, row in complaints_df.head(200).iterrows():
        complaint_text = row['Complaint'] 
        sentiment_score = get_sentiment_score(complaint_text)
        print(f"Calculated sentiment score for complaint {index + 1}: {sentiment_score:.4f}")

        entities = extract_entities(complaint_text)
        date_context = interpret_date_context(complaint_text, entities)
        issue_category = classify_issue(complaint_text)

        complaint = {"text": complaint_text, "category": issue_category, "sentiment_score": sentiment_score}

        # Extract requested date if available
        requested_date = None
        if "Service Requesting Date" in date_context.values():
            requested_date = list(date_context.keys())[0] 

        # Prepare state
        state = prepare_state(complaint, employee_df)
        done = False

        #timing the assignment
        start_time = time.time()

        # Assign complaint using DQN
        while not done:
            assigned, action = assign_complaint_dqn(agent, complaint, requested_date)

            if assigned:
                reward = 1  # Reward for successful assignment
                success_count += 1
                done = True 
            else:
                reward = -1  # Penalty for no assignment
                done = True 

        # End timing the assignment
        end_time = time.time()
        assignment_time = end_time - start_time

        # Track action distribution
        action_counts[action] += 1

        next_state = prepare_state(complaint, employee_df)

        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # Train the agent if enough experiences are collected
        if len(agent.memory) > batch_size:
            minibatch = random.sample(agent.memory, batch_size)
            for m_state, m_action, m_reward, m_next_state, m_done in minibatch:
                m_state = np.reshape(m_state, [1, agent.state_size])
                m_next_state = np.reshape(m_next_state, [1, agent.state_size])

                target = m_reward
                if not m_done:
                    target += agent.gamma * np.amax(agent.model.predict(m_next_state, verbose=0)[0])
                target_f = agent.model.predict(m_state, verbose=0)
                target_f[0][m_action] = target
                loss = agent.model.fit(m_state, target_f, epochs=1, verbose=0)
                losses.append(loss.history['loss'][0]) 

        #episode results
        print(f"Processed complaint {index + 1}/{len(complaints_df)} - Reward: {reward}, Assignment Time: {assignment_time:.4f} seconds")

        # Decay epsilon after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    # Save the trained model 
    agent.model.save("dqn_model.h5")

    print("\n--- Model Evaluation Metrics ---")
    print(f"Final Success Rate: {success_count / num_episodes:.2%}")
    print(f"Average Loss: {np.mean(losses):.4f}")
    print(f"Action Distribution: {action_counts}")