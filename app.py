import imaplib 
import email
from email.header import decode_header
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import torch
import numpy as np
import re
from collections import deque
from datetime import datetime, timedelta
import random
import streamlit as st
import smtplib
from email.message import EmailMessage
from email.utils import parseaddr
import requests

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# ---------------------------
# MongoDB Setup
# ---------------------------
from pymongo import MongoClient

# Connect to the MongoDB server (adjust the connection string as needed)
client = MongoClient("mongodb://localhost:27017/")
db = client["MAIN_EL"]
employee_collection = db["employee_db"]

# ========================
# Streamlit Interface Setup
# ========================
def create_streamlit_interface():
    st.title("Customer Complaint Assignment Tracker")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Live Assignments", "Assignment History"])
    
    with tab1:
        st.header("Recent Complaint Assignments")
        st.empty()
        
    with tab2:
        st.header("Historical Assignments")
        # Fetch all employees from MongoDB for display
        employees = list(employee_collection.find({}, {"_id": 0}))
        if employees:
            st.dataframe(employees)

def update_assignment_display(complaint_text, complaint_id, employee_id, assigned_time, category):
    """Update the Streamlit display with new assignment details"""
    with st.container():
        st.success(f"New Complaint Assigned! - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Assignment Details")
            st.write(f"🎫 Complaint ID: {complaint_id}")
            st.write(f"👤 Assigned to: Employee {employee_id}")
            st.write(f"📅 Time Slot: {assigned_time}")
            st.write(f"🏷 Category: {category}")
            
        with col2:
            st.markdown("Complaint Content")
            st.text_area("Complaint", complaint_text, height=150, disabled=True, key=f"complaint_{complaint_id}")
        
        st.markdown("---")

# ========================
# Email Configuration
# ========================
EMAIL_HOST = "imap.gmail.com"
EMAIL_USER = "mainel5thsem@gmail.com"
EMAIL_PASS = "xvdr xkdo qufb ghxi"  # Use your actual credentials or app password

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# ========================
# NLP and ML Model Initialization
# ========================
model_name = "assemblyai/distilbert-base-uncased-sst2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
nlp = spacy.load("en_core_web_sm")

# Define time slots
TIME_SLOTS = ["10AM-11AM", "11AM-12PM", "12PM-1PM", "2PM-6PM"]

# ========================
# DQN Agent Definition
# ========================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  
        self.epsilon = 1.0  
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
        state = np.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

# ---------------------------
# Update Agent's State Size Dynamically
# ---------------------------
unique_categories = employee_collection.distinct("Category")
state_size = len(unique_categories) + 1  # +1 for the sentiment score
print("Unique Categories:", unique_categories)
print("State Size set to:", state_size)

agent = DQNAgent(state_size=state_size, action_size=10)
agent.model = load_model(r"C:\Users\Supriya S\OneDrive\Desktop\MAIN_EL\dqn_model.h5", custom_objects={"mse": MeanSquaredError()})
agent.model.compile(loss='mse', optimizer=Adam(learning_rate=agent.learning_rate), metrics=[MeanSquaredError()])

# ========================
# Issue Categories and Helper Functions
# ========================
ISSUE_CATEGORIES = {
    "Billing": ["invoice", "billing", "bill", "charges", "cost", "payment", "duplicate", "processing", "extra", "charge"],
    "Technical": ["engine", "brakes", "noise", "battery", "window", "broken", "engine stalling", "transmission", "overheating"],
    "Maintenance": ["service", "schedule", "repair", "replace", "routine", "parts", "availability", "lock"],
    "Vehicle Performance": ["engine", "stalling", "noise", "shifting", "brakes", "vibration", "transmission"],
    "Sales and Purchase": ["delivery", "financing", "loan", "payment", "damage", "warranty"],
    "Customer Service": ["communication", "response", "representative", "dealer", "advertising", "complaints"],
    "Safety Recalls": ["recall", "repair", "safety", "notification"],
    "Technology and Connectivity": ["infotainment", "bluetooth", "ADAS", "autonomous", "privacy", "cybersecurity"],
    "Legal and Regulatory": ["lemon laws", "safety regulations", "compliance"],
    "Environmental Concerns": ["fuel", "electric", "emissions", "range", "emission"]
}

def get_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                return part.get_payload(decode=True).decode()
    else:
        return msg.get_payload(decode=True).decode()

def process_email(raw_email):
    msg = email.message_from_bytes(raw_email)
    subject = decode_header(msg["Subject"])[0][0]
    if isinstance(subject, bytes):
        subject = subject.decode()
    body = get_email_body(msg)
    sender = msg.get("From")
    sender_email = parseaddr(sender)[1]
    complaint_text = f"{subject}\n{body}"
    return complaint_text, sender_email

def fetch_emails():
    mail = imaplib.IMAP4_SSL(EMAIL_HOST)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")
    
    status, messages = mail.search(None, "UNSEEN")
    if status != "OK":
        print("Error checking emails")
        return []
    
    email_ids = messages[0].split()
    complaints = []
    
    for email_id in email_ids:
        status, data = mail.fetch(email_id, "(RFC822)")
        if status != "OK":
            continue
        raw_email = data[0][1]
        complaint_text, sender_email = process_email(raw_email)
        complaints.append({
            "text": complaint_text,
            "sender_email": sender_email
        })
        mail.store(email_id, '+FLAGS', '\\Seen')
    
    mail.close()
    mail.logout()
    return complaints

def get_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    negative_score, positive_score = probs.squeeze().tolist()
    sentiment_score = (positive_score - negative_score) * 10
    return sentiment_score

def extract_entities(doc):
    parsed_doc = nlp(doc)
    entities = {ent.label_: ent.text for ent in parsed_doc.ents}
    return entities

def interpret_date_context(doc, entities):
    date_patterns = {
        "Last Serviced Date": r"\b(last service|serviced on|last maintenance)\b",
        "Service Requesting Date": r"\b(request service|book for|schedule|appointment|visit on|schedule service on)\b",
        "Issue Start Date": r"\b(since|started on|from|began|yesterday)\b",
        "Invoice/Billing Date": r"\b(invoice|billing|charged on|last invoice|payment due)\b"
    }

    date_results = {}

    today = datetime.today().strftime("%Y-%m-%d")
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    if "today" in doc.lower():
        date_results[today] = "Service Requesting Date"
    if "tomorrow" in doc.lower():
        date_results[tomorrow] = "Service Requesting Date"

    # Extract date entities from NLP
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

    # Common date formats to recognize
    date_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
        "%m-%d-%Y", "%m/%d/%Y"
    ]

    found_dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', doc)
    for found_date in found_dates:
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(found_date, fmt)
                date_results[parsed_date.strftime("%Y-%m-%d")] = "Service Requesting Date"
                break  # Exit the loop once a valid date is found
            except ValueError:
                continue

    return date_results

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

# ========================
# Modified assign_time_slot Function
# ========================
def assign_time_slot(employee, requested_date=None):
    servicing_dates = employee.get("Servicing_Dates", "")
    if isinstance(servicing_dates, list):
        existing_slots = [slot.strip() for slot in servicing_dates]
    else:
        existing_slots = [slot.strip() for slot in servicing_dates.split(";") if slot]
    today = datetime.today()
    current_time = today.time()  # Get the current time

    # Define time slot boundaries
    time_slot_boundaries = {
        "10AM-11AM": (datetime.strptime("10:00 AM", "%I:%M %p").time(), datetime.strptime("11:00 AM", "%I:%M %p").time()),
        "11AM-12PM": (datetime.strptime("11:00 AM", "%I:%M %p").time(), datetime.strptime("12:00 PM", "%I:%M %p").time()),
        "12PM-1PM": (datetime.strptime("12:00 PM", "%I:%M %p").time(), datetime.strptime("1:00 PM", "%I:%M %p").time()),
        "2PM-6PM": (datetime.strptime("2:00 PM", "%I:%M %p").time(), datetime.strptime("6:00 PM", "%I:%M %p").time())
    }

    # Check for requested date
    if requested_date:
        requested_date_obj = datetime.strptime(requested_date, "%Y-%m-%d")
        for slot in TIME_SLOTS:
            full_slot = f"{requested_date_obj.strftime('%Y-%m-%d')} {slot}"
            if full_slot not in existing_slots:
                slot_start, _ = time_slot_boundaries[slot]
                if slot_start > current_time:  # Ensure the slot is after the current time
                    return full_slot

    # Check for today's slots
    for slot in TIME_SLOTS:
        full_slot = f"{today.strftime('%Y-%m-%d')} {slot}"
        if full_slot not in existing_slots:
            slot_start, _ = time_slot_boundaries[slot]
            if slot_start > current_time:  # Ensure the slot is after the current time
                return full_slot

    # If no slots available today, check for tomorrow
    tomorrow = today + timedelta(days=1)
    for slot in TIME_SLOTS:
        full_slot = f"{tomorrow.strftime('%Y-%m-%d')} {slot}"
        if full_slot not in existing_slots:
            return full_slot  # Return the first available slot for tomorrow

    return None  # No available slots

def prepare_state(complaint, employees):
    unique_categories = employee_collection.distinct("Category")
    category_encoding = np.zeros(len(unique_categories))
    if complaint["category"] in unique_categories:
        category_encoding[unique_categories.index(complaint["category"])] = 1
    else:
        print(f"Warning: Category '{complaint['category']}' not found in employee categories.")
    sentiment_score = (complaint["sentiment_score"] + 10) / 20
    return np.concatenate([category_encoding, [sentiment_score]])

def generate_complaint_id():
    all_docs = employee_collection.find({}, {"Assigned_Complaint_IDs": 1, "_id": 0})
    all_complaint_ids = []
    for doc in all_docs:
        ids = doc.get("Assigned_Complaint_IDs", [])
        if isinstance(ids, list):
            all_complaint_ids.extend(ids)
    if not all_complaint_ids:
        return "C001"
    last_id_num = max([int(x[1:]) for x in all_complaint_ids if x.startswith("C")], default=0)
    return f"C{last_id_num + 1:03d}"

def assign_complaint_dqn(agent, complaint, requested_date=None):
    eligible_employees = list(employee_collection.find({"Category": complaint["category"]}))
    if not eligible_employees:
        print("No eligible employees available for this complaint.")
        return False
    agent.action_size = len(eligible_employees)
    state = prepare_state(complaint, eligible_employees)
    action = agent.act(state)
    if action >= len(eligible_employees):
        print("DQN action index out of bounds. Skipping this complaint.")
        return False
    selected_employee = eligible_employees[action]
    assigned_time_slot = assign_time_slot(selected_employee, requested_date)
    if assigned_time_slot:
        complaint_id = generate_complaint_id()
        update_result = employee_collection.update_one(
            { "Employee_ID": selected_employee["Employee_ID"] },
            {
                "$inc": { "Number_of_Customers": 1 },
                "$push": { "Servicing_Dates": assigned_time_slot, "Assigned_Complaint_IDs": complaint_id }
            }
        )
        if update_result.modified_count:
            print(f"Complaint {complaint_id} assigned to Employee {selected_employee['Employee_ID']} on {assigned_time_slot}.")
            return {
                "complaint_id": complaint_id,
                "employee_id": selected_employee["Employee_ID"],
                "employee_name": selected_employee.get("Employee_Name", "Unknown"),  # New field added
                "assigned_time": assigned_time_slot
            }
        else:
            print("Failed to update employee record in MongoDB.")
            return False
    else:
        print(f"Employee {selected_employee['Employee_ID']} has no available slots.")
        return False

# ========================
# Gemini API Integration (Updated)
# ========================
def call_gemini_api(prompt):
    # Replace YOUR_ACTUAL_API_KEY with your actual Gemini API key.
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyDFBU6ElTE7DpYguRzj-jacDqYgHfdmtTs"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    print("Gemini API status:", response.status_code)
    print("Gemini API response:", response.text)
    
    if response.status_code == 200:
        data = response.json()
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            generated_text = None
            if isinstance(candidate, dict):
                # Check if the candidate has a "content" field (new structure)
                if "content" in candidate:
                    content = candidate["content"]
                    parts = content.get("parts", [])
                    if parts and isinstance(parts, list):
                        generated_text = parts[0].get("text")
                else:
                    # Fallback to previous keys if they exist
                    generated_text = candidate.get("output") or candidate.get("text")
            else:
                generated_text = candidate
            
            if generated_text and generated_text.strip():
                return generated_text.strip()
            else:
                return None
    else:
        print(f"Gemini API Error {response.status_code}: {response.text}")
        return None


# Compose reply email using Gemini-generated content.
def compose_reply_email(complaint_id, employee_id, employee_name, assigned_time, category, original_complaint_text):
    prompt = (
        "You are a customer support assistant. A customer has sent the following complaint:\n\n"
        f"\"{original_complaint_text}\"\n\n"
        f"This complaint has been assigned the ID {complaint_id} and will be handled by Employee {employee_name} (ID: {employee_id}) "
        f"specializing in {category}. The service is scheduled for {assigned_time}.\n\n"
        "Please generate a professional, empathetic, and detailed email response confirming the assignment and "
        "providing the necessary details to the customer. Do not state that someone will visit the customer, do not mention "
        "anything about arriving at a location, and do not include any request for a reply or further information. "
        "Also, avoid using bold formatting. Instead of placeholders like [Customer Name] or [Your Company Name], "
        "simply refer to the recipient as 'customer'."
    )

    generated_body = call_gemini_api(prompt)

    if not generated_body:
        generated_body = (
            f"Dear Customer,\n\n"
            f"Thank you for contacting us. We would like to inform you that your complaint (ID: {complaint_id}) "
            f"regarding {category} has been assigned to Employee {employee_name} (ID: {employee_id}). "
            f"The service is scheduled for {assigned_time}.\n\n"
            f"We appreciate your patience and will ensure that your concern is addressed promptly.\n\n"
            f"Sincerely,\n"
            f"Customer Support Team"
        )

    subject = f"Complaint Assignment Confirmation: {complaint_id}"
    return subject, generated_body

def send_reply_email(to_address, subject, body):
    msg = EmailMessage()
    msg["From"] = EMAIL_USER
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print(f"Reply email sent to {to_address}")
    except Exception as e:
        print(f"Failed to send reply email to {to_address}: {str(e)}")

# ========================
# Main Execution Loop with Streamlit Integration
# ========================
if __name__ == "__main__":
    create_streamlit_interface()
    status_placeholder = st.empty()
    
    while True:
        try:
            status_placeholder.text("Checking for new complaints...")
            complaints_list = fetch_emails()
            if not complaints_list:
                time.sleep(1)
                continue
                
            for complaint_data in complaints_list:
                complaint_text = complaint_data["text"]
                sender_email = complaint_data["sender_email"]
                
                # Get sentiment score and print it to the terminal.
                sentiment_score = get_sentiment_score(complaint_text)
                print(f"Sentiment Score: {sentiment_score}")  # <-- Printing sentiment score here

                entities = extract_entities(complaint_text)
                date_context = interpret_date_context(complaint_text, entities)
                issue_category = classify_issue(complaint_text)
                complaint = {
                    "text": complaint_text,
                    "category": issue_category,
                    "sentiment_score": sentiment_score
                }
                requested_date = None
                if "Service Requesting Date" in date_context.values():
                    requested_date = list(date_context.keys())[0]
                assignment = assign_complaint_dqn(agent, complaint, requested_date)
                if assignment:
                    update_assignment_display(
                        complaint_text,
                        assignment["complaint_id"],
                        assignment["employee_id"],
                        assignment["assigned_time"],
                        issue_category
                    )
                    subject, body = compose_reply_email(
                        assignment["complaint_id"],
                        assignment["employee_id"],
                        assignment["employee_name"],  # Passing employee_name here
                        assignment["assigned_time"],
                        issue_category,
                        complaint_text
                    )
                    send_reply_email(sender_email, subject, body)

            time.sleep(1)
        except KeyboardInterrupt:
            st.warning("Stopping email monitoring...")
            break
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
            time.sleep(5)
