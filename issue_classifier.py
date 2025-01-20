from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example pre-trained classifier (using a simple approach for demonstration)
def classify_issue_type(description):
    issue_types = ["Billing", "Maintenance", "Vehicle Performance", "Sales and Purchase", "Customer Service", 
                   "Safety Recalls", "Technology and Connectivity", "Legal and Regulatory", "Environmental Concerns"]
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(issue_types)
    model = MultinomialNB()
    model.fit(X, issue_types)

    description_vectorized = vectorizer.transform([description])
    predicted_issue_type = model.predict(description_vectorized)
    return predicted_issue_type[0]
