from flask import Flask, render_template, request, redirect, url_for
#from flask_pymongo import PyMongo
#from sentiment_analysis import get_sentiment
from issue_classifier import classify_issue_type
#from employee_routing import route_issue

app = Flask(__name__)
'''
# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/issueRoutingDB"
#mongo = PyMongo(app)
'''
@app.route('/')
def home():
    return render_template('index.html')
'''
@app.route('/submit_issue', methods=['POST'])
def submit_issue():
    description = request.form['description']
    sentiment_score = get_sentiment(description)
    issue_type = classify_issue_type(description)
    
    # Get employee data from MongoDB
    employees = list(mongo.db.employees.find({'department': issue_type}))
    
    # Route issue to an employee
    assigned_employee = route_issue(description, employees)
    
    # Save issue in the database
    mongo.db.issues.insert_one({
        'description': description,
        'sentiment_score': sentiment_score,
        'issue_type': issue_type,
        'assigned_employee': assigned_employee
    })
    
    return redirect(url_for('result', issue_id=str(assigned_employee)))

@app.route('/result/<issue_id>')
def result(issue_id):
    issue = mongo.db.issues.find_one({'assigned_employee': issue_id})
    return render_template('result.html', issue=issue)
'''
if __name__ == '__main__':
    app.run(debug=True)

