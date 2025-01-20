from flask_pymongo import PyMongo

def get_employee_data(mongo):
    return list(mongo.db.employees.find())

def insert_issue(mongo, issue_data):
    mongo.db.issues.insert_one(issue_data)
