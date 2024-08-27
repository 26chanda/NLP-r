import json
import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Load role configurations
with open('roles_config.json', 'r') as f:
    roles_config = json.load(f)

# Define roles and languages
roles = list(roles_config['roles'].keys())
languages = ['English', 'Spanish', 'Chinese', 'German', 'French', 'Malayalam', 'Telugu', 'Kannada', 'Gujarati', 'Punjabi', 'Tamil', 'Konkani']

# Define the list of departments
departments = [
    'Human Resources (HR)', 'Finance', 'Operations', 'Sales', 'Marketing', 
    'Information Technology (IT)', 'Customer Service', 'Research and Development (R&D)', 
    'Legal', 'Procurement', 'Product Management', 'Business Development', 
    'Corporate Strategy', 'Administration', 'Engineering', 'Compliance and Risk Management', 
    'Health and Safety', 'Public Relations (PR)', 'Training and Development', 
    'Corporate Communications'
]

# Define the list of responsibilities
responsibilities = [
    'Customer Segmentation', 'Campaign Effectiveness', 'Lead Scoring', 
    'Customer Churn Prediction', 'Financial Forecasting', 'Risk Management', 
    'Cost Optimization', 'Compliance and Auditing', 'Employee Retention', 
    'Recruitment and Talent Acquisition', 'Performance Management', 
    'Training and Development', 'Demand Forecasting', 'Supply Chain Optimization', 
    'Quality Control', 'Resource Allocation', 'Customer Support Automation', 
    'Sentiment Analysis', 'Service Quality Monitoring', 'Complaint Resolution', 
    'Feature Prioritization', 'User Behavior Analysis', 'Innovation and R&D', 
    'Product Lifecycle Management', 'Network Security', 'Data Privacy', 
    'System Performance', 'Incident Response', 'Innovation Management', 
    'Product Testing and Validation', 'Market Research', 'Technology Adoption', 
    'Market Expansion', 'Mergers and Acquisitions', 'Competitive Analysis', 
    'Business Model Innovation', 'Regulatory Compliance', 'Contract Management', 
    'Intellectual Property Management', 'Litigation and Dispute Resolution', 
    'Collecting, cleaning, and ensuring the accuracy and integrity of data.', 
    'Monitoring and improving data quality.', 
    'Developing, maintaining, and working with databases, data warehousing, ETL processes, and data pipelines.', 
    'Automating data collection and reporting processes.', 
    'Ensuring data security and compliance with regulatory standards.', 
    'Conducting A/B testing and experimental design for data-driven decision-making.', 
    'Performing statistical analysis to identify trends and patterns in data.', 
    'Developing and refining predictive models, forecasting trends, and guiding strategic decisions.', 
    'Optimizing algorithms and processes for scalability and efficiency.', 
    'Deploying models into production, monitoring performance, and iterating based on feedback.', 
    'Implementing continuous integration and continuous deployment (CI/CD) pipelines for ML models.', 
    'Designing interactive dashboards to communicate insights and support decision-making.', 
    'Interpreting data results, generating reports, and presenting actionable insights to stakeholders.', 
    'Collaborating with other departments to understand data needs and inform business decisions.', 
    'Developing and maintaining data products or tools for internal and external use.', 
    'Documenting data processes, methodologies, and quality improvements.'
]

# Function to create role descriptions
def create_role_description(role):
    return roles_config['roles'][role]['description']

# Function to generate project descriptions
def create_project(role):
    project_templates = {
        'Data Scientist': [
            'Developed a predictive model to forecast sales using {tool}.',
            'Conducted sentiment analysis on customer reviews using {tool}.',
            'Implemented a recommendation system with {tool}.'
        ],
        'Data Engineer': [
            'Built a data pipeline using {tool} to process large datasets.',
            'Designed a data warehouse architecture using {tool}.',
            'Optimized data retrieval processes using {tool}.'
        ],
        'Data Analyst': [
            'Created dashboards using {tool} to visualize sales data.',
            'Analyzed customer segmentation with {tool}.',
            'Generated reports on market trends using {tool}.'
        ],
        'Business Analyst': [
            'Led a project to improve business processes using {tool}.',
            'Analyzed business performance using {tool}.',
            'Developed a business strategy using insights from {tool}.'
        ],
        'ML Engineer': [
            'Deployed a deep learning model using {tool}.',
            'Fine-tuned a neural network with {tool} for better accuracy.',
            'Implemented real-time data processing with {tool}.'
        ]
    }
    project_template = random.choice(project_templates[role])
    tool = random.choice(roles_config['roles'][role]['tools'])
    return project_template.format(tool=tool)

# Function to generate multiple educational backgrounds based on role
def create_education(role):
    education_options = {
        'Data Scientist': ['PhD in Computer Science', 'MSc in Data Science', 'BSc in Statistics'],
        'Data Engineer': ['BSc in Computer Science', 'BSc in Information Technology', 'MSc in Data Engineering'],
        'Data Analyst': ['BSc in Economics', 'BSc in Business Administration', 'BSc in Mathematics'],
        'Business Analyst': ['MBA', 'BBA', 'MSc in Business Analytics'],
        'ML Engineer': ['MSc in Machine Learning', 'BSc in Computer Science', 'BSc in Artificial Intelligence']
    }
    return random.sample(education_options[role], random.randint(1, 3))  # Select 1 to 3 education qualifications

# Function to generate multiple experiences based on role
def create_experience(role):
    experience_options = {
        'Data Scientist': ['5+ years in data science', '3+ years in machine learning', '4+ years in AI research'],
        'Data Engineer': ['4+ years in data engineering', '3+ years in ETL processes', '5+ years in database management'],
        'Data Analyst': ['3+ years in data analysis', '2+ years in business intelligence', '4+ years in market analysis'],
        'Business Analyst': ['5+ years in business strategy', '3+ years in financial analysis', '4+ years in business consulting'],
        'ML Engineer': ['4+ years in ML model development', '3+ years in deep learning', '5+ years in AI system deployment']
    }
    return random.sample(experience_options[role], random.randint(1, 3))  # Select 1 to 3 experiences

# Function to generate multiple certifications based on role
def create_certifications(role):
    certifications_options = {
        'Data Scientist': ['Certified Data Scientist', 'TensorFlow Developer Certification', 'Microsoft Certified: Azure AI Engineer'],
        'Data Engineer': ['Google Cloud Professional Data Engineer', 'AWS Certified Big Data', 'Microsoft Certified: Azure Data Engineer Associate'],
        'Data Analyst': ['Google Data Analytics Certificate', 'Microsoft Certified: Data Analyst Associate', 'Tableau Desktop Specialist'],
        'Business Analyst': ['Certified Business Analysis Professional (CBAP)', 'PMI Professional in Business Analysis (PMI-PBA)', 'Certified ScrumMaster (CSM)'],
        'ML Engineer': ['TensorFlow Developer Certification', 'AWS Certified Machine Learning', 'Google Cloud Professional Machine Learning Engineer']
    }
    return random.sample(certifications_options[role], random.randint(1, 3))  # Select 1 to 3 certifications

# Function to generate responsibilities based on role
def create_responsibilities():
    return random.sample(responsibilities, random.randint(3, 7))  # Select 3 to 7 responsibilities

# Generate dataset
data = []
for _ in range(600000):
    name = fake.name()
    language = random.choice(languages)
    company = fake.company()
    department = random.choice(departments)  # Select a department from the predefined list
    role = random.choice(roles)
    role_description = create_role_description(role)
    project = create_project(role)
    location = fake.city()
    education = "; ".join(create_education(role))  # Join multiple educational qualifications with a semicolon
    experience = "; ".join(create_experience(role))  # Join multiple experiences with a semicolon
    certifications = "; ".join(create_certifications(role))  # Join multiple certifications with a semicolon
    responsibilities_list = "; ".join(create_responsibilities())  # Join responsibilities with a semicolon
    
    data.append({
        'Name': name,
        'Language': language,
        'Company': company,
        'Department': department,
        'Role': role,
        'Role Description': role_description,
        'Projects': project,
        'Location': location,
        'Education': education,
        'Experience': experience,
        'Certifications': certifications,
        'Responsibilities': responsibilities_list
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_dataset.csv', index=False)
 