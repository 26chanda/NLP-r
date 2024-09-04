import json
import pandas as pd
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

# Define responsibilities for each role
responsibilities_by_role = {
    'Data Scientist': [
        'Collecting, cleaning, and ensuring the accuracy and integrity of data.',
        'Developing, maintaining, and working with databases, data warehousing, ETL processes, and data pipelines.',
        'Automating data collection and reporting processes.',
        'Conducting A/B testing and experimental design for data-driven decision-making.',
        'Performing statistical analysis to identify trends and patterns in data.'
    ],
    'Data Engineer': [
        'Designing, building, and maintaining large-scale data systems.',
        'Developing, testing, and deploying data pipelines and architectures.',
        'Ensuring data quality, security, and compliance with regulatory standards.',
        'Optimizing data processing and storage systems for scalability and efficiency.',
        'Collaborating with data scientists to develop and deploy machine learning models.'
    ],
    'Data Analyst': [
        'Analyzing and interpreting complex data sets to identify trends and patterns.',
        'Developing and maintaining databases, data warehouses, and data visualization tools.',
        'Creating reports and dashboards to communicate insights to stakeholders.',
        'Collaborating with business stakeholders to understand data needs and inform business decisions.',
        'Identifying areas for process improvement and implementing changes.'
    ],
    'Business Analyst': [
        'Analyzing business needs and identifying opportunities for improvement.',
        'Developing and implementing business strategies and solutions.',
        'Collaborating with stakeholders to gather requirements and develop solutions.',
        'Creating business cases and cost-benefit analyses to support proposals.',
        'Communicating insights and recommendations to stakeholders.'
    ],
    'ML Engineer': [
        'Designing, developing, and deploying machine learning models and algorithms.',
        'Collaborating with data scientists to develop and deploy models.',
        'Ensuring model performance, scalability, and reliability.',
        'Developing and maintaining data pipelines and architectures for ML models.',
        'Optimizing model training and deployment processes for efficiency and accuracy.'
    ]
}

# Define the mapping of projects to departments
department_projects = {
    'Human Resources (HR)': [
        'Employee Retention', 'Recruitment and Talent Acquisition', 'Performance Management', 'Training and Development'
    ],
    'Finance': [
        'Fraud Detection', 'Financial Forecasting', 'Risk Management', 'Cost Optimization'
    ],
    'Operations': [
        'Supply Chain Optimization', 'Quality Control', 'Resource Allocation', 'Incident Response'
    ],
    'Sales': [
        'Customer Segmentation', 'Campaign Effectiveness', 'Lead Scoring', 'Market Expansion'
    ],
    'Marketing': [
        'Market Research', 'Competitive Analysis', 'Brand Management', 'Product Lifecycle Management'
    ],
    'Information Technology (IT)': [
        'Network Security', 'Data Privacy', 'System Performance', 'Robotic Process Automation'
    ],
    'Customer Service': [
        'Customer Support Automation', 'Complaint Resolution', 'Sentiment Analysis', 'Service Quality Monitoring'
    ],
    'Research and Development (R&D)': [
        'Innovation and R&D', 'Product Testing and Validation', 'Technology Adoption', 'Product Lifecycle Management'
    ],
    'Legal': [
        'Regulatory Compliance', 'Contract Management', 'Intellectual Property Management', 'Litigation and Dispute Resolution'
    ],
    'Procurement': [
        'Supplier Relationship Management', 'Contract Negotiation', 'Cost Optimization', 'Risk Management'
    ],
    'Product Management': [
        'Feature Prioritization', 'User Behavior Analysis', 'Product Lifecycle Management', 'Market Expansion'
    ],
    'Business Development': [
        'Mergers and Acquisitions', 'Market Research', 'Competitive Analysis', 'Technology Adoption'
    ],
    'Corporate Strategy': [
        'Business Model Innovation', 'Corporate Governance', 'Strategic Planning', 'Innovation Management'
    ],
    'Administration': [
        'Facility Management', 'Office Operations', 'Supply Management', 'Vendor Coordination'
    ],
    'Engineering': [
        'System Performance', 'Robotic Process Automation', 'Quality Control', 'Resource Allocation'
    ],
    'Compliance and Risk Management': [
        'Regulatory Compliance', 'Risk Management', 'Incident Response', 'Data Privacy'
    ],
    'Health and Safety': [
        'Workplace Safety Programs', 'Incident Response', 'Health Compliance', 'Risk Management'
    ],
    'Public Relations (PR)': [
        'Brand Management', 'Crisis Communication', 'Media Relations', 'Corporate Communication'
    ],
    'Training and Development': [
        'Employee Training Programs', 'Leadership Development', 'Skill Gap Analysis', 'Performance Management'
    ],
    'Corporate Communications': [
        'Internal Communication Strategy', 'Public Relations Campaigns', 'Corporate Branding', 'Stakeholder Engagement'
    ]
}
# Adjusted create_project function to return multiple projects
def create_projects(department):
    projects = department_projects[department]
    
    num_projects = random.randint(2, 5)
    selected_projects = random.sample(projects, min(num_projects, len(projects)))
    
    return selected_projects


# Define a list of possible educational backgrounds
educational_backgrounds = [
    'Business Analytics', 'Data Analytics', 'Computer Science', 'Software Engineering', 
    'Information Technology', 'Cybersecurity', 'Data Science', 'Artificial Intelligence',
    'Statistical Science', 'Biostatistics', 'Supply Chain', 'Finance',
    'Information Technology & Management', 'Bioinformatics', 'Information Systems',
    'Biology', 'Biotechnology', 'Molecular Biology', 'Genetics', 'Microbiology',
    'Zoology', 'Botany', 'Biochemistry', 'Environmental Science', 'Ecology',
    'Chemistry', 'Physics', 'Astronomy', 'Accounting', 'Earth Science', 'Geology',
    'Meteorology', 'Oceanography', 'Mathematics', 'Applied Mathematics', 'Statistics',
    'Actuarial Science', 'Biomedical Sciences', 'Nursing', 'Health Informatics',
    'Epidemiology', 'Medical Technology', 'Cognitive Science', 'Neuroscience',
    'Experimental Psychology', 'Mechanical Engineering', 'Electrical Engineering',
    'Civil Engineering', 'Chemical Engineering', 'Aerospace Engineering',
    'Industrial Engineering', 'Materials Science and Engineering',
    'Environmental Engineering', 'Biomedical Engineering', 'Agricultural Engineering',
    'Nuclear Engineering', 'Petroleum Engineering', 'Mining Engineering',
    'Systems Engineering', 'Pure Mathematics', 'Abstract Mathematics',
    'Theoretical Mathematics', 'Computational Mathematics', 'Environmental Chemistry',
    'Environmental Biology', 'Environmental Health', 'Robotics Engineering',
    'Mechatronics', 'Energy Engineering', 'Sustainable Energy', 'AI Research',
    'Machine Learning', 'Cognitive Systems', 'Genetic Engineering', 'Biotechnology',
    'Management'
]

# Function to generate multiple educational backgrounds randomly
def create_education():
    return random.sample(educational_backgrounds, random.randint(1, 3))

experience_ranges = {
    '<1': '<1 year',
    '1-3': '1-3 years',
    '3-5': '3-5 years',
    '5-7': '5-7 years',
    '7-10': '7-10 years',
    '10+': '10+ years'
}

# Function to generate responsibilities based on role
def create_responsibilities(role):
    num_responsibilities = min(len(responsibilities_by_role[role]), random.randint(3, 5))
    return random.sample(responsibilities_by_role[role], num_responsibilities)

# Generate dataset
data = []
for _ in range(600000):
    role = random.choice(roles)
    department = random.choice(departments)  # Randomly assign a department to the role
    name = fake.name()
    language = random.choice(languages)
    company = fake.company()
    projects = "; ".join(create_projects(department))  # Get multiple projects for the department
    location = fake.city()
    education = "; ".join(create_education())  # Join multiple educational qualifications with a semicolon
    experience = experience_ranges[random.choice(list(experience_ranges.keys()))]  # Select experience range
    responsibilities_list = ";".join(create_responsibilities(role))

    data.append({
        'Name': name,
        'Language': language,
        'Company': company,
        'Department': department,
        'Role': role,
        'Projects': projects,  
        'Location': location,
        'Education': education,
        'Experience': experience,
        'Responsibilities': responsibilities_list
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_dataset.csv', index=False)
