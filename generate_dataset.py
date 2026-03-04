"""
Generate synthetic employee dataset for Nikhil's Software Development Company.
100 employees, 30 columns, designed so:
- Only 3 employees have ALL 3 required skills (Python + English + French)
- Heavy weightage on software developers
- Realistic correlations for clustering, classification, and association rule mining
"""
import pandas as pd
import numpy as np

np.random.seed(42)

n = 100

# --- EMPLOYEE IDs & NAMES ---
ids = [f"E{str(i+1).zfill(3)}" for i in range(n)]
first_names = [
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Reyansh","Sai","Ayaan","Krishna","Ishaan",
    "Dhruv","Kabir","Ritvik","Anirudh","Aryan","Rohan","Harsh","Kunal","Nitin","Sahil",
    "Priya","Ananya","Diya","Myra","Sara","Anika","Aadhya","Isha","Kavya","Meera",
    "Ravi","Amit","Suresh","Vikram","Deepak","Rahul","Gaurav","Manish","Tarun","Vishal",
    "Sneha","Pooja","Nisha","Swati","Rekha","Divya","Neha","Shruti","Pallavi","Komal",
    "Pranav","Siddharth","Varun","Akash","Mohit","Tushar","Ankit","Rajesh","Pankaj","Sumit",
    "Anjali","Ritu","Simran","Tanvi","Bhavna","Jyoti","Kiran","Madhu","Lata","Sunita",
    "Yash","Chirag","Parth","Dev","Krish","Om","Jay","Neil","Raj","Ajay",
    "Megha","Tanya","Aditi","Parul","Shweta","Sonal","Richa","Geeta","Seema","Usha",
    "Hemant","Lalit","Vinod","Manoj","Ashok","Dinesh","Naveen","Sanjay","Brijesh","Ramesh"
]
np.random.shuffle(first_names)

# --- DEPARTMENTS & ROLES ---
# Heavy developer weightage: ~55 devs, 15 testers, 10 PM, 10 admin, 5 devops, 5 UI/UX
dept_role_map = {
    'Development': ['Senior Developer', 'Mid Developer', 'Junior Developer'],
    'Testing': ['Senior Tester', 'Junior Tester'],
    'Project Management': ['Project Manager', 'Scrum Master'],
    'Administration': ['Admin Executive', 'HR Executive'],
    'DevOps': ['DevOps Engineer'],
    'UI/UX': ['UI/UX Designer']
}

departments = (
    ['Development'] * 55 +
    ['Testing'] * 15 +
    ['Project Management'] * 10 +
    ['Administration'] * 10 +
    ['DevOps'] * 5 +
    ['UI/UX'] * 5
)
np.random.shuffle(departments)

roles = []
for dept in departments:
    possible_roles = dept_role_map[dept]
    if dept == 'Development':
        role = np.random.choice(possible_roles, p=[0.25, 0.45, 0.30])
    else:
        role = np.random.choice(possible_roles)
    roles.append(role)

# --- AGE ---
ages = np.random.randint(22, 52, size=n)
# Make seniors older
for i in range(n):
    if 'Senior' in roles[i]:
        ages[i] = np.random.randint(30, 52)
    elif 'Junior' in roles[i]:
        ages[i] = np.random.randint(22, 32)

# --- GENDER ---
genders = np.random.choice(['Male', 'Female'], size=n, p=[0.62, 0.38])

# --- JOINING YEAR ---
# Company started 2019, all hired in 2019
joining_months = np.random.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], size=n)
joining_dates = [f"2019-{str(list(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']).index(m)+1).zfill(2)}-{np.random.randint(1,28):02d}" for m in joining_months]

# --- SALARY (Monthly INR) ---
salary_map = {
    'Senior Developer': (80000, 150000),
    'Mid Developer': (50000, 90000),
    'Junior Developer': (25000, 55000),
    'Senior Tester': (60000, 100000),
    'Junior Tester': (25000, 50000),
    'Project Manager': (70000, 130000),
    'Scrum Master': (60000, 100000),
    'Admin Executive': (20000, 45000),
    'HR Executive': (30000, 55000),
    'DevOps Engineer': (60000, 110000),
    'UI/UX Designer': (45000, 85000),
}
salaries = [np.random.randint(*salary_map[r]) for r in roles]

# ============================================================
# SKILLS - THE CRITICAL PART
# Required skills: Python, English, French
# Only 3 employees should have ALL 3
# ============================================================

# Technical skills
skill_python = np.zeros(n, dtype=int)
skill_java = np.zeros(n, dtype=int)
skill_sql = np.zeros(n, dtype=int)
skill_javascript = np.zeros(n, dtype=int)
skill_csharp = np.zeros(n, dtype=int)

# Language skills
lang_english = np.zeros(n, dtype=int)
lang_french = np.zeros(n, dtype=int)
lang_hindi = np.zeros(n, dtype=int)

# Assign Python broadly to developers
for i in range(n):
    if departments[i] == 'Development':
        skill_python[i] = np.random.choice([0,1], p=[0.35, 0.65])
        skill_java[i] = np.random.choice([0,1], p=[0.45, 0.55])
        skill_javascript[i] = np.random.choice([0,1], p=[0.50, 0.50])
        skill_sql[i] = np.random.choice([0,1], p=[0.40, 0.60])
        skill_csharp[i] = np.random.choice([0,1], p=[0.70, 0.30])
    elif departments[i] == 'Testing':
        skill_python[i] = np.random.choice([0,1], p=[0.60, 0.40])
        skill_java[i] = np.random.choice([0,1], p=[0.55, 0.45])
        skill_sql[i] = np.random.choice([0,1], p=[0.35, 0.65])
        skill_javascript[i] = np.random.choice([0,1], p=[0.70, 0.30])
        skill_csharp[i] = np.random.choice([0,1], p=[0.80, 0.20])
    elif departments[i] == 'DevOps':
        skill_python[i] = np.random.choice([0,1], p=[0.30, 0.70])
        skill_java[i] = np.random.choice([0,1], p=[0.65, 0.35])
        skill_sql[i] = np.random.choice([0,1], p=[0.50, 0.50])
        skill_javascript[i] = np.random.choice([0,1], p=[0.60, 0.40])
        skill_csharp[i] = np.random.choice([0,1], p=[0.85, 0.15])
    elif departments[i] == 'UI/UX':
        skill_python[i] = np.random.choice([0,1], p=[0.80, 0.20])
        skill_java[i] = np.random.choice([0,1], p=[0.85, 0.15])
        skill_javascript[i] = np.random.choice([0,1], p=[0.25, 0.75])
        skill_sql[i] = np.random.choice([0,1], p=[0.70, 0.30])
        skill_csharp[i] = np.random.choice([0,1], p=[0.90, 0.10])
    else:  # PM, Admin
        skill_python[i] = np.random.choice([0,1], p=[0.85, 0.15])
        skill_java[i] = np.random.choice([0,1], p=[0.90, 0.10])
        skill_sql[i] = np.random.choice([0,1], p=[0.60, 0.40])
        skill_javascript[i] = np.random.choice([0,1], p=[0.85, 0.15])
        skill_csharp[i] = np.random.choice([0,1], p=[0.92, 0.08])

# English - fairly common (Indian company, ~70% know English)
for i in range(n):
    if departments[i] in ['Development', 'Project Management', 'DevOps']:
        lang_english[i] = np.random.choice([0,1], p=[0.20, 0.80])
    else:
        lang_english[i] = np.random.choice([0,1], p=[0.40, 0.60])

# French - RARE (~8-10% know it)
for i in range(n):
    lang_french[i] = np.random.choice([0,1], p=[0.92, 0.08])

# Hindi - very common
for i in range(n):
    lang_hindi[i] = np.random.choice([0,1], p=[0.10, 0.90])

# NOW FORCE exactly 3 employees to have ALL 3 required skills
# First, clear any accidental triple-skill holders
for i in range(n):
    if skill_python[i] == 1 and lang_english[i] == 1 and lang_french[i] == 1:
        # Remove one skill randomly
        choice = np.random.choice(['python', 'french'])
        if choice == 'python':
            skill_python[i] = 0
        else:
            lang_french[i] = 0

# Pick 3 senior developers to have all 3 skills
triple_candidates = [i for i in range(n) if 'Senior' in roles[i] and departments[i] == 'Development']
if len(triple_candidates) < 3:
    triple_candidates += [i for i in range(n) if departments[i] == 'Development']
triple_skilled = np.random.choice(triple_candidates, size=3, replace=False)

for idx in triple_skilled:
    skill_python[idx] = 1
    lang_english[idx] = 1
    lang_french[idx] = 1

# Verify
triple_count = sum(1 for i in range(n) if skill_python[i]==1 and lang_english[i]==1 and lang_french[i]==1)
assert triple_count == 3, f"Expected 3 triple-skilled, got {triple_count}"

# --- PERFORMANCE & BEHAVIORAL METRICS ---

# Manager Feedback (1-10)
manager_feedback = np.clip(np.random.normal(6.5, 1.8, n), 1, 10).round(1)
# Seniors get slightly better feedback
for i in range(n):
    if 'Senior' in roles[i]:
        manager_feedback[i] = np.clip(manager_feedback[i] + 1.5, 1, 10)

# Performance Rating (1-5)
perf_rating = np.clip(np.random.normal(3.2, 0.9, n), 1, 5).round(1)
for i in range(n):
    if 'Senior' in roles[i]:
        perf_rating[i] = np.clip(perf_rating[i] + 0.5, 1, 5)

# Learning Attitude (1-10)
learning_attitude = np.clip(np.random.normal(5.8, 2.0, n), 1, 10).round(1)
# Younger employees slightly more eager to learn
for i in range(n):
    if ages[i] < 30:
        learning_attitude[i] = np.clip(learning_attitude[i] + 1.0, 1, 10)

# Skills Upgraded in Last Year (0-5)
skills_upgraded = np.zeros(n, dtype=int)
for i in range(n):
    if learning_attitude[i] > 7:
        skills_upgraded[i] = np.random.choice([2,3,4,5], p=[0.3, 0.35, 0.25, 0.1])
    elif learning_attitude[i] > 5:
        skills_upgraded[i] = np.random.choice([0,1,2,3], p=[0.15, 0.35, 0.35, 0.15])
    else:
        skills_upgraded[i] = np.random.choice([0,1,2], p=[0.50, 0.35, 0.15])

# Training Hours Last Year (0-120)
training_hours = np.zeros(n, dtype=int)
for i in range(n):
    base = learning_attitude[i] * 8 + np.random.normal(0, 12)
    training_hours[i] = int(np.clip(base, 0, 120))

# Projects Completed (1-15)
projects_completed = np.zeros(n, dtype=int)
for i in range(n):
    if departments[i] in ['Development', 'DevOps']:
        projects_completed[i] = np.random.randint(3, 15)
    elif departments[i] in ['Testing', 'UI/UX']:
        projects_completed[i] = np.random.randint(2, 12)
    else:
        projects_completed[i] = np.random.randint(1, 8)

# Certifications (0-5)
certifications = np.zeros(n, dtype=int)
for i in range(n):
    if learning_attitude[i] > 7 and perf_rating[i] > 3.5:
        certifications[i] = np.random.choice([1,2,3,4], p=[0.25, 0.35, 0.25, 0.15])
    elif learning_attitude[i] > 5:
        certifications[i] = np.random.choice([0,1,2], p=[0.30, 0.45, 0.25])
    else:
        certifications[i] = np.random.choice([0,1], p=[0.65, 0.35])

# Last Promotion Months Ago (0-12, since company is ~1 year old at COVID time)
last_promo_months = np.random.randint(0, 13, size=n)
for i in range(n):
    if perf_rating[i] > 4:
        last_promo_months[i] = np.random.randint(0, 5)  # recent promotion

# OverTime (Yes/No)
overtime = []
for i in range(n):
    if departments[i] == 'Development':
        overtime.append(np.random.choice(['Yes','No'], p=[0.45, 0.55]))
    elif departments[i] in ['DevOps', 'Testing']:
        overtime.append(np.random.choice(['Yes','No'], p=[0.35, 0.65]))
    else:
        overtime.append(np.random.choice(['Yes','No'], p=[0.15, 0.85]))

# Work-Life Balance (1-4)
wlb = np.zeros(n, dtype=int)
for i in range(n):
    if overtime[i] == 'Yes':
        wlb[i] = np.random.choice([1,2,3,4], p=[0.25, 0.40, 0.25, 0.10])
    else:
        wlb[i] = np.random.choice([1,2,3,4], p=[0.05, 0.20, 0.45, 0.30])

# Job Satisfaction (1-4)
job_sat = np.random.choice([1,2,3,4], size=n, p=[0.12, 0.25, 0.38, 0.25])

# Environment Satisfaction (1-4)
env_sat = np.random.choice([1,2,3,4], size=n, p=[0.10, 0.22, 0.40, 0.28])

# Absenteeism Days (last quarter, 0-15)
absenteeism = np.zeros(n, dtype=int)
for i in range(n):
    if job_sat[i] <= 2 and wlb[i] <= 2:
        absenteeism[i] = np.random.randint(5, 15)
    elif job_sat[i] <= 2 or wlb[i] <= 2:
        absenteeism[i] = np.random.randint(2, 10)
    else:
        absenteeism[i] = np.random.randint(0, 5)

# Team Collaboration Score (1-10)
team_collab = np.clip(np.random.normal(6.5, 1.5, n), 1, 10).round(1)

# ============================================================
# ATTRITION RISK (Will leave in 3 months? Yes/No)
# Correlate with: low satisfaction, overtime, low salary, low manager feedback,
# low learning attitude, high absenteeism
# Target: ~20-25% attrition risk
# ============================================================
attrition_score = np.zeros(n, dtype=float)
for i in range(n):
    score = 0
    # Overtime adds risk
    if overtime[i] == 'Yes': score += 2.0
    # Low job satisfaction
    score += (4 - job_sat[i]) * 1.2
    # Low WLB
    score += (4 - wlb[i]) * 1.0
    # Low environment satisfaction
    score += (4 - env_sat[i]) * 0.8
    # Low manager feedback
    score += max(0, 5 - manager_feedback[i]) * 0.6
    # Low salary (normalized)
    sal_norm = (salaries[i] - min(salaries)) / (max(salaries) - min(salaries))
    score += (1 - sal_norm) * 2.5
    # High absenteeism
    score += absenteeism[i] * 0.3
    # Low performance
    score += max(0, 3 - perf_rating[i]) * 1.0
    # Random noise
    score += np.random.normal(0, 1.5)
    attrition_score[i] = score

# Convert to binary: top ~22% are at risk
threshold = np.percentile(attrition_score, 78)
attrition_risk = ['Yes' if s >= threshold else 'No' for s in attrition_score]

# Ensure the 3 triple-skilled employees are NOT at risk
for idx in triple_skilled:
    attrition_risk[idx] = 'No'

# --- BUILD DATAFRAME ---
df = pd.DataFrame({
    'EmployeeID': ids,
    'Name': first_names,
    'Age': ages,
    'Gender': genders,
    'Department': departments,
    'Role': roles,
    'JoiningDate': joining_dates,
    'MonthlySalary_INR': salaries,
    'Skill_Python': skill_python,
    'Skill_Java': skill_java,
    'Skill_SQL': skill_sql,
    'Skill_JavaScript': skill_javascript,
    'Skill_CSharp': skill_csharp,
    'Lang_English': lang_english,
    'Lang_French': lang_french,
    'Lang_Hindi': lang_hindi,
    'ManagerFeedback': manager_feedback,
    'PerformanceRating': perf_rating,
    'LearningAttitude': learning_attitude,
    'SkillsUpgradedLastYear': skills_upgraded,
    'TrainingHoursLastYear': training_hours,
    'ProjectsCompleted': projects_completed,
    'Certifications': certifications,
    'LastPromotionMonthsAgo': last_promo_months,
    'OverTime': overtime,
    'WorkLifeBalance': wlb,
    'JobSatisfaction': job_sat,
    'EnvironmentSatisfaction': env_sat,
    'AbsenteeismDays': absenteeism,
    'TeamCollaborationScore': team_collab,
    'Attrition_Risk_3Months': attrition_risk
})

# --- COMPUTED COLUMNS FOR ANALYSIS ---
# Count of required skills (Python + English + French) each employee has
df['RequiredSkillsCount'] = df['Skill_Python'] + df['Lang_English'] + df['Lang_French']

# Total technical skills
df['TotalTechSkills'] = df['Skill_Python'] + df['Skill_Java'] + df['Skill_SQL'] + df['Skill_JavaScript'] + df['Skill_CSharp']

# Total languages
df['TotalLanguages'] = df['Lang_English'] + df['Lang_French'] + df['Lang_Hindi']

print(f"Dataset shape: {df.shape}")
print(f"Employees with ALL 3 required skills: {len(df[df['RequiredSkillsCount']==3])}")
print(f"Employees with at least 1 required skill: {len(df[df['RequiredSkillsCount']>=1])}")
print(f"Employees with 0 required skills: {len(df[df['RequiredSkillsCount']==0])}")
print(f"Attrition risk Yes: {len(df[df['Attrition_Risk_3Months']=='Yes'])}")
print(f"Attrition risk No: {len(df[df['Attrition_Risk_3Months']=='No'])}")
print(f"\nDepartment distribution:")
print(df['Department'].value_counts())
print(f"\nRole distribution:")
print(df['Role'].value_counts())

# Save
df.to_csv('/home/claude/nikhil-dashboard/nikhil_company_data.csv', index=False)
print("\nSaved to nikhil_company_data.csv")
