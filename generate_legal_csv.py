import pandas as pd
import random

# Configuration
NUM_CASES = 5500
YEARS = list(range(2000, 2026))
COURTS = [
    "Supreme Court of India", "Delhi High Court", "Bombay High Court", "Madras High Court",
    "Allahabad High Court", "Calcutta High Court", "Karnataka High Court", "Kerala High Court",
    "Hyderabad High Court", "Patna High Court", "Gujarat High Court", "Rajasthan High Court",
    "Punjab & Haryana High Court", "Lucknow Family Court", "Mumbai District Court",
    "Bangalore Civil Court", "Jaipur Civil Court", "Pune District Court", "Nagpur District Court",
    "Chennai Civil Court", "Chandigarh District Court", "Trivandrum District Court", "Bhopal Civil Court",
    "Ranchi District Court"
]
CATEGORIES = [
    "Property Dispute", "Criminal Case", "Fraud / Financial Crime", "Medical Negligence",
    "Divorce / Family Matter", "Contract / Corporate Law", "Environmental Violation",
    "Cyber Crime", "Employment / Labor Case", "Education / Institutional Case",
    "Consumer Protection Case", "Intellectual Property Case", "Tax / Revenue Case",
    "Human Rights Case", "Road / Traffic Accident Case"
]
VERDICTS = [
    "Accused acquitted", "Accused convicted", "Case dismissed", "Compensation awarded",
    "Divorce granted", "Divorce denied", "Case settled", "Fine imposed", "Contract cancelled",
    "Licence revoked", "Injunction granted", "Temporary relief granted", "Petition rejected",
    "Appeal allowed", "Appeal dismissed"
]
FIRST_NAMES = ["Amit", "Ramesh", "Neha", "Priya", "Rohit", "Sunita", "Anil", "Kiran", "Vikram", "Meera", "Rajesh",
               "Seema", "Lalita", "Pankaj", "Nita", "Suresh", "Ritu", "Arjun", "Kavita", "Rakesh"]
LAST_NAMES = ["Sharma", "Verma", "Mehta", "Iyer", "Kapoor", "Patel", "Khanna", "Rao", "Banerjee", "Singh",
              "Tiwari", "Deshmukh", "Joshi", "Bhatia", "Chauhan", "Menon", "Ghosh", "Nanda", "Pillai", "Tripathi"]
PLACES = ["Delhi", "Mumbai", "Chennai", "Bangalore", "Hyderabad", "Lucknow", "Patna", "Kolkata", "Jaipur", "Pune",
          "Ahmedabad", "Surat", "Indore", "Bhopal", "Nagpur", "Vadodara", "Varanasi", "Amritsar", "Ludhiana",
          "Chandigarh", "Ranchi", "Bhubaneswar", "Dehradun", "Raipur", "Guwahati", "Trivandrum", "Kochi", "Jodhpur",
          "Udaipur", "Mysore", "Shimla", "Agra", "Kanpur", "Allahabad", "Navi Mumbai", "Thane", "Noida",
          "Ghaziabad", "Faridabad", "Gurgaon", "Cuttack", "Jabalpur", "Meerut", "Vellore", "Madurai", "Rajkot",
          "Panaji", "Siliguri", "Gwalior", "Kota", "Suratgarh"]

# Category topics
CATEGORY_TOPICS = {
    "Property Dispute": ["property sale", "land acquisition", "housing dispute", "eviction", "inheritance"],
    "Criminal Case": ["domestic conflict", "smuggling", "human trafficking", "robbery", "assault"],
    "Fraud / Financial Crime": ["corporate fraud", "loan scam", "insurance fraud", "financial embezzlement", "tax evasion"],
    "Medical Negligence": ["hospital treatment", "medical surgery", "medical malpractice"],
    "Divorce / Family Matter": ["custody dispute", "domestic conflict", "child custody"],
    "Contract / Corporate Law": ["contract breach", "corporate fraud", "business dispute"],
    "Environmental Violation": ["factory pollution", "industrial waste", "illegal mining", "environmental clearance"],
    "Cyber Crime": ["cyber phishing", "online fraud", "IT firm breach", "hacking"],
    "Employment / Labor Case": ["employee wages", "workplace harassment", "unfair dismissal"],
    "Education / Institutional Case": ["educational malpractice", "school negligence"],
    "Consumer Protection Case": ["consumer goods defect", "unfair trade practice"],
    "Intellectual Property Case": ["trademark infringement", "copyright violation", "patent dispute"],
    "Tax / Revenue Case": ["tax evasion", "incorrect tax computation"],
    "Human Rights Case": ["civil liberties", "discrimination", "constitutional concern"],
    "Road / Traffic Accident Case": ["traffic accident", "road safety violation", "vehicular injury"]
}

# Expanded summary templates: 10 per category
SUMMARY_TEMPLATES = {}
for cat in CATEGORIES:
    SUMMARY_TEMPLATES[cat] = [
        f"Case related to {{topic}} in {{place}}, filed under Section {{section}}.",
        f"Petitioner filed a complaint regarding {{topic}} in {{place}}.",
        f"Dispute concerning {{topic}} occurred in {{place}}, legal proceedings initiated.",
        f"Trial examined issues on {{topic}} in {{place}}.",
        f"Court reviewed allegations involving {{topic}} in {{place}}.",
        f"Allegations of {{topic}} led to proceedings in {{place}}.",
        f"Judicial review on matters of {{topic}} in {{place}}.",
        f"Defendant challenged claims of {{topic}} in {{place}}.",
        f"Petitioner sought remedy for {{topic}} in {{place}}.",
        f"Legal evaluation of {{topic}} undertaken in {{place}}."
    ]

# Helper functions
def random_name(): return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
def random_judge(): return f"Justice {random.choice(['A.', 'B.', 'C.', 'D.', 'E.', 'F.'])} {random.choice(LAST_NAMES)}"
def random_lawyer(): return f"{random.choice(['S.', 'R.', 'A.', 'V.', 'M.', 'N.', 'P.'])} {random.choice(LAST_NAMES)}"
def random_summary(category):
    topic = random.choice(CATEGORY_TOPICS[category])
    place = random.choice(PLACES)
    section = random.randint(100, 500)
    template = random.choice(SUMMARY_TEMPLATES[category])
    return template.format(topic=topic, place=place, section=section)

# Generate records
records = []
for i in range(1, NUM_CASES + 1):
    case_id = f"C{i:04d}"
    category = random.choice(CATEGORIES)
    plaintiff = random_name()
    defendant = random_name()
    while defendant == plaintiff:
        defendant = random_name()
    case_name = f"{plaintiff} vs {defendant}"
    # Victim logic
    if category in ["Criminal Case", "Fraud / Financial Crime", "Medical Negligence", "Road / Traffic Accident Case"]:
        victim = random_name()
    else:
        victim = plaintiff
    year = random.choice(YEARS)
    court = random.choice(COURTS)
    judge = random_judge()
    lawyer = random_lawyer()
    verdict = random.choice(VERDICTS)
    summary = random_summary(category)
    records.append([case_id, case_name, year, court, judge, lawyer, victim, defendant, verdict, summary, category])

# Save CSV
columns = ["CaseID", "CaseName", "Year", "Court", "Judge", "Lawyer", "Victim", "Defendant", "Verdict", "Summary", "Category"]
df = pd.DataFrame(records, columns=columns)
df.to_csv("legal_cases_expanded.csv", index=False)
print("CSV generated successfully: legal_cases_expanded.csv")
