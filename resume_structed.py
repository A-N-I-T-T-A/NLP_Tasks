import spacy
import re
import json
import pdfplumber

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Utility: Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Extract entities and structured info from text
def extract_resume_data(text):
    doc = nlp(text)

    structured_data = {
        "Name": None,
        "Email": None,
        "Phone": None,
        "Organizations": [],
        "Skills": [],
        "Education": [],
        "Experience": []
    }

    # Basic regex for emails and phone numbers
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phone_match = re.search(r"(\+91[\-\s]?|0)?[6-9]\d{9}", text)

    if email_match:
        structured_data["Email"] = email_match
    if phone_match:
        structured_data["Phone"] = phone_match

    # Named Entity Recognition for Name, Org, Dates
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not structured_data["Name"]:
            structured_data["Name"] = ent.text.strip()
        elif ent.label_ == "ORG":
            structured_data["Organizations"].append(ent.text.strip())
        elif ent.label_ in ["DATE", "TIME"]:
            structured_data["Experience"].append(ent.text.strip())


    # Education matching using degree keywords
    degree_keywords = [
        "B.Sc", "M.Sc", "B.Tech", "M.Tech",
        "Bachelor of Computer Applications", "Master of Computer Applications",
        "Bachelor of Science", "Master of Science", 
        "Bachelor of Technology", "Master of Technology", "PhD", "Diploma"
    ]
    for degree in degree_keywords:
        matches = re.findall(rf"\b{degree}\b", text, re.IGNORECASE)
        for match in matches:
            if match not in structured_data["Education"]:
                structured_data["Education"].append(match)

    # Skill matching
    skills_keywords = [
        "Python", "Django", "Machine Learning", "Render",
        "JavaScript", "Data Analysis", "SQL", "Excel", "Power BI", "NLP"
    ]
    for skill in skills_keywords:
        if re.search(rf"\b{skill}\b", text, re.IGNORECASE):
            structured_data["Skills"].append(skill)

    return structured_data



# Save output as JSON
def save_to_json(data, filename="resume_output.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# --------- MAIN USAGE ---------
if __name__ == "__main__":
    resume_path = "Anitta_Kurian.pdf"  

    # Extract text
    text = extract_text_from_pdf(resume_path)

    # Extract structured info
    structured_resume = extract_resume_data(text)

    # Save result
    save_to_json(structured_resume)

    print("✅ Resume parsing complete. Output saved to resume_output.json")
