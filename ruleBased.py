import re

# Input sentence
text = "my name is anitta kurian and my email is anitta6238@gmail.com and my group members emails are Anatt- anatt@gmail.com,Sony- sony24pmc154@mariancollege.org,Athul- athuroy656@gmail.com,Beneesh - beneesh24pmc119@gmail.com"

# Rule-based pattern to match emails
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# Extract all matches using re.findall()
emails = re.findall(email_pattern, text)

# Output the result
print("✅ Extracted Emails:")
for email in emails:
    print(email)
