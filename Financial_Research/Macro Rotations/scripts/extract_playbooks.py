import os
from pypdf import PdfReader

directory = "/Users/andrespena/Desktop/Capital Flows"

def extract_text_from_pdf(filepath):
    try:
        reader = PdfReader(filepath)
        text = ""
        # Read first 5 pages to get the core thesis/summary
        for i in range(min(len(reader.pages), 5)):
            text += reader.pages[i].extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading {filepath}: {e}"

print("--- EXTRACTING CAPITAL FLOWS PLAYBOOKS ---\n")

for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        filepath = os.path.join(directory, filename)
        print(f"PROCESSING: {filename}")
        print("-" * 50)
        content = extract_text_from_pdf(filepath)
        # Print a snippet to avoid overwhelming output, but enough to capture key points
        print(content[:2000]) 
        print("\n" + "="*80 + "\n")
