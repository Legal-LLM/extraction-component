from dotenv import load_dotenv
load_dotenv()

import os
import pathlib
import google.generativeai as genai
import time
import json

# --- Configuration ---
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=API_KEY)

# Using "." means the script looks for act folders in its current directory
TOP_LEVEL_FOLDER = "." 

# --- Output Folder Setup ---
FINAL_OUTPUT_FOLDER = "extracted_acts_grouped_json" # New folder for this final format
TEMP_CHUNK_JSON_FOLDER = "temp_chunk_grouped_json"

# --- Robustness Settings ---
MAX_RETRIES = 3 
RETRY_DELAY_SECONDS = 70 
PROACTIVE_DELAY_SECONDS = 5
# ---------------------

def extract_structure_from_chunk(pdf_chunk_path):
    """
    Uploads a PDF chunk and uses a JSON schema to force the model to return
    a list of objects, where each object represents a grouped, top-level section.
    """
    print(f"  > Processing chunk with GROUPED JSON Schema: {os.path.basename(pdf_chunk_path)}...")
    
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
    
    uploaded_file = None
    try:
        # --- THIS IS THE NEW, GROUPING SCHEMA ---
        # The schema asks for a list of objects, where each object is a complete top-level section.
        response_schema = {
            "type": "ARRAY",
            "description": "A list of all top-level legal sections found in this document chunk.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "act_title": {
                        "type": "STRING",
                        "description": "The main title of the act, e.g., 'Consumer Affairs Authority Act'."
                    },
                    "act_id": {
                        "type": "STRING",
                        "description": "The identifying number and year of the act, e.g., 'No. 9 of 2003'."
                    },
                    "clause_number": {
                        "type": "STRING",
                        "description": "The top-level section number ONLY (e.g., '2', '3', 'Preamble'). Do not include subsection numbers like '(1)' or '(a)'."
                    },
                    "full_citation": {
                        "type": "STRING",
                        "description": "The complete citation for the top-level section."
                    },
                    "content": {
                        "type": "STRING",
                        "description": "The COMBINED verbatim text of the main section AND ALL of its nested subsections (e.g., for section '2', this should include the text of 2(1), 2(2), etc.)."
                    }
                },
                "required": ["act_title", "act_id", "clause_number", "full_citation", "content"]
            }
        }
        
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema
        )

        print(f"    - Uploading...")
        uploaded_file = genai.upload_file(path=pdf_chunk_path, mime_type="application/pdf")

        print(f"    - Extracting structure...")
        # --- THE NEW, GROUPING PROMPT ---
        prompt = "Analyze the provided legal document fragment. For EACH top-level section number (e.g., Section 2, Section 3), create a single JSON object. The 'content' for that object must contain the combined text of that section and ALL of its subsections ((1), (2), (a), (b), etc.). Return a JSON array of these grouped objects, conforming strictly to the provided schema."
        
        response = model.generate_content(
            [prompt, uploaded_file],
            generation_config=generation_config,
            request_options={"timeout": 300}
        )
        
        return json.loads(response.text)

    finally:
        if uploaded_file:
            print(f"    - Deleting uploaded file...")
            genai.delete_file(uploaded_file.name)

def main():
    final_output_path = os.path.join(TOP_LEVEL_FOLDER, FINAL_OUTPUT_FOLDER)
    temp_output_path_base = os.path.join(TOP_LEVEL_FOLDER, TEMP_CHUNK_JSON_FOLDER)
    os.makedirs(final_output_path, exist_ok=True)
    os.makedirs(temp_output_path_base, exist_ok=True)

    print(f"Starting GROUPED JSON extraction from: '{os.path.abspath(TOP_LEVEL_FOLDER)}'")
    
    main_act_folders = [f for f in os.listdir(TOP_LEVEL_FOLDER) if os.path.isdir(os.path.join(TOP_LEVEL_FOLDER, f)) and f not in [FINAL_OUTPUT_FOLDER, TEMP_CHUNK_JSON_FOLDER, '.git', 'venv']]
    
    for act_name in main_act_folders:
        print(f"\n{'='*60}\nProcessing Act: {act_name}\n{'='*60}")
        
        final_act_json_path = os.path.join(final_output_path, f"{act_name}.json")
        if os.path.exists(final_act_json_path):
            print(f"  > Final file already exists. Skipping entire act.")
            continue

        nested_act_path = os.path.join(TOP_LEVEL_FOLDER, act_name, act_name)
        temp_act_chunk_path = os.path.join(temp_output_path_base, act_name)
        os.makedirs(temp_act_chunk_path, exist_ok=True)
        
        chunks_to_process = sorted([os.path.join(nested_act_path, sub, f) for sub in ["Initial Chunk", "Overlap Chunk"] if os.path.isdir(os.path.join(nested_act_path, sub)) for f in os.listdir(os.path.join(nested_act_path, sub)) if f.lower().endswith('.pdf')])
        
        if not chunks_to_process:
            print(f"  No PDF chunks found for {act_name}. Skipping.")
            continue

        print(f"Found {len(chunks_to_process)} chunks to process for '{act_name}'.")

        for chunk_path in chunks_to_process:
            chunk_basename = os.path.basename(chunk_path)
            chunk_json_filename = f"{pathlib.Path(chunk_basename).stem}.json"
            temp_json_path = os.path.join(temp_act_chunk_path, chunk_json_filename)

            if os.path.exists(temp_json_path):
                print(f"  > Skipping '{chunk_basename}'. Already processed.")
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    list_of_clauses = extract_structure_from_chunk(chunk_path)
                    with open(temp_json_path, "w", encoding="utf-8") as f:
                        json.dump(list_of_clauses, f, indent=2)
                    print(f"  > SUCCESS on '{chunk_basename}'. Temp JSON saved.\n")
                    time.sleep(PROACTIVE_DELAY_SECONDS)
                    break 
                except Exception as e:
                    print(f"  >! ERROR on attempt {attempt + 1}/{MAX_RETRIES} for '{chunk_basename}': {e}")
                    if "429" in str(e) or "ResourceExhausted" in str(e):
                        print(f"  >! Rate limit hit. Waiting for {RETRY_DELAY_SECONDS} seconds...")
                        time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        time.sleep(10)
            else: 
                print(f"  >!!! FAILED to process chunk '{chunk_basename}' after {MAX_RETRIES} attempts.")

        # --- Final Combination Logic ---
        print(f"\nCombining all temporary JSON lists for {act_name}...")
        
        master_clause_list = []
        
        temp_json_files = sorted([os.path.join(temp_act_chunk_path, f) for f in os.listdir(temp_act_chunk_path) if f.endswith('.json')])

        for temp_json_path in temp_json_files:
            try:
                with open(temp_json_path, "r", encoding="utf-8") as f:
                    clauses_from_chunk = json.load(f)
                    master_clause_list.extend(clauses_from_chunk)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"  >! Warning: Could not read or decode {os.path.basename(temp_json_path)}. Error: {e}")
        
        if master_clause_list:
            with open(final_act_json_path, "w", encoding="utf-8") as f:
                json.dump(master_clause_list, f, ensure_ascii=False, indent=2)
            print(f"SUCCESS: Master list of clauses saved to '{final_act_json_path}'")

    print(f"\n{'='*60}\nAll legal acts have been processed.")

if __name__ == "__main__":
    main()