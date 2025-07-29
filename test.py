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
FINAL_OUTPUT_FOLDER = "extracted_acts_structured_json"
TEMP_CHUNK_JSON_FOLDER = "temp_chunk_json"

# --- Robustness Settings ---
MAX_RETRIES = 3 
RETRY_DELAY_SECONDS = 70 # Increased delay to be safe
PROACTIVE_DELAY_SECONDS = 5
# ---------------------

def extract_structure_from_chunk(pdf_chunk_path):
    """
    Uploads a single PDF chunk and uses a powerful prompt to get its structure as JSON.
    This version uses the file upload API for maximum reliability.
    """
    print(f"  > Processing chunk: {os.path.basename(pdf_chunk_path)}...")
    uploaded_file = None
    try:
        # Step 1: Upload the file to the server.
        print(f"    - Uploading...")
        uploaded_file = genai.upload_file(path=pdf_chunk_path, mime_type="application/pdf")

        # Step 2: Make the API call using the uploaded file handle.
        print(f"    - Extracting structure...")
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
        
        prompt = """
        You are an expert legal document parser. The user has provided you with a PDF that is a SMALL FRAGMENT of a larger Sri Lankan legal act. Your task is to analyze ONLY this fragment and convert it into a structured JSON format.

        Follow these rules precisely:
        1.  The final output MUST be a single, valid JSON object and nothing else.
        2.  The root of the JSON object must have three keys: "act_name", "act_number", and "clauses".
        3.  From the text within this fragment, do your best to extract the "act_name" and "act_number". If they are not present, set them to "Unknown".
        4.  The "clauses" key must contain a JSON array of objects, representing every piece of text in THIS FRAGMENT.
        5.  For each piece of text, create an object with three keys: "citation_path", "full_citation_string", and "content".
        6.  Build the "citation_path" by identifying the hierarchy ONLY within the text you can see. Example: `["Section 8", "Subsection (a)", "Clause (iv)"]`.
        7.  The "content" must be the verbatim text for that specific citation.
        8.  Ignore all page headers, footers, page numbers, and marginal notes.
        """
        
        response = model.generate_content([prompt, uploaded_file], request_options={"timeout": 300})
        
        clean_json_string = response.text.strip().lstrip("```json").rstrip("```")
        return json.loads(clean_json_string)

    finally:
        # Step 3: ALWAYS delete the file from the server to clean up storage.
        if uploaded_file:
            print(f"    - Deleting uploaded file from server...")
            genai.delete_file(uploaded_file.name)

def main():
    final_output_path = os.path.join(TOP_LEVEL_FOLDER, FINAL_OUTPUT_FOLDER)
    temp_output_path_base = os.path.join(TOP_LEVEL_FOLDER, TEMP_CHUNK_JSON_FOLDER)
    os.makedirs(final_output_path, exist_ok=True)
    os.makedirs(temp_output_path_base, exist_ok=True)

    print(f"Starting CHUNK-BY-CHUNK structured JSON extraction from: '{os.path.abspath(TOP_LEVEL_FOLDER)}'")
    
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
                    structured_data = extract_structure_from_chunk(chunk_path)
                    with open(temp_json_path, "w", encoding="utf-8") as f:
                        json.dump(structured_data, f, indent=2)
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

        # --- Final Combination from Temp JSONs ---
        print(f"\nCombining all temporary JSON files for {act_name}...")
        all_clauses = []
        final_act_name, final_act_number = "Unknown", "Unknown"
        
        temp_json_files = sorted([os.path.join(temp_act_chunk_path, f) for f in os.listdir(temp_act_chunk_path) if f.endswith('.json')])

        for temp_json_path in temp_json_files:
            try:
                with open(temp_json_path, "r", encoding="utf-8") as f:
                    temp_data = json.load(f)
                if temp_data.get("clauses"):
                    all_clauses.extend(temp_data["clauses"])
                if final_act_name == "Unknown" and temp_data.get("act_name") != "Unknown":
                    final_act_name = temp_data.get("act_name")
                if final_act_number == "Unknown" and temp_data.get("act_number") != "Unknown":
                    final_act_number = temp_data.get("act_number")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"  >! Warning: Could not read or decode {os.path.basename(temp_json_path)}. Error: {e}")
        
        if all_clauses:
            final_json_data = {"act_name": final_act_name, "act_number": final_act_number, "clauses": all_clauses}
            with open(final_act_json_path, "w", encoding="utf-8") as f:
                json.dump(final_json_data, f, ensure_ascii=False, indent=2)
            print(f"SUCCESS: Master structured JSON saved to '{final_act_json_path}'")

    print(f"\n{'='*60}\nAll legal acts have been processed.")

if __name__ == "__main__":
    main()