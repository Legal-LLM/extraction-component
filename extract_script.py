from dotenv import load_dotenv

load_dotenv()

import os
import pathlib
from google import genai
import time
from google.genai import types

# --- Configuration ---

TOP_LEVEL_FOLDER = "extraction-component"

# --- New Checkpoint & Output Folders ---
# This folder will store the text of each successfully processed chunk
TEMP_OUTPUT_FOLDER = "temp_extracted_chunks"
# This folder will store the final, combined text files
FINAL_OUTPUT_FOLDER = "extracted_acts_final"

# --- New Robustness Settings ---
MAX_RETRIES = 3 
RETRY_DELAY_SECONDS = 70 # Increased delay slightly
PROACTIVE_DELAY_SECONDS = 65 # Drastically increased the proactive delay
# ---------------------

def extract_text_from_pdf(file):
    client = genai.Client()
    prompt = "You are a high-precision data extraction tool. Extract the full text from the PDF. IGNORE and OMIT all headers, footers, page numbers, and marginal notes. Output only the clean, verbatim body text."
    filepath = pathlib.Path(file)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    ),
        contents=[
            types.Part.from_bytes(
                data=filepath.read_bytes(),
                mime_type='application/pdf',
            ),
            prompt])
    return response.text

def main():
    final_output_path = os.path.join(TOP_LEVEL_FOLDER, FINAL_OUTPUT_FOLDER)
    temp_output_path_base = os.path.join(TOP_LEVEL_FOLDER, TEMP_OUTPUT_FOLDER)
    os.makedirs(final_output_path, exist_ok=True)
    os.makedirs(temp_output_path_base, exist_ok=True)

    print(f"Starting extraction inside folder: '{TOP_LEVEL_FOLDER}'")
    
    main_act_folders = [f for f in os.listdir(TOP_LEVEL_FOLDER) if os.path.isdir(os.path.join(TOP_LEVEL_FOLDER, f)) and f not in [FINAL_OUTPUT_FOLDER, TEMP_OUTPUT_FOLDER, '.git', 'venv']]
    
    for act_name in main_act_folders:
        print(f"\n{'='*50}\nProcessing Act: {act_name}\n{'='*50}")

        nested_act_path = os.path.join(TOP_LEVEL_FOLDER, act_name, act_name)
        temp_act_chunk_path = os.path.join(temp_output_path_base, act_name)
        os.makedirs(temp_act_chunk_path, exist_ok=True)
        
        chunks_to_process = []
        for subfolder in ["Initial Chunk", "Overlap Chunk"]:
            subfolder_path = os.path.join(nested_act_path, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in sorted(os.listdir(subfolder_path)):
                    if filename.lower().endswith(".pdf"):
                        chunks_to_process.append(os.path.join(subfolder_path, filename))
        
        if not chunks_to_process:
            print(f"  No PDF chunks found for {act_name}. Skipping.")
            continue
            
        print(f"Found {len(chunks_to_process)} PDF chunks to process for '{act_name}'.")

        for chunk_path in chunks_to_process:
            chunk_basename = os.path.basename(chunk_path)
            chunk_txt_filename = f"{pathlib.Path(chunk_basename).stem}.txt"
            temp_txt_path = os.path.join(temp_act_chunk_path, chunk_txt_filename)

            # --- THE CHECKPOINT LOGIC ---
            if os.path.exists(temp_txt_path):
                print(f"  > Skipping '{chunk_basename}'. Already processed.")
                continue
            # ---------------------------

            for attempt in range(MAX_RETRIES):
                try:
                    clean_text = extract_text_from_pdf(chunk_path)
                    
                    # Save the successful chunk to the temp folder
                    with open(temp_txt_path, "w", encoding="utf-8") as f:
                        f.write(clean_text)
                    print(f"  > SUCCESS on '{chunk_basename}'. Saved to temp folder.\n")
                    
                    print(f"  ...waiting for {PROACTIVE_DELAY_SECONDS} seconds to avoid rate limit...")
                    time.sleep(PROACTIVE_DELAY_SECONDS)
                    break 

                except Exception as e:
                    print(f"  >! UNEXPECTED ERROR on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                    time.sleep(10)
            else: 
                print(f"  >!!! FAILED to process chunk '{chunk_basename}' after {MAX_RETRIES} attempts.")

        # --- Final Combination ---
        print(f"\nCombining all successfully extracted chunks for {act_name}...")
        
        final_text_parts = []
        for chunk_path in chunks_to_process: # Iterate in original sorted order
            chunk_txt_filename = f"{pathlib.Path(os.path.basename(chunk_path)).stem}.txt"
            temp_txt_path = os.path.join(temp_act_chunk_path, chunk_txt_filename)
            if os.path.exists(temp_txt_path):
                with open(temp_txt_path, "r", encoding="utf-8") as f:
                    final_text_parts.append(f.read())
        
        if final_text_parts:
            full_act_text = "\n\n".join(final_text_parts)
            output_filename = f"{act_name}.txt"
            final_act_path = os.path.join(final_output_path, output_filename)
            with open(final_act_path, "w", encoding="utf-8") as f:
                f.write(full_act_text)
            print(f"SUCCESS: Consolidated act saved to '{final_act_path}'")

    print(f"\n{'='*50}\nAll legal acts have been processed.")

if __name__ == "__main__":
    main()