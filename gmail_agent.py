# gmail_agent.py
import os
import time
import base64
import json
import logging
from typing import Dict, Any, List, Optional
from shared_models import OllamaLLMIntegration

# --- Google API Imports ---
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- CVA Project Imports ---
from alert_store import get_alert_store
from email_schemas import SCHEMAS

# --- Configuration ---
# We now need permission to READ and MODIFY (to mark as read)
SCOPES = ["https://www.googleapis.com/auth/gmail.modify","https://www.googleapis.com/auth/calendar.readonly"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

# Initialize a logger for this agent
logger = logging.getLogger("GmailAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Google Authentication  ---

def authenticate_google():
    """Handles the OAuth2 flow and returns an authenticated service object."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Credentials expired. Refreshing...")
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}. Re-authentication needed.")
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE) # Remove bad token
                creds = None # Force re-auth
        
        if not creds:
            logger.info("No valid credentials. Starting authentication flow...")
            # This will open a browser window for you to log in
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
            logger.info(f"Authentication successful. Token saved to {TOKEN_FILE}")
    
    try:
        service = build("gmail", "v1", credentials=creds)
        logger.info("Gmail API service built successfully.")
        return service
    except HttpError as error:
        logger.error(f"An error occurred building the service: {error}")
        return None

# --- Step 1: The "Router" (Triage) ---

def quick_analyze(llm: OllamaLLMIntegration, email_snippet: str, email_from: str, email_subject: str) -> Dict[str, Any]:
    """
    Step 1: Triage (Router)
    Uses the agent's built-in self.llm to categorize an email.
    """
    valid_categories = list(SCHEMAS.keys())
    
    prompt = f"""
    You are an email triage assistant. Your job is to categorize an email and
    decide if it's important.
    
    Here are the *only* categories you can choose from:
    {valid_categories}

    If the email is not one of those, you MUST categorize it as "other".

    Analyze the email below:
    From: {email_from}
    Subject: {email_subject}
    Snippet: {email_snippet}

    Respond with ONLY a valid JSON object in the following format:
    {{
      "category": "...",
      "urgency": "critical" | "medium" | "low",
      "requires_action": true | false
    }}
    """
    response_text = ""
    try:
        response_text = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.0)
        if not response_text:
             raise ValueError("LLM returned an empty response.")
        
        # Clean up the response (LLMs sometimes add "```json" wrappers)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()

        data = json.loads(response_text)
        
        # --- This is our key "routing" logic ---
        category = data.get("category")
        if data.get("requires_action") and category in SCHEMAS:
            data["extract_with_schema"] = category
        else:
            data["extract_with_schema"] = None

        logger.info(f"Triage complete: {data}")
        return data

    except Exception as e:
        logger.error(f"ERROR in quick_analyze: {e}\nRaw LLM response: {response_text}")
        return {"category": "unknown", "urgency": "low", "requires_action": False, "extract_with_schema": None}

# --- Step 2: The "Extractor" ---

def deep_extract(llm: OllamaLLMIntegration, email_body: str, category: str) -> Optional[Dict[str, Any]]:
    """
    Step 2: Extract (Extractor)
    Uses the specific schema and prompt for a given category to extract structured data.
    """
    if not category or category not in SCHEMAS:
        logger.error(f"ERROR: Invalid category for deep_extract: {category}")
        return None
        
    schema = SCHEMAS[category]
    prompt = f"""
    {schema['prompt']}

    Here is the email body:
    ---
    {email_body}
    ---
    Respond with ONLY the valid JSON object.
    """
    response_text = ""
    try:
        response_text = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.0)
        if not response_text:
             raise ValueError("LLM returned an empty response for extraction.")
        
        # Clean up the response (LLMs sometimes add "```json" wrappers)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
            
        data = json.loads(response_text)
        logger.info(f"Deep extraction complete for '{category}': {data}")
        return data

    except Exception as e:
        logger.error(f"ERROR in deep_extract: {e}\nRaw LLM response: {response_text}")
        return None

# --- Step 3: The "Heartbeat" (Processing Logic) ---

def process_email(service, llm, msg_summary):
    """
    The full process for a single email: Get, Triage, Extract, Alert, and Mark Read.
    """
    alert_store = get_alert_store()
    msg_id = msg_summary['id']

    try:
        # 1. Get the full email message
        msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        
        snippet = msg.get('snippet', '')
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        
        email_from = ""
        email_subject = ""
        for h in headers:
            if h['name'].lower() == 'subject':
                email_subject = h['value']
            if h['name'].lower() == 'from':
                email_from = h['value']

        # Get the email body (this is complex, we'll just get the first part)
        body_data = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    body_data = part['body'].get('data', '')
                    break
        elif 'body' in payload:
            body_data = payload['body'].get('data', '')

        # Handle empty body
        if not body_data:
            logger.warning(f"Email {msg_id} has no plain text body. Using snippet.")
            email_body = snippet
        else:
            email_body = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')

        # 2. Step 1: Triage (Router)
        logger.info(f"Analyzing new email. Subject: {email_subject}")
        triage = quick_analyze(llm, snippet, email_from, email_subject)

        if not triage.get("requires_action"):
            logger.info("Email does not require action. Marking as read.")
            service.users().messages().modify(userId="me", id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
            return # Stop processing

        schema_to_use = triage.get("extract_with_schema")
        if not schema_to_use:
            logger.warning(f"Email required action but had no schema: {triage.get('category')}. Marking as read.")
            service.users().messages().modify(userId="me", id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
            return # Stop processing

        # 3. Step 2: Extract
        logger.info(f"Email requires deep extraction with schema: '{schema_to_use}'")
        extracted_data = deep_extract(llm, email_body, schema_to_use)
        
        if not extracted_data:
            logger.error("Deep extraction failed. Email will not be processed.")
            # We don't mark as read, so we can retry
            return

        # 4. Write to AlertStore (Blackboard)
        alert_store.add_alert({
            "type": triage.get("category"),
            "urgency": triage.get("urgency"),
            "source": "GmailAgent",
            "source_email_id": msg_id,
            "subject": email_subject,
            "details": extracted_data # This is the full, structured JSON!
        })
        logger.info(f"Successfully added structured alert for '{schema_to_use}' to AlertStore.")

        # 5. Mark as Read (Consume)
        service.users().messages().modify(userId="me", id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
        logger.info(f"Marked email {msg_id} as read.")

    except HttpError as error:
        logger.error(f"An HttpError occurred processing email {msg_id}: {error}")
    except Exception as e:
        logger.error(f"A general error occurred processing email {msg_id}: {e}", exc_info=True)

# --- Main Gmail API Loop (Replaces old 'check_emails') ---

def check_emails_loop(service, llm):
    """
    The main loop that searches for unread emails and sends them to be processed.
    """
    try:
        # We only want to process a few emails at a time to not get rate-limited
        results = service.users().messages().list(
            userId="me", 
            q="is:unread",
            maxResults=5 # Process 5 at a time
        ).execute()
        
        messages = results.get("messages", [])

        if not messages:
            logger.info("No unread emails found.")
            return

        logger.info(f"Found {len(messages)} unread emails. Processing...")
        
        for msg_summary in messages:
            process_email(service, llm, msg_summary)

    except HttpError as error:
        logger.error(f"An error occurred in check_emails_loop: {error}")

# --- Main Thread Function (Called by app.py) ---

def main_loop():
    """The main loop for the Gmail Agent."""
    logger.info("[GmailAgent] Starting main loop...")
    
    # 1. Authenticate with Google
    service = authenticate_google()
    if not service:
        logger.critical("Failed to authenticate with Google. GmailAgent cannot start.")
        return # Thread will exit

    # 2. Initialize the LLM (one instance for this thread)
    try:
        llm = OllamaLLMIntegration(logger=logger)
        if not llm.chat_client:
            raise ConnectionError("Failed to connect to Ollama.")
        logger.info("OllamaLLMIntegration initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize OllamaLLMIntegration: {e}. GmailAgent cannot start.")
        return # Thread will exit

    # 3. Start the infinite loop
    while True:
        try:
            check_emails_loop(service, llm)
            time.sleep(60) # Check for new emails every 60 seconds
        except KeyboardInterrupt:
            logger.info("Shutting down.")
            break
        except Exception as e:
            logger.error(f"Main loop critical error: {e}", exc_info=True)
            # Don't crash the loop, just wait and retry
            time.sleep(300) 

if __name__ == "__main__":
    main_loop()