# email_schemas.py

"""
This file defines the structured JSON schemas that the GmailAgent uses
for its "deep extraction" (Step 2) analysis.
"""

# This is the "dictionary" that maps a category (from Step 1)
# to a specific, structured prompt for Step 2.
SCHEMAS = {
    
    "flight_update": {
        "description": "An email containing a flight delay, cancellation, or gate change.",
        "prompt": """
        The user's email is a flight update. Extract the following information precisely.
        
        CRITICAL TIMEZONE HANDLING:
        - The user is in Toronto, Canada (America/Toronto timezone)
        - Times in emails are in LOCAL Toronto time unless explicitly stated otherwise
        - You MUST convert local times to UTC before outputting
        - Toronto is UTC-5 (EST in winter) or UTC-4 (EDT in summer)
        - For November 2025, use EST (UTC-5)
        - Example: "11:30 PM" Toronto time = "04:30:00" UTC (next day)
        - ALWAYS add 5 hours to convert EST to UTC
        - If time is PM and after 7 PM EST, the UTC date will be the NEXT day
        
        Extract:
        - flight_number: The full flight number (e.g., "AA123", "AC999")
        - airline: The name of the airline
        - status: The new status ("delayed", "cancelled", "on_time", "gate_change")
        - old_time_utc: The ORIGINAL time converted to UTC (ISO 8601 with Z)
        - new_time_utc: The NEW time converted to UTC (ISO 8601 with Z)
        
        Respond with ONLY the JSON object.
        """,
        "example_output": {
            "flight_number": "AA123",
            "airline": "American Airlines",
            "status": "delayed",
            "old_time_utc": "2025-11-16T23:00:00Z",
            "new_time_utc": "2025-11-17T04:30:00Z"
        }
    },
    
    "free_trial": {
        "description": "An email about a free trial starting, ending, or auto-renewing.",
        "prompt": """
        The user's email is about a free trial. Extract the following information precisely.
        - service_name: The name of the product or service.
        - trial_end_date: The date the trial ends (in YYYY-MM-DD format).
        - renewal_cost: The cost after the trial (e.g., 29.99).
        - renewal_currency: The currency (e.g., "USD").
        
        Respond with ONLY the JSON object. If a value is not found, use null.
        """,
        "example_output": {
            "service_name": "SuperSaaS",
            "trial_end_date": "2025-11-23",
            "renewal_cost": 29.99,
            "renewal_currency": "USD"
        }
    },
    
    "invoice": {
        "description": "An email that is a bill, invoice, or receipt.",
        "prompt": """
        The user's email is an invoice or bill. Extract the following information precisely.
        -biller_name: The name of the company that sent the bill.
        - amount_due: The total amount due (e.g., 85.30).
        - currency: The currency (e.g., "USD").
        - due_date: The payment due date (in YYYY-MM-DD format).
        
        Respond with ONLY the JSON object. If a value is not found, use null.
        """,
        "example_output": {
            "biller_name": "ConEdison",
            "amount_due": 85.30,
            "currency": "USD",
            "due_date": "2025-12-01"
        }
    }
    
    # We can add more schemas here later (e.g., "meeting_request", "purchase_receipt")
    # without having to change the agent's code!
}