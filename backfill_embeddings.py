#!/usr/bin/env python3
"""Backfill embeddings for old tasks without them."""

from database import cva_db
from shared_models import OllamaLLMIntegration
import os

def backfill_embeddings():
    # Get tasks without embeddings
    from db_postgres import execute_query
    
    tasks = execute_query('''
        SELECT task_id, task_description 
        FROM task_history 
        WHERE task_embedding IS NULL
        ORDER BY completed_at DESC
    ''', fetch=True)
    
    if not tasks:
        print("No tasks need embeddings!")
        return
    
    print(f"Found {len(tasks)} tasks without embeddings. Starting backfill...")
    
    # Initialize LLM
    llm = OllamaLLMIntegration(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        chat_model="mistral-small",
        embedding_model="mxbai-embed-large"
    )
    
    success = 0
    for i, task in enumerate(tasks, 1):
        try:
            embedding = llm.generate_embedding(task['task_description'])
            if embedding:
                execute_query('''
                    UPDATE task_history 
                    SET task_embedding = %s 
                    WHERE task_id = %s
                ''', (embedding, task['task_id']))
                success += 1
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(tasks)} tasks processed...")
        except Exception as e:
            print(f"  Failed on task {task['task_id']}: {e}")
    
    print(f"\n✓ Backfill complete! {success}/{len(tasks)} tasks now have embeddings.")

if __name__ == "__main__":
    backfill_embeddings()
