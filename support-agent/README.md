# Customer Support Agent

An AI agent connected to a live Supabase database. Handles customer support 
conversations, looks up accounts, manages tickets, and writes changes back 
to the database in real time.

## Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| /chat | POST | Talk to the support agent |
| /upload | POST | Inject your own CSV data |
| /reset | DELETE | Wipe all demo data |
| /logs | GET | See agent audit trail |
| /stats | GET | Live ticket dashboard |
| /status | GET | System health check |

## Try it

Ask the agent about any of these customers:
- tom@example.com — has a refund issue
- sarah@example.com — has an unactivated add-on
- priya@example.com — enterprise customer

Or upload your own data using the CSV format below.

## CSV Upload Format

Upload any CSV with these columns:

name, email, issue, priority

Priority values: low, normal, high

See sample_upload.csv for an example.

## Stack

- LLM: Claude Sonnet 4.6
- Database: Supabase (PostgreSQL)
- API: FastAPI
- Tool calling: 5 database tools
- Memory: session-scoped conversation history