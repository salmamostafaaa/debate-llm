# Government vs Opposition: LLM Debate

Two free LLMs debate each other about a country's policies. One plays the **government**, the other plays the **opposition**. Built with **Gradio** and **OpenRouter**.

## Requirements covered
1. LLM #1 = government, LLM #2 = opposition.
2. System prompts force both to use valid data and statistics (with source and year).
3. The country (and optional topic) is given in the user prompt.
4. Free LLMs: OpenRouter `:free` models (two different models, with an automatic fallback).
5. UI: Gradio.
6. Debate is capped at 10 rounds (slider max + hard cap in code).

## Setup
```bash
git clone <your-repo-url>
cd debate-llm
pip install -r requirements.txt
copy .env.example .env      # Windows (use `cp` on Mac/Linux), then put your key inside .env
python app.py
```
Get a free key at https://openrouter.ai/keys

Open the local URL shown in the terminal (http://127.0.0.1:7860), type a country, choose the number of rounds, and press **Start Debate**.

## Notes
- API keys live only in `.env`, which is ignored by git.
- One round = one government statement + one opposition reply.
- Free models on OpenRouter rotate; change `GOV_MODEL` / `OPP_MODEL` in `.env` if one disappears.
