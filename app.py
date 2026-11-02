"""
Government vs Opposition: a debate between two free LLMs (via OpenRouter) with a Gradio UI.

- LLM #1 plays the government, LLM #2 plays the opposition.
- Both are instructed (system prompt) to use valid data and statistics.
- The country is given in the USER prompt.
- Maximum 10 rounds (1 round = government statement + opposition reply).
"""

import os
import time

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from a local .env file (never committed)

GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

if GROQ_KEY:
    # Groq (free tier, OpenAI-compatible): two different free models.
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
    GOV_MODEL = os.getenv("GROQ_GOV_MODEL", "openai/gpt-oss-20b")
    OPP_MODEL = os.getenv("GROQ_OPP_MODEL", "llama-3.1-8b-instant")
    FALLBACK_MODEL = "llama-3.1-8b-instant"
elif OPENROUTER_KEY:
    # OpenRouter (free ":free" models; limits are small and models rotate).
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
    GOV_MODEL = os.getenv("GOV_MODEL", "google/gemma-4-31b-it:free")
    OPP_MODEL = os.getenv("OPP_MODEL", "qwen/qwen3-next-80b-a3b-instruct:free")
    FALLBACK_MODEL = "openrouter/free"
else:
    raise RuntimeError(
        "No API key found. Put GROQ_API_KEY (or OPENROUTER_API_KEY) in your .env file."
    )

MAX_ROUNDS = 10
print(f'Provider: {"Groq" if GROQ_KEY else "OpenRouter"} | GOV={GOV_MODEL} | OPP={OPP_MODEL}')

GOV_SYSTEM = (
    "You are the official spokesperson of the GOVERNMENT in a formal political debate. "
    "Defend the government's policies and record. "
    "You MUST support every argument with valid, verifiable data and statistics "
    "(cite the source or organization and the year, e.g. World Bank, IMF, UN, national statistics office). "
    "Never invent numbers; if you are unsure of an exact figure, say it is approximate. "
    "Directly rebut the opposition's last points. "
    "Keep each answer under 120 words, professional and respectful, and always finish your last sentence completely. Output ONLY your final speech as one or two short paragraphs: no planning, no drafts, no word counts, no notes about the instructions."
)

OPP_SYSTEM = (
    "You are the leader of the OPPOSITION in a formal political debate. "
    "Challenge the government's policies and record and propose better alternatives. "
    "You MUST support every argument with valid, verifiable data and statistics "
    "(cite the source or organization and the year, e.g. World Bank, IMF, UN, national statistics office). "
    "Never invent numbers; if you are unsure of an exact figure, say it is approximate. "
    "Directly rebut the government's last points. "
    "Keep each answer under 120 words, professional and respectful, and always finish your last sentence completely. Output ONLY your final speech as one or two short paragraphs: no planning, no drafts, no word counts, no notes about the instructions."
)


def build_messages(side, system_prompt, context, turns):
    """Build the chat history from one side's point of view.

    Own past turns -> 'assistant', the other side's turns -> 'user'.
    Consecutive messages with the same role are merged.
    """
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": context}]
    for speaker, text in turns:
        role = "assistant" if speaker == side else "user"
        if msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + text
        else:
            msgs.append({"role": role, "content": text})
    return msgs


def ask(model, messages):
    """Returns (text, error). Waits longer on rate limits, then uses the fallback router."""
    attempts = [(model, 0), (model, 10), (FALLBACK_MODEL, 15), (model, 30)]
    last_error = None
    for m, wait in attempts:
        if wait:
            time.sleep(wait)
        try:
            kwargs = {}
            tokens = 700
            if "gpt-oss" in m:  # reasoning model on Groq: keep its thinking short
                kwargs["extra_body"] = {"reasoning_effort": "low"}
                tokens = 1500
            resp = client.chat.completions.create(
                model=m, messages=messages, temperature=0.7, max_tokens=tokens, **kwargs
            )
            text = resp.choices[0].message.content
            if text and text.strip():
                return text.strip(), None
        except Exception as e:  # rate limit (429) / model down / network
            last_error = e
            if any(c in str(e) for c in ('401', '403', 'invalid_api_key')):
                break  # wrong key: retrying will not help
    return None, last_error


def run_debate(country, topic, rounds):
    country = (country or "").strip()
    if not country:
        raise gr.Error("Please enter a country.")
    topic = (topic or "").strip() or "the government's overall economic and social policies"
    rounds = max(1, min(int(rounds), MAX_ROUNDS))  # hard cap at 10

    context = (
        f"Country: {country}\n"
        f"Debate topic: {topic}\n"
        f"The debate has {rounds} rounds. "
        "The government speaks first."
    )

    turns = []   # list of (speaker, text)
    chat = []    # what Gradio displays

    for r in range(1, rounds + 1):
        # --- Government turn ---
        msgs = build_messages("gov", GOV_SYSTEM, context, turns)
        gov_text, err = ask(GOV_MODEL, msgs)
        if err or not gov_text:
            chat.append({"role": "assistant",
                         "content": f"⚠️ Government model failed ({GOV_MODEL}).\n\nReal error: {err}"})
            yield chat
            return
        turns.append(("gov", gov_text))
        chat.append({"role": "assistant",
                     "content": f"🏛️ **Government, Round {r}/{rounds}**\n\n{gov_text}"})
        yield chat

        # --- Opposition turn ---
        msgs = build_messages("opp", OPP_SYSTEM, context, turns)
        opp_text, err = ask(OPP_MODEL, msgs)
        if err or not opp_text:
            chat.append({"role": "user",
                         "content": f"⚠️ Opposition model failed ({OPP_MODEL}).\n\nReal error: {err}"})
            yield chat
            return
        turns.append(("opp", opp_text))
        chat.append({"role": "user",
                     "content": f"🗳️ **Opposition, Round {r}/{rounds}**\n\n{opp_text}"})
        yield chat
        time.sleep(3)  # small pause to respect free-tier rate limits


with gr.Blocks(title="Government vs Opposition Debate") as demo:
    gr.Markdown(
        "# 🏛️ Government vs 🗳️ Opposition\n"
        "Two free LLMs debate each other using data and statistics. Max 10 rounds."
    )
    with gr.Row():
        country_in = gr.Textbox(label="Country", placeholder="e.g. Egypt", scale=2)
        topic_in = gr.Textbox(label="Debate topic (optional)",
                              placeholder="e.g. Inflation and cost of living", scale=3)
        rounds_in = gr.Slider(1, MAX_ROUNDS, value=3, step=1, label="Rounds", scale=2)
    start_btn = gr.Button("Start Debate", variant="primary")
    chatbot = gr.Chatbot(type="messages", height=550, label="Debate")

    start_btn.click(run_debate, [country_in, topic_in, rounds_in], chatbot)

if __name__ == "__main__":
    demo.launch()