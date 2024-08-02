from openai import OpenAI
from google.colab import userdata
import requests

# Setup client and helper method.
open_ai_key = userdata.get('open_ai_key')
client = OpenAI(api_key=open_ai_key)

def ask(context, message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": message}
        ]
    )

    text = response.choices[0].message.content
    return text

# Summarize a CNBC article's opening excerpts (https://www.cnbc.com/2024/07/17/stock-market-today-live-updates.html).
res = ask(
    "You are an LLM.",
    """
    Generate a single sentence summarizing the text below in an understandable format.

    Article excerpts:

    Stocks fell on Thursday, as the technology sector continued struggling amid the market’s rotation on hopes of easing monetary policy.
    The technology-heavy Nasdaq Composite lost 1.2%. The S&P 500dropped 0.8%. The Dow Jones Industrial Average slid 337 points, or 0.8%.
    The Nasdaq’s underperformance marks a continuation of the broader shift away from tech seen in recent days. Wall Street has dumped
    shares of artificial intelligence plays as the growing likelihood of a September interest rate cut from the Federal Reserve bolstered
    optimism in the broader market. On the other hand, that’s largely helped small-cap and more cyclical names, which are seen as bigger
    beneficiaries of lower borrowing costs.
    """
);
print(res)

# Compare 2 documents.

randomJokesEndpoint = "https://official-joke-api.appspot.com/random_joke";
def getRandomJoke():
    res = requests.get(randomJokesEndpoint)
    data = res.json()

    return data["setup"] + " " + data["punchline"]

input = f"""
Document 1:
{getRandomJoke()}

Document 2:
{getRandomJoke()}
"""
res = ask(
  "You are an LLM.",
  f"""
  Analyze and compare the performance of the 2 documents below in generating humorous content.
  Determine which is more funny.
  Be as objective as possible, and ensure that 1 joke is designated as \"more funny\" than the other.

  {input}
  """
)

print(f"{input}\nResponse:\n{res}")