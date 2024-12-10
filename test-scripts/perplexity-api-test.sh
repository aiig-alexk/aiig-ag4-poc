curl --location 'https://api.perplexity.ai/chat/completions' \
--header 'accept: application/json' \
--header 'content-type: application/json' \
--header 'Authorization: Bearer {PPLX-API_KEY}' \
--data '{
  "model": "llama-3.1-sonar-small-128k-online",
  "messages": [
    {
      "role": "system",
      "content": "You are friendly FinTech analyst. You are providing up-to-date information to your clients. Be precise and concise. if you do not know the answer, reply NO-INFO-AVAILABLE."
    },
    {
      "role": "user",
      "content": "What is Argentina GDP in 2023?"
    }
  ]
}'
