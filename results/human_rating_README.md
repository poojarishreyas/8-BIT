# Blind A/B human rating

`human_rating_template.csv`: 20 Kannada + 20 Kanglish prompts. For each prompt, generate one response from each
model (e.g. original Qwen3 and the bigo-trained model), randomly assign them to A/B, and **keep the A/B key in a
separate file the raters never see**. Each rater fills fluency and correctness (1–5) and a preference. No automation.
