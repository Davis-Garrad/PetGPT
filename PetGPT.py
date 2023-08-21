import GPT_TTS as brain

session = brain.get_session()
brain.inject_tts("Session created.")

with session:
    brain.inject_tts("Ready")
    brain.inject_gpt("Hello")
    while(True):
        brain.inject_gpt(input())
