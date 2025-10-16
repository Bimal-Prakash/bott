# dynamicbot_no_wake.py
import ollama
import pyttsx3
import speech_recognition as sr
import time
import datetime
import os
from dotenv import load_dotenv

# ====== Real time date/time helpers ======
def get_current_date():
    today = datetime.date.today()
    return f"Today's date is {today.strftime('%B %d, %Y')}."

def get_current_day():
    today = datetime.date.today()
    return f"Today is {today.strftime('%A')}."

def get_current_time():
    now = datetime.datetime.now()
    return f"The current time is {now.strftime('%I:%M %p')}."

# ====== Voice Bot TTS / STT setup ======
engine = pyttsx3.init()
engine.setProperty("rate", 180)
engine.setProperty("volume", 1.0)

recognizer = sr.Recognizer()
mic = sr.Microphone()

def speak(text):
    """Speak text via TTS and also print to console."""
    print("🤖 Bot:", text)
    engine.say(text)
    engine.runAndWait()

def listen(timeout=5, phrase_time_limit=6):
    """Listen via microphone and return lowercase text, or None."""
    with mic as source:
        print(" Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.4)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return None
    try:
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        speak("Speech recognition service is unavailable.")
        return None

# ====== Ollama chat helper ======
def chat_with_bot(prompt):
    """Query Ollama streaming API and return full response text."""
    response_text = ""
    stream = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            response_text += chunk["message"]["content"]
    return response_text.strip()

def summarize_response(response):
    """Return up to three short bullet points from the model response."""
    sentences = response.replace("\n", " ").split(". ")
    summary_points = sentences[:3]
    return "\n".join(f"- {s.strip()}" for s in summary_points if s.strip())

# ====== Main bot loop ======
def bot_main():
    """Main voice-interaction loop (no wake word)."""
    speak("Hello! I'm your assistant. How may I help you?")
    try:
        while True:
            user_input = listen()
            if not user_input:
                speak("Sorry, I didn't catch that.")
                continue

            user_input = user_input.strip()
            print("You:", user_input)

            # Exit commands
            if any(cmd in user_input for cmd in ["exit", "quit", "stop", "goodbye", "bye"]):
                speak("Goodbye!")
                break

            # Real-time simple queries
            if "date" in user_input:
                speak(get_current_date())
                continue
            if "day" in user_input:
                speak(get_current_day())
                continue
            if "time" in user_input:
                speak(get_current_time())
                continue

            # Forward to Ollama
            bot_response = chat_with_bot(user_input)

            # If user asked for detail, speak full response; otherwise give a short summary
            if any(word in user_input for word in ["detail", "detailed", "full", "explain", "tell me more"]):
                speak(bot_response)
            else:
                speak(summarize_response(bot_response))

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting bot loop.")
    except Exception as e:
        print("Bot encountered an error:", e)
        speak("I encountered an error. Check the console for details.")

if __name__ == "__main__":
    # load environment if present (optional)
    load_dotenv()
    # Run the bot immediately (no wake word)
    bot_main()
