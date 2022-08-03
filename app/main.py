from fastapi import FastAPI
from pysentimiento import create_analyzer


app = FastAPI()
analyzer_sentimient = create_analyzer(task="sentiment", lang="es")
analyzer_sentimient_en = create_analyzer(task="sentiment", lang="en")
emotion_analyzer_en = create_analyzer(task="emotion", lang="en")
emotion_analyzer = create_analyzer(task="emotion", lang="es")

print("---------------- MODULOS LISTOS ---------------- ")


@app.get("/sentimiento/{texto}")
def read_item(texto: str):
    return {"negativo": analyzer_sentimient.predict(texto).probas['NEG'], 
    "neurtal": analyzer_sentimient.predict(texto).probas['NEU'],
    "positivo": analyzer_sentimient.predict(texto).probas['POS']}

@app.get("/sentimient_english/{text}")
def read_item(text: str):
    return {"negativo": analyzer_sentimient_en.predict(text).probas['NEG'], 
    "neurtal": analyzer_sentimient_en.predict(text).probas['NEU'],
    "positivo": analyzer_sentimient_en.predict(text).probas['POS']}


@app.get("/emocion/{texto}")
def read_item(texto: str):
    return {"otros": emotion_analyzer.predict(texto).probas['others'], 
    "tristeza": emotion_analyzer.predict(texto).probas['sadness'],
    "miedo": emotion_analyzer.predict(texto).probas['fear'],
    "enojo": emotion_analyzer.predict(texto).probas['anger'], 
    "disgusto": emotion_analyzer.predict(texto).probas['disgust'],
    "sorpresa": emotion_analyzer.predict(texto).probas['surprise'],
    "alegria": emotion_analyzer.predict(texto).probas['joy']}


@app.get("/emotion_english/{text}")
def read_item(text: str):
    return {"otros": emotion_analyzer_en.predict(text).probas['others'], 
    "tristeza": emotion_analyzer_en.predict(text).probas['sadness'],
    "miedo": emotion_analyzer_en.predict(text).probas['fear'],
    "enojo": emotion_analyzer_en.predict(text).probas['anger'], 
    "disgusto": emotion_analyzer_en.predict(text).probas['disgust'],
    "sorpresa": emotion_analyzer_en.predict(text).probas['surprise'],
    "alegria": emotion_analyzer_en.predict(text).probas['joy']}