import json
import keyboard
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# ==========================================
# CONFIG
# ==========================================

ARQUIVO_MEMORIA = "memoria.json"

modo_aprendizado = False

# ==========================================
# MEMÓRIA
# ==========================================

def carregar_memoria():

    try:

        with open(ARQUIVO_MEMORIA, "r", encoding="utf-8") as f:
            return json.load(f)

    except:

        return []


def salvar_memoria(dados):

    with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:

        json.dump(
            dados,
            f,
            ensure_ascii=False,
            indent=4
        )


# ==========================================
# PREPARAR DADOS
# ==========================================

def separar_dados(dados):

    frases = [
        item["texto"]
        for item in dados
    ]

    respostas = [
        item["resposta"]
        for item in dados
    ]

    return frases, respostas


# ==========================================
# IA
# ==========================================

def criar_vectorizer():

    return TfidfVectorizer()


def criar_modelo():

    return SGDClassifier()


# ==========================================
# TREINAR
# ==========================================

def treinar_modelo(
    modelo,
    vectorizer,
    frases,
    respostas
):

    x = vectorizer.fit_transform(frases)

    modelo.fit(x, respostas)


# ==========================================
# PREVER
# ==========================================

def prever(
    modelo,
    vectorizer,
    texto
):

    teste = vectorizer.transform([texto])

    resposta = modelo.predict(teste)[0]

    return resposta


# ==========================================
# APRENDER
# ==========================================

def aprender(
    dados,
    texto,
    resposta
):

    dados.append({

        "texto": texto,
        "resposta": resposta

    })

    salvar_memoria(dados)


# ==========================================
# TOGGLE APRENDIZADO
# ==========================================

def alternar_modo():

    global modo_aprendizado

    modo_aprendizado = not modo_aprendizado

    if modo_aprendizado:

        print("\n🧠 Modo aprendizado: ON\n")

    else:

        print("\n💬 Modo conversa: ON\n")
def modoa():
    texto = input("Você: ")

    try:
        resposta = prever(
            modelo,
            vectorizer,
            texto
        )

        print("IA:", resposta)

    except:

        print("IA: Não sei responder isso ainda 😭")

keyboard.add_hotkey(
    "alt+l",
    alternar_modo
)

# ==========================================
# INICIAR MEMÓRIA
# ==========================================

dados = carregar_memoria()

if len(dados) == 0:

    dados = [

        {
            "texto": "oi",
            "resposta": "Olá 😎"
        },

        {
            "texto": "tchau",
            "resposta": "Até mais 👋"
        },

        {
            "texto": "qual seu nome",
            "resposta": "Sou uma IA em Python 😎"
        }

    ]

    salvar_memoria(dados)

# ==========================================
# CRIAR IA
# ==========================================

frases, respostas = separar_dados(dados)

vectorizer = criar_vectorizer()

modelo = criar_modelo()

treinar_modelo(
    modelo,
    vectorizer,
    frases,
    respostas
)

# ==========================================
# LOOP
# ==========================================

print("|----------|")
print("|* Pegcat *|")
print("|----------|")

while True:

    texto = input("Você: ")

    # =========================
    # COMANDOS
    # =========================

    if texto == "/learn":

        modo_aprendizado = True

        print("🧠 Modo aprendizado ON")

        continue
    if texto == "/exit":
        os._exit(0)
    elif texto == "/chat":

        modo_aprendizado = False

        print("💬 Modo conversa ON")

        continue

    # =========================
    # APRENDIZADO
    # =========================

    if modo_aprendizado:

        resposta_correta = input(
            "Resposta correta: "
        )

        aprender(
            dados,
            texto,
            resposta_correta
        )

        frases, respostas = separar_dados(dados)

        treinar_modelo(
            modelo,
            vectorizer,
            frases,
            respostas
        )

        print("🧠 Aprendi!")

    # =========================
    # CONVERSA
    # =========================

    else:

        try:

            resposta = prever(
                modelo,
                vectorizer,
                texto
            )

            print("IA:", resposta)

        except:

            print("IA: Não sei responder 😭")