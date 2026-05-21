import json
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# CONFIG
# ==========================================

ARQUIVO_MEMORIA = "pegcat_memoria.json"

EPOCAS = 2000

TAMANHO_OCULTO = 64

# ==========================================
# MEMÓRIA
# ==========================================

def carregar_memoria():

    if os.path.exists(ARQUIVO_MEMORIA):

        with open(
            ARQUIVO_MEMORIA,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return []


def salvar_memoria(dados):

    with open(
        ARQUIVO_MEMORIA,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dados,
            f,
            ensure_ascii=False,
            indent=4
        )


# ==========================================
# DADOS INICIAIS
# ==========================================

dados = carregar_memoria()

if len(dados) == 0:

    dados = [

        {
            "entrada": "oi",
            "resposta": "olá 😎"
        },

        {
            "entrada": "tudo bem",
            "resposta": "sim e você"
        },

        {
            "entrada": "qual seu nome",
            "resposta": "eu sou a PegcatAIT 😼"
        },

        {
            "entrada": "python",
            "resposta": "python é muito legal"
        },

        {
            "entrada": "tchau",
            "resposta": "até mais 👋"
        }

    ]

    salvar_memoria(dados)

# ==========================================
# VOCABULÁRIO
# ==========================================

def criar_vocabulario(dados):

    palavras = []

    for item in dados:

        palavras += item["entrada"].split()

        palavras += item["resposta"].split()

    palavras = sorted(
        list(set(palavras))
    )

    palavra_para_id = {}

    id_para_palavra = {}

    for i, palavra in enumerate(palavras):

        palavra_para_id[palavra] = i

        id_para_palavra[i] = palavra

    return (
        palavras,
        palavra_para_id,
        id_para_palavra
    )


# ==========================================
# TEXTO -> TENSOR
# ==========================================

def texto_para_tensor(
    texto,
    palavra_para_id,
    tamanho_vocab
):

    vetor = torch.zeros(tamanho_vocab)

    for palavra in texto.split():

        if palavra in palavra_para_id:

            indice = palavra_para_id[palavra]

            vetor[indice] = 1

    return vetor


# ==========================================
# MODELO
# ==========================================

class PegcatIA(nn.Module):

    def __init__(
        self,
        entrada,
        oculto,
        saida
    ):

        super().__init__()

        self.rede = nn.Sequential(

            nn.Linear(
                entrada,
                oculto
            ),

            nn.ReLU(),

            nn.Linear(
                oculto,
                oculto
            ),

            nn.ReLU(),

            nn.Linear(
                oculto,
                saida
            )

        )

    def forward(self, x):

        return self.rede(x)


# ==========================================
# TREINAR
# ==========================================

def treinar():

    global modelo
    global palavra_para_id
    global id_para_palavra
    global palavras

    (
        palavras,
        palavra_para_id,
        id_para_palavra
    ) = criar_vocabulario(dados)

    tamanho_vocab = len(palavras)

    x = []

    y = []

    for item in dados:

        entrada = texto_para_tensor(
            item["entrada"],
            palavra_para_id,
            tamanho_vocab
        )

        resposta = item["resposta"].split()[0]

        if resposta in palavra_para_id:

            x.append(entrada)

            y.append(
                palavra_para_id[resposta]
            )

    x = torch.stack(x)

    y = torch.tensor(y)

    modelo = PegcatIA(

        tamanho_vocab,
        TAMANHO_OCULTO,
        tamanho_vocab

    )

    criterio = nn.CrossEntropyLoss()

    otimizador = optim.Adam(

        modelo.parameters(),
        lr=0.01

    )

    print("🧠 Treinando IA...")

    for epoca in range(EPOCAS):

        previsao = modelo(x)

        perda = criterio(
            previsao,
            y
        )

        otimizador.zero_grad()

        perda.backward()

        otimizador.step()

    print("✅ IA treinada!\n")


# ==========================================
# GERAR RESPOSTA
# ==========================================

def responder(texto):

    tamanho_vocab = len(palavras)

    entrada = texto_para_tensor(

        texto,
        palavra_para_id,
        tamanho_vocab

    )

    with torch.no_grad():

        resultado = modelo(entrada)

        indice = torch.argmax(
            resultado
        ).item()

    palavra = id_para_palavra[indice]

    respostas_possiveis = []

    for item in dados:

        if item["resposta"].startswith(
            palavra
        ):

            respostas_possiveis.append(
                item["resposta"]
            )

    if len(respostas_possiveis) > 0:

        return random.choice(
            respostas_possiveis
        )

    return "não sei 😭"


# ==========================================
# APRENDER
# ==========================================

def aprender():

    entrada = input(
        "Nova entrada: "
    )

    resposta = input(
        "Resposta: "
    )

    dados.append({

        "entrada": entrada,
        "resposta": resposta

    })

    salvar_memoria(dados)

    treinar()

    print("🧠 Aprendi!\n")


# ==========================================
# INICIAR
# ==========================================

treinar()

print("😼 PegcatAIT iniciada!")
print("Digite /learn para ensinar")
print("Digite /sair para fechar\n")

# ==========================================
# LOOP
# ==========================================

while True:

    texto = input("Você: ")

    if texto == "/sair":

        break

    if texto == "/learn":

        aprender()

        continue

    resposta = responder(texto)

    print("PegcatAIT:", resposta)