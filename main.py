import json
from pylamp import ChatML
import os
if not os.path.exists("memoria.json"):
    with open("memoria.json", "w") as f:
        json.dump([
            {
                "text": "Oi!",
                "response": "Ola!"
            }
        ], f)
with open("memoria.json", "r") as f:
    

        memoria = json.load(f)
    

def savemem():
    with open("memoria.json", "w") as f:
        json.dump(memoria, f)
def loadmem():
    with open("memoria.json", "r") as f:
        return json.load(f)
if len(memoria) == 0:
    memoria.append([
        {
            "text": "Oi!",
            "response": "Ola!"
        }
    ])
ai = ChatML()
ai.learn(memoria)
mlea = False
while True:
    a = input("Você:")
    if a == "/exit":
        break
    if a == "/learn":
        mlea = True
    if mlea:
        print("Modo de aprendizado ativo.")
        a = input("Você:")
        b = input("AI Response:")
        memoria.append({
            "text": a,
            "response": b
        })
        ai.learn(memoria)
        mlea = False
        savemem()
        memoria = loadmem()
    else:

        print(f"AI: {ai.chat(a)}")