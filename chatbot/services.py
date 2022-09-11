
def bot_response_async():
    conversation = ["Hola","¿Cómo estás?","¿En qué puedo ayudarte?","Estaré atento por si necesitas algo"]
    for i in conversation:
        yield i
    return "Iniciar nueva conversación"
