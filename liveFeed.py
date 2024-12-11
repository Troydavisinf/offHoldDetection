from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, Request
import ollama

app = FastAPI()
templates = Jinja2Templates(directory="templates")

#Creating endpoint for simple front-end application using html to play with ollama
@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
	# Render the HTML template
	return templates.TemplateResponse("index.html", {"request" : request})

#Websocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        response = await websocket.receive_text()
        # Condition if user enters an empty string
        if(len(response)==0):
             await websocket.send_text("Empty Text not acceptable. Please write something in the Text Box.")
        else:
            #sending the response recieved from user to llama2 model
            data_from_llama = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': """The following is a live transcription of a phone call. It is your job to determine whether or not """ +
                                                  """"someone is on the other end of the phone and therefore should not be on hold any longer. Please remember """ +
                                                  """that the transcription could be empty (only .'s), meaning the call should still be on hold. Please only say yes or no given the following transcription: """+ response}],
            stream=True,
            )
            # Sending the data word by word to Front-end application

            fullMessage = ""

            for chunk in data_from_llama:
                await websocket.send_text(chunk['message']['content'])
                fullMessage+=chunk['message']['content']

            print(response + ": " + fullMessage)