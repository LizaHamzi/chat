from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import pinecone
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import os
from pinecone import Pinecone as PineconeClient
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import json
import uuid
import subprocess
from transformers import AutoTokenizer, AutoModel
from fastapi import HTTPException

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")  


load_dotenv()


api_key = os.environ.get("PINECONE_API_KEY")


pc = PineconeClient(api_key=api_key)


index_name = "rcwprj"
host = "https://rcwprj-bsy7yur.svc.aped-4627-b74a.pinecone.io"


pinecone_index = pc.Index(index_name=index_name, host=host)
print(f"Index {index_name} prêt à l'utilisation.")


OLLAMA_PATH = r"C:\Users\aziz\AppData\Local\Programs\Ollama\ollama.exe"


class Question(BaseModel):
    question: str
    document_id: str = None  


# Fonctions Utilitaires
def extract_text(file_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    content = "".join([page.extract_text() + "\n" for page in reader.pages])
    print(f"Longueur totale du texte extrait : {len(content)} caractères")
    return content


def split_into_chunks(content, chunk_size=600):
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    print(f"Chunks générés ({len(chunks)}): {chunks[:2]}...")  # Afficher les premiers chunks pour vérification
    return chunks



def generate_embeddings(chunks):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        embeddings.append(embedding)
    return embeddings



def generate_embedding(question):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embedding.tolist()


def query_ollama(question, context):
    try:
        prompt = f"""
        Context:
        {context}

        Question:
        {question}

        Please answer based on the provided context only. If the context does not contain the answer, reply: 'Sorry, I have no clue.'
        """
        print(f"Prompt envoyé à Ollama : {prompt}")

        result = subprocess.run(
            [OLLAMA_PATH, "run", "llama3.2"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            print(f"Erreur lors de l'exécution d'Ollama : {result.stderr.decode('utf-8')}")
            return "Sorry, I have no clue."

        response = result.stdout.decode("utf-8")
        print(f"Réponse brute d'Ollama : {response}")

        try:
            response_json = json.loads(response)
            return response_json.get("completion", "Sorry, I have no clue.")
        except json.JSONDecodeError:
            return response.strip()
    except Exception as e:
        print(f"Erreur lors de l'appel à Ollama : {str(e)}")
        return "Sorry, I have no clue."




last_uploaded_document_id = None  # Variable globale pour stocker le dernier document ID

@app.post("/upload/")
async def upload_file(file: UploadFile):
    global last_uploaded_document_id  # Utiliser une variable globale
    document_id = str(uuid.uuid4())  # Générer un identifiant unique

    temp_path = f"temp_files/{file.filename}"
    os.makedirs("temp_files", exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(await file.read())
    print(f"Fichier {file.filename} sauvegardé temporairement.")

    # Extraction du texte
    content = extract_text(temp_path)
    os.remove(temp_path)

    # Découpage en chunks
    chunks = split_into_chunks(content)

    # Génération des embeddings
    embeddings = generate_embeddings(chunks)

    # Stockage dans Pinecone
    vectors = [
        (f"{document_id}-{i}", embedding, {"chunk": chunk, "document_id": document_id})
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    pinecone_index.upsert(vectors)
    print(f"Embeddings insérés dans Pinecone : {len(vectors)} vecteurs")

    # Vérification via métadonnées
    try:
        test_results = pinecone_index.query(
            vector=[0] * len(embeddings[0]),  
            top_k=1,
            filter={"document_id": document_id},
            include_metadata=True
        )
        print(f"Vérification des vecteurs insérés pour document_id '{document_id}': {test_results}")
    except pinecone.core.exceptions.PineconeException as e:
        print(f"Erreur lors de la vérification des vecteurs : {e}")

    # Mettre à jour le dernier document téléversé
    last_uploaded_document_id = document_id
    print(f"Dernier document ID enregistré : {last_uploaded_document_id}")

    return JSONResponse({
        "message": "File processed successfully",
        "document_id": document_id,
        "chunks": len(chunks)
    })



@app.post("/ask-question/")
async def ask_question(data: Question):
    global last_uploaded_document_id  # Dernier document téléversé

    # Utiliser le dernier document téléversé si `document_id` est manquant
    document_id = data.document_id or last_uploaded_document_id
    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="Aucun document n'a été téléversé. Veuillez téléverser un document ou spécifier un document_id."
        )

    try:
        print(f"Question reçue : {data.question}")
        print(f"Utilisation du document ID : {document_id}")

        # Génération de l'embedding de la question
        question_embedding = generate_embedding(data.question)

        # Rechercher les chunks dans Pinecone
        results = pinecone_index.query(
            vector=question_embedding,
            top_k=1,
            include_metadata=True,
            filter={"document_id": document_id}
        )
        print(f"Résultats bruts de Pinecone : {results}")
        if not results.matches:
            return {"response": "Aucune correspondance trouvée pour ce document. Vérifiez le contenu téléversé."}

        
        # Préparer le chunk récupéré
        relevant_chunk = (
    results.matches[0].metadata.get("chunk") if results.matches and results.matches[0].metadata.get("chunk") else ""
)


        # Appeler Ollama
        response = query_ollama(data.question, relevant_chunk)
        print(f"Réponse obtenue : {response}")
        return {"response": response}

    except Exception as e:
        print(f"Erreur interne : {str(e)}")
        return JSONResponse({"response": f"Erreur lors de la recherche dans le PDF : {str(e)}"}, status_code=500)



@app.delete("/clean-vectors/")
async def clean_vectors(document_id: str):
    try:
        pinecone_index.delete(filter={"document_id": document_id})
        print(f"Vecteurs supprimés pour document_id : {document_id}")
        return JSONResponse({"message": f"Vectors for document_id '{document_id}' have been deleted."})
    except Exception as e:
        print(f"Erreur lors de la suppression des vecteurs : {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)