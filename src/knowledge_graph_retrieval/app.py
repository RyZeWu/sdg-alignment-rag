from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, List, Dict, Any
from src.QA_integration import QA_RAG
from langchain_neo4j import Neo4jGraph
import logging
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Models ----------------

class QAResponse(BaseModel):
    status: str
    data: Dict[str, Any]

# ---------------- Defaults ----------------

DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI")
DEFAULT_NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DEFAULT_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

def create_graph_database_connection(uri, userName, password, database):
    try:
        return Neo4jGraph(
            url=uri,
            database=database,
            username=userName,
            password=password,
            refresh_schema=False,
            sanitize=True
        )
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to Neo4j database.")

# ---------------- Error Handlers ----------------

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."}
    )

# ---------------- Endpoints ----------------
@app.post("/chat_bot", response_model=QAResponse)
def qa_form_endpoint(
    question: str = Form(...),
    model: str = Form("gpt-4o"),
    session_id: str = Form("default_session"),
    mode: str = Form("graph_of_thought"),
    write_access: bool = Form(True),
    uri: Optional[str] = Form(None),
    userName: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    evaluation: Optional[bool] = Form(False)
):
    try:
        neo4j_uri = uri or DEFAULT_NEO4J_URI
        neo4j_username = userName or DEFAULT_NEO4J_USERNAME
        neo4j_password = password or DEFAULT_NEO4J_PASSWORD
        neo4j_database = database or DEFAULT_NEO4J_DATABASE

        graph = create_graph_database_connection(neo4j_uri, neo4j_username, neo4j_password, neo4j_database)

        result = QA_RAG(
            graph=graph,
            model=model,
            question=question,
            document_names=None,
            session_id=session_id,
            mode=mode,
            write_access=write_access,
            evaluation=evaluation
        )

        response = { "status": "success", "data": result }

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /qa-form endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "GraphRAG FastAPI is running."}
