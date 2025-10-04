from asgiref.wsgi import AsgiToWsgi  # type: ignore

# Provide the same WSGI `application` callable when importing the package
from app.main import app as fastapi_app

application = AsgiToWsgi(fastapi_app) 