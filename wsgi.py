from asgiref.wsgi import AsgiToWsgi  # type: ignore

from app.main import app as fastapi_app

application = AsgiToWsgi(fastapi_app) 