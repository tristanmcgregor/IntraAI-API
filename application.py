from asgiref.wsgi import AsgiToWsgi  # type: ignore

try:
	# Import the ASGI FastAPI app
	from app.main import app as fastapi_app
except Exception as e:
	# Fallback to a minimal app to avoid crashing import (will still fail later if real app missing)
	from fastapi import FastAPI
	fastapi_app = FastAPI()

# Expose a WSGI-compatible callable named `application`
# Beanstalk may default to gunicorn application:application
application = AsgiToWsgi(fastapi_app) 