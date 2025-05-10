import logging
from logging.handlers import RotatingFileHandler
from flask import Flask

def create_app():
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static"
    )

    # ——— set up logging ———
    handler = RotatingFileHandler(
        "user_inputs.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=2               # keep 2 old logs
    )
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    # ————————

    from .routes import main
    app.register_blueprint(main)
    return app