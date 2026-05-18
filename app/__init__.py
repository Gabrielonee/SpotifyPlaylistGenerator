import logging
from flask import Flask
from app.config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    logging.basicConfig(
        level=getattr(logging, config_class.LOG_LEVEL, logging.INFO),
        format='%(asctime)s %(name)s %(levelname)s %(message)s'
    )

    from app.controllers.routes import main_bp
    app.register_blueprint(main_bp)

    return app
