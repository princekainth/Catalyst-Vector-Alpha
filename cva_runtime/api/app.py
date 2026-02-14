from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from cva_runtime.api.routes_agents import agents_bp
from cva_runtime.api.routes_health import health_bp
from cva_runtime.api.routes_ops import ops_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(health_bp)
    app.register_blueprint(agents_bp)
    app.register_blueprint(ops_bp)

    return app
