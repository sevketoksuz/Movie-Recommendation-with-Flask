from flask import Flask

def create_app():
    app = Flask(__name__, template_folder='../templates')
    app.config['SECRET_KEY'] = 'mysecretkey'

    from .routes import main
    app.register_blueprint(main)

    return app