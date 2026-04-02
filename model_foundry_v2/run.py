"""Development entrypoint for Model Foundry V2."""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=int(app.config["DEFAULT_PORT"]))
