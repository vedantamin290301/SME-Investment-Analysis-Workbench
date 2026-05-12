# Cloud Deployment

## Streamlit Community Cloud

Use this option for the easiest free deployment.

1. Create a GitHub repository.
2. Upload these project files and folders:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `DEPLOYMENT.md`
   - `.gitignore`
   - `modules/`
   - `data/`
3. Go to <https://share.streamlit.io>.
4. Click **Create app**.
5. Select the GitHub repository.
6. Set the main file path to:

```text
app.py
```

7. Set Python version to `3.12` if asked.
8. Add your Claude key as a secret, not in code:

```toml
ANTHROPIC_API_KEY = "your_key_here"
OPENAI_API_KEY = "your_openai_key_here"
```

9. Deploy.

## Notes

- Do not upload `.streamlit/secrets.toml`, `.env`, `__pycache__`, or `*.pyc`.
- The Windows `.bat` launchers are for local use only and are ignored by Git.
- If PDF extraction or AI extraction fails in cloud, check that `pdfplumber`, `reportlab`, and `anthropic` installed successfully from `requirements.txt`.
