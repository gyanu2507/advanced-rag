web: python3 -m uvicorn backend:app --host 0.0.0.0 --port $PORT
worker: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true

