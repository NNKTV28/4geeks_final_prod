#!/bin/bash
streamlit run src/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
