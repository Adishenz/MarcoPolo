mkdir -p ~/.streamlit/

echo "[theme]
primaryColor = ‘#0a1931’
backgroundColor = ‘#f1ecc3’
secondaryBackgroundColor = ‘#3f5669’
textColor= ‘#fe7171’
font = ‘sans serif’
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
