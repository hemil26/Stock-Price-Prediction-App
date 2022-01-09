mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"hemilmehta26@gmail.com\"\n\
" > ~/.streamlit/credentials.toml


echo "[theme]
primaryColor='#14207e'
backgroundColor='#d6eff6'
secondaryBackgroundColor='#fff9f9'
textColor='#000000'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml