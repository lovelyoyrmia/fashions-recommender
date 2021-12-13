mkdir -p ~/.streamlit/

echo "\
[theme]\n\
base='dark'\n\
primaryColor='#4bff64'\n\
backgroundColor='#484545'\n\
secondaryBackgroundColor='#292828'\n\
[server]\n\
port = $PORT\n\
enableCORS = true\n\
\n\
"> ~/.streamlit/config.toml
